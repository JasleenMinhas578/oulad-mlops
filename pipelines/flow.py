"""One pipeline run processes one new batch: ingest, monitor, decide, retrain, gate, promote."""
from __future__ import annotations

import argparse
import os

import pandas as pd
import requests
from prefect import flow, get_run_logger, task

from oulad import registry, storage
from oulad.config import Settings
from oulad.data import KEYS, local_bronze
from oulad.drift import detect_drift, evidently_report_html, simulate_engagement_drop
from oulad.evaluate import (
    classification_metrics,
    fairness_audit,
    flatten_fairness,
    passes_fairness,
)
from oulad.features import AUDIT_COLUMNS, LABEL, build_features
from oulad.serving import ModelBundle
from oulad.stream import batch_schedule, load_state, save_state, state_uri
from oulad.train import DEFAULT_PARAMS, train_model

DRIFT_EXCLUDE = {"code_module"}  # one module per batch always looks drifted (Phase 5.5)


def _uri(s: Settings, *parts: str) -> str:
    return storage.join(s.data_uri, *parts)


def notify_deploy(s: Settings, version: str) -> None:
    """Fire a GitHub repository_dispatch event so the deploy workflow restarts the API."""
    log = get_run_logger()
    if not (s.github_repo and s.github_token):
        log.info("No GitHub token configured; skipping the redeploy trigger")
        return
    resp = requests.post(
        f"https://api.github.com/repos/{s.github_repo}/dispatches",
        headers={"Authorization": f"Bearer {s.github_token}",
                 "Accept": "application/vnd.github+json"},
        json={"event_type": "model-promoted", "client_payload": {"version": str(version)}},
        timeout=15,
    )
    resp.raise_for_status()
    log.info("Sent model-promoted event for version %s", version)


@task(retries=2, retry_delay_seconds=10)
def ingest(s: Settings, module: str, term: str, index: int, batch_id: str) -> pd.DataFrame:
    df = build_features(local_bronze(s.data_uri), s.cutoff_day, [(module, term)])
    if os.getenv("INJECT_DRIFT_AT") == str(index):
        get_run_logger().warning("Injecting a synthetic engagement drop into %s", batch_id)
        df = simulate_engagement_drop(df)
    storage.write_parquet(df, _uri(s, "processed", f"{batch_id}.parquet"))
    return df


@task
def score_champion(
    s: Settings, champion: ModelBundle, current: pd.DataFrame, batch_id: str
) -> tuple[dict, dict]:
    proba = champion.predict_proba(current)
    preds = current[KEYS].copy()
    preds["risk_score"] = proba
    preds["flagged"] = (proba >= champion.threshold).astype(int)
    preds["model_version"] = champion.version
    storage.write_parquet(preds, _uri(s, "predictions", f"{batch_id}.parquet"))
    metrics = classification_metrics(current[LABEL], proba, champion.threshold)
    audit = fairness_audit(current[LABEL], proba, champion.threshold, current["disability"])
    return metrics, audit


@task
def check_drift(
    s: Settings,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    batch_id: str,
) -> dict:
    cats = [c for c in categorical if c not in DRIFT_EXCLUDE]
    report = detect_drift(reference, current, numeric, cats, s.psi_threshold)
    report["audit_psi"] = detect_drift(reference, current, [], AUDIT_COLUMNS, s.psi_threshold)["psi"]
    storage.write_json(report, _uri(s, "reports", batch_id, "drift.json"))
    html = evidently_report_html(reference[numeric + cats], current[numeric + cats])
    if html:
        storage.write_bytes(_uri(s, "reports", batch_id, "evidently.html"), html)
    return report


@task
def retrain_and_gate(
    s: Settings,
    state: dict,
    schedule: list[tuple[str, str]],
    index: int,
    current: pd.DataFrame,
    champ_metrics: dict,
    batch_id: str,
) -> dict:
    log = get_run_logger()
    # Expanding window: what the champion learned from + every batch since, except this one.
    parts = [storage.read_parquet(_uri(s, "reference", "features.parquet"))]
    for j in range(state.get("trained_through", -1) + 1, index):
        m, t = schedule[j]
        parts.append(storage.read_parquet(_uri(s, "processed", f"{j:02d}_{m}_{t}.parquet")))
    train_df = pd.concat(parts, ignore_index=True)

    trained, _ = train_model(train_df, s.include_demographics, s.target_recall)
    proba = trained.predict_proba(current)
    cand = classification_metrics(current[LABEL], proba, trained.threshold)
    audit = fairness_audit(current[LABEL], proba, trained.threshold, current["disability"])

    beats = bool(cand["roc_auc"] >= champ_metrics["roc_auc"])  # False if either is NaN
    fair = passes_fairness(audit, s.fairness_gap_max)
    promote = beats and fair

    client = registry.setup(s)
    version = registry.log_run(
        trained,
        params={**DEFAULT_PARAMS, "cutoff_day": s.cutoff_day, "train_rows": len(train_df)},
        metrics={**{f"batch_{k}": v for k, v in cand.items()}, **flatten_fairness(audit),
                 "champion_batch_auc": champ_metrics["roc_auc"]},
        audit=audit,
        tags={"reason": "retrain", "batch": batch_id, "promoted": str(promote)},
        model_name=s.model_name,
    )
    state_update: dict = {}
    if promote:
        registry.promote(client, s.model_name, version)
        registry.export_bundle(client, s.model_name, version, storage.join(s.models_uri, f"v{version}"))
        registry.export_bundle(client, s.model_name, version, storage.join(s.models_uri, "champion"))
        storage.write_parquet(train_df, _uri(s, "reference", "features.parquet"))
        state_update = {"champion_version": str(version), "baseline_auc": cand["roc_auc"],
                        "trained_through": index - 1}
        notify_deploy(s, version)
    else:
        client.set_registered_model_alias(s.model_name, "challenger", str(version))
    log.info("candidate v%s auc=%.3f gap=%.3f beats=%s fair=%s promoted=%s",
             version, cand["roc_auc"], audit["recall_gap"], beats, fair, promote)
    return {"candidate_version": version, "candidate_auc": cand["roc_auc"],
            "candidate_recall_gap": audit["recall_gap"], "beats_champion": beats,
            "passes_fairness": fair, "promoted": promote, "state_update": state_update}


@flow(name="oulad-batch-pipeline")
def run_next_batch() -> dict:
    s = Settings.from_env()
    log = get_run_logger()
    if not storage.exists(state_uri(s)):
        # First run on a fresh environment (e.g. the AWS cluster): create the champion.
        log.info("No stream state found: bootstrapping the first champion")
        from oulad.bootstrap import main as bootstrap

        bootstrap()
        return {"status": "bootstrapped"}
    state = load_state(s)
    schedule = batch_schedule(local_bronze(s.data_uri))
    i = state["next_index"]
    if i >= len(schedule):
        log.info("All %d batches processed. The stream is finished.", len(schedule))
        return {"status": "finished"}

    module, term = schedule[i]
    batch_id = f"{i:02d}_{module}_{term}"
    log.info("Processing batch %s", batch_id)

    current = ingest(s, module, term, i, batch_id)
    reference = storage.read_parquet(_uri(s, "reference", "features.parquet"))
    champion = ModelBundle.load(storage.join(s.models_uri, "champion"))
    numeric, categorical = champion.metadata["numeric"], champion.metadata["categorical"]

    champ_metrics, champ_audit = score_champion(s, champion, current, batch_id)
    drift = check_drift(s, reference, current, numeric, categorical, batch_id)

    drift_trigger = drift["drift_share"] >= s.drift_share_threshold
    perf_trigger = bool(state["baseline_auc"] - champ_metrics["roc_auc"] >= s.auc_drop_threshold)
    record = {
        "batch": batch_id, "rows": len(current),
        "drift_share": round(drift["drift_share"], 3), "drifted": drift["drifted"],
        "champion_version": champion.version, "champion_auc": champ_metrics["roc_auc"],
        "champion_recall": champ_metrics["recall"],
        "champion_recall_gap": champ_audit["recall_gap"],
        "drift_trigger": drift_trigger, "perf_trigger": perf_trigger,
    }
    if drift_trigger or perf_trigger:
        outcome = retrain_and_gate(s, state, schedule, i, current, champ_metrics, batch_id)
        state.update(outcome.pop("state_update"))
        record.update(outcome)

    state["history"].append(record)
    state["next_index"] = i + 1
    save_state(s, state)
    log.info("Batch result: %s", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="run on a schedule instead of once")
    parser.add_argument("--cron", default="*/10 * * * *")
    args = parser.parse_args()
    if args.serve:
        run_next_batch.serve(name="oulad-stream", cron=args.cron)
    else:
        run_next_batch()
