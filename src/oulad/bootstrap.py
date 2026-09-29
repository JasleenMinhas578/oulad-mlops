"""Train the first model on 2013, register it as champion, export it, initialise the stream."""
from __future__ import annotations

from oulad import registry, storage
from oulad.config import Settings
from oulad.data import TRAIN_TERMS, local_bronze
from oulad.evaluate import classification_metrics, fairness_audit, flatten_fairness
from oulad.features import LABEL, build_features
from oulad.train import DEFAULT_PARAMS, train_model


def main() -> None:
    s = Settings.from_env()
    df = build_features(local_bronze(s.data_uri), s.cutoff_day)
    train_df = df[df["code_presentation"].isin(TRAIN_TERMS)].reset_index(drop=True)

    trained, va = train_model(train_df, s.include_demographics, s.target_recall)
    metrics = classification_metrics(va[LABEL], va["proba"], trained.threshold)
    audit = fairness_audit(va[LABEL], va["proba"], trained.threshold, va["disability"])

    client = registry.setup(s)
    version = registry.log_run(
        trained,
        params={**DEFAULT_PARAMS, "cutoff_day": s.cutoff_day, "train_rows": len(train_df)},
        metrics={**{f"val_{k}": v for k, v in metrics.items()}, **flatten_fairness(audit)},
        audit=audit,
        tags={"reason": "bootstrap", "train_terms": ",".join(TRAIN_TERMS)},
        model_name=s.model_name,
    )
    registry.promote(client, s.model_name, version)
    registry.export_bundle(client, s.model_name, version, storage.join(s.models_uri, f"v{version}"))
    registry.export_bundle(client, s.model_name, version, storage.join(s.models_uri, "champion"))

    # The drift monitor compares every future batch with the data the champion learned from.
    storage.write_parquet(train_df, storage.join(s.data_uri, "reference", "features.parquet"))
    storage.write_json(
        {"next_index": 0, "champion_version": str(version),
         "baseline_auc": metrics["roc_auc"], "history": []},
        storage.join(s.data_uri, "state", "stream_state.json"),
    )
    print(f"champion v{version}: AUC={metrics['roc_auc']:.3f} recall={metrics['recall']:.3f} "
          f"recall_gap={audit['recall_gap']:.3f} groups={audit['group_sizes']}")


if __name__ == "__main__":
    main()
