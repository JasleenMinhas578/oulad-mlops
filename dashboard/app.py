"""Read-only dashboard: the workflow, live predictions with explanations, training history,
drift and fairness. Reads the same state, reports and model bundle the pipeline writes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from oulad import storage
from oulad.config import Settings
from oulad.evaluate import fairness_audit
from oulad.serving import ModelBundle, prepare_frame
from oulad.stream import load_state

LABEL = "at_risk"
S = Settings.from_env()

st.set_page_config(page_title="OULAD early warning", layout="wide")


@st.cache_data(ttl=60)
def get_state() -> dict:
    return load_state(S)


@st.cache_resource(ttl=300)
def get_champion() -> ModelBundle:
    return ModelBundle.load(storage.join(S.models_uri, "champion"))


@st.cache_data(ttl=300)
def get_batch(batch_id: str) -> pd.DataFrame:
    return storage.read_parquet(storage.join(S.data_uri, "processed", f"{batch_id}.parquet"))


@st.cache_data(ttl=300)
def get_drift(batch_id: str) -> dict:
    return storage.read_json(storage.join(S.data_uri, "reports", batch_id, "drift.json"))


def score(bundle: ModelBundle, df: pd.DataFrame) -> pd.Series:
    return pd.Series(bundle.predict_proba(df), index=df.index)


def explain(bundle: ModelBundle, row: pd.DataFrame) -> pd.Series:
    """Per-feature push on the score in log-odds (LightGBM's built-in contributions)."""
    meta = bundle.metadata
    X = prepare_frame(row, meta["numeric"], meta["categorical"], meta["categories"])
    contrib = bundle.booster.predict(X, pred_contrib=True)[0]
    return pd.Series(contrib[:-1], index=bundle.feature_names)  # last value is the bias


def outcome(rec: dict) -> str:
    if "promoted" not in rec:
        return "No retrain"
    if rec["promoted"]:
        return "Promoted"
    if rec.get("beats_champion") and not rec.get("passes_fairness"):
        return "Blocked by fairness gate"
    return "Not better than champion"


WORKFLOW = """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#dbe7f5", fontcolor="#111", color="#5b7fa6"];
  edge [color="#8899aa"];
  batch [label="New batch\\n(one course, day 28)"];
  feats [label="Features\\n(DuckDB, only events\\nbefore day 28)"];
  score [label="Score with\\nchampion"];
  drift [label="Drift check\\n(PSI per feature)"];
  keep [label="Keep champion", fillcolor="#e6efe0"];
  train [label="Retrain a\\nchallenger"];
  gate [label="Gate: AUC >= champion\\nAND recall gap <= 0.10", fillcolor="#f7e3b5"];
  promote [label="Promote to champion\\n(MLflow alias + S3)", fillcolor="#cfe8cf"];
  park [label="Park as challenger", fillcolor="#f2cfcf"];
  deploy [label="Deploy workflow\\nrestarts the API"];
  batch -> feats -> score -> drift;
  drift -> keep [label=" no drift"];
  drift -> train [label=" drift or\\n AUC drop"];
  train -> gate;
  gate -> promote [label=" pass"];
  gate -> park [label=" fail", style=dashed];
  promote -> deploy;
}
"""

try:
    state = get_state()
    bundle = get_champion()
except Exception as exc:  # noqa: BLE001  show a readable message instead of a stack trace
    st.error(f"Pipeline output not found yet: {exc}")
    st.stop()

history = pd.DataFrame(state["history"])
history["outcome"] = [outcome(r) for r in state["history"]]

st.title("OULAD early warning: live model dashboard")
st.caption(
    "Flags learners at risk of failing or withdrawing at day 28 of a course. "
    "Disability is never a model input; it is only used to audit fairness."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Champion version", bundle.version)
c2.metric("Batches processed", f"{state['next_index']} of 13")
c3.metric("Promotions", int((history["outcome"] == "Promoted").sum()))
c4.metric("Blocked by fairness gate", int((history["outcome"] == "Blocked by fairness gate").sum()))

tab_flow, tab_pred, tab_train, tab_drift = st.tabs(
    ["How it works", "Live predictions", "Training history", "Drift and fairness"]
)

with tab_flow:
    st.subheader("The pipeline, one batch at a time")
    st.graphviz_chart(WORKFLOW, width="stretch")
    st.markdown(
        "- **Data:** 32,593 student enrolments and 10.6M daily click records (Open University, 2013-2014).\n"
        "- **Training:** the first model learns from 2013. 2014 is replayed as a live stream, "
        "one course presentation per run.\n"
        "- **Retraining:** triggered when at least 30% of features drift, or AUC drops 0.05 below baseline.\n"
        "- **Safety:** a new model ships only if it beats the champion on unseen data "
        "and keeps the recall gap between disability groups at 0.10 or less."
    )

with tab_pred:
    st.subheader("What the model predicts, and why")
    batch_id = st.selectbox("Course batch", history["batch"].tolist(), key="pred_batch")
    df = get_batch(batch_id).reset_index(drop=True)
    df["risk_score"] = score(bundle, df)
    df["flagged"] = df["risk_score"] >= bundle.threshold
    a, b, c = st.columns(3)
    a.metric("Students in batch", len(df))
    b.metric("Flagged for outreach", int(df["flagged"].sum()))
    c.metric("Decision threshold", f"{bundle.threshold:.3f}")

    top = df.sort_values("risk_score", ascending=False).head(50)
    student = st.selectbox(
        "Pick a student (highest risk first)",
        top["id_student"].tolist(),
        format_func=lambda i: f"Student {i}  (risk {top.loc[top['id_student'] == i, 'risk_score'].iloc[0]:.2f})",
    )
    row = df[df["id_student"] == student].head(1)
    r = row.iloc[0]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Risk score", f"{r['risk_score']:.2f}")
        st.write("**Flagged for outreach**" if r["flagged"] else "Not flagged")
        st.caption(f"Actual outcome in the dataset: {r['final_result']}")
        st.write(
            f"Clicks before day 28: {int(r['clicks_total'])}  \n"
            f"Days since last active: {r['days_since_last_active']}  \n"
            f"Assessments missed: {int(r['n_missed'])}"
        )
    with right:
        contrib = explain(bundle, row)
        shown = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index)
        st.write("**What pushed this score up (positive) or down (negative)**")
        st.bar_chart(shown.rename("push on risk (log-odds)"))
    st.write("**Highest-risk students in this batch**")
    st.dataframe(
        top[["id_student", "risk_score", "flagged", "clicks_total", "n_missed", "final_result"]]
        .head(15)
        .reset_index(drop=True),
        width="stretch",
    )

with tab_train:
    st.subheader("How the model was trained and retrained across the 2014 stream")
    st.caption("Each row is one batch. A retrained challenger must win on that batch to replace the champion.")
    chart = history.set_index("batch")[["champion_auc"]].copy()
    if "candidate_auc" in history:
        chart["challenger_auc"] = history.set_index("batch")["candidate_auc"]
    st.line_chart(chart)
    cols = ["batch", "rows", "drift_share", "champion_version", "champion_auc",
            "candidate_auc", "candidate_recall_gap", "outcome"]
    st.dataframe(history[[c for c in cols if c in history]], width="stretch")
    st.info(
        "Rows marked 'Blocked by fairness gate' had a challenger with higher AUC that was "
        "rejected because it would have widened the recall gap between disability groups."
    )

with tab_drift:
    st.subheader("Has student behaviour changed, and is the model fair?")
    d_batch = st.selectbox("Batch", history["batch"].tolist(), key="drift_batch")
    left, right = st.columns(2)
    with left:
        st.write("**Drift score (PSI) per feature**, above 0.2 counts as drifted")
        psi = pd.Series(get_drift(d_batch)["psi"]).sort_values(ascending=False).head(15)
        st.bar_chart(psi.rename("PSI"))
    with right:
        st.write("**Recall by disability group** (share of truly at-risk learners caught)")
        bdf = get_batch(d_batch).reset_index(drop=True)
        audit = fairness_audit(bdf[LABEL], score(bundle, bdf), bundle.threshold, bdf["disability"])
        groups = pd.DataFrame(audit["by_group"]).T
        groups["group_size"] = pd.Series(audit["group_sizes"])
        groups.index = groups.index.map({"Y": "Disability declared", "N": "No disability declared"})
        st.dataframe(groups[["recall", "fpr", "selection_rate", "group_size"]], width="stretch")
        gap = audit["recall_gap"]
        st.metric("Recall gap", f"{gap:.3f}", "passes gate (<= 0.10)" if gap <= S.fairness_gap_max else "fails gate")
        st.caption("Small groups make single-batch gaps noisy; read them next to the group sizes.")
