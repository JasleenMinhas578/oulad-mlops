"""Read-only dashboard for the OULAD early-warning system.

Shows the measured results first, then how it works and how it is deployed. Everything is read
from the same state, reports and model files the pipeline writes (local disk or S3)."""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import diagrams
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # so Streamlit Cloud finds oulad
from oulad import outcomes, storage
from oulad.config import Settings
from oulad.evaluate import fairness_audit
from oulad.serving import ModelBundle, prepare_frame
from oulad.stream import load_state

LABEL = "at_risk"
S = Settings.from_env()
SNAPSHOT = Path(__file__).resolve().parent / "snapshot"
# On Streamlit Cloud there is no pipeline output or S3 access, so read the saved snapshot instead.
USING_SNAPSHOT = (
    not storage.is_s3(S.data_uri)
    and not Path(S.data_uri, "state", "stream_state.json").exists()
    and (SNAPSHOT / "data" / "state" / "stream_state.json").exists()
)
if USING_SNAPSHOT:
    S = dataclasses.replace(S, data_uri=str(SNAPSHOT / "data"), models_uri=str(SNAPSHOT / "models"))
# The pipeline injects a fake engagement drop into one batch to prove the monitor fires.
# That batch is excluded from the real-world results.
INJECTED = f"{int(os.getenv('INJECT_DRIFT_AT', '5')):02d}_"

st.set_page_config(page_title="OULAD early warning", layout="wide")


@st.cache_data(ttl=60)
def get_state() -> dict:
    return load_state(S)


@st.cache_resource(ttl=300)
def get_bundle(name: str) -> ModelBundle:
    return ModelBundle.load(storage.join(S.models_uri, name))


@st.cache_data(ttl=300)
def get_batch(batch_id: str) -> pd.DataFrame:
    return storage.read_parquet(storage.join(S.data_uri, "processed", f"{batch_id}.parquet"))


@st.cache_data(ttl=300)
def get_drift(batch_id: str) -> dict:
    return storage.read_json(storage.join(S.data_uri, "reports", batch_id, "drift.json"))


@st.cache_data(ttl=300)
def get_stream(batch_ids: tuple[str, ...]) -> pd.DataFrame:
    """Real 2014 outcomes, plus what the never-retrained first model (v1) would have done."""
    df = outcomes.load_stream(S.data_uri, list(batch_ids))
    v1 = get_bundle("v1")
    df["v1_score"] = v1.predict_proba(df)
    df["v1_flag"] = (df["v1_score"] >= v1.threshold).astype(int)
    return df


def score(bundle: ModelBundle, df: pd.DataFrame) -> pd.Series:
    return pd.Series(bundle.predict_proba(df), index=df.index)


def explain(bundle: ModelBundle, row: pd.DataFrame) -> pd.Series:
    """Per-feature push on the score (LightGBM's built-in contributions, in log-odds)."""
    meta = bundle.metadata
    X = prepare_frame(row, meta["numeric"], meta["categorical"], meta["categories"])
    contrib = bundle.booster.predict(X, pred_contrib=True)[0]
    return pd.Series(contrib[:-1], index=bundle.feature_names)  # last value is the bias


def show(value) -> str:
    if pd.isna(value):
        return "not available"
    if isinstance(value, str):
        return value
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def outcome(rec: dict) -> str:
    if "promoted" not in rec:
        return "Kept current model"
    if rec["promoted"]:
        return "New model went live"
    if rec.get("beats_champion") and not rec.get("passes_fairness"):
        return "New model blocked: not fair enough"
    return "New model not better, kept old"


try:
    state = get_state()
    champion = get_bundle("champion")
except Exception as exc:  # noqa: BLE001  show a readable message instead of a stack trace
    st.error(f"Pipeline output not found yet: {exc}")
    st.stop()

history = pd.DataFrame(state["history"])
history["outcome"] = [outcome(r) for r in state["history"]]
real_batches = tuple(b for b in history["batch"] if not b.startswith(INJECTED))
stream = get_stream(real_batches)
overall = outcomes.summarize(stream)
groups = outcomes.summarize_by_group(stream)
static = outcomes.summarize(stream, "v1_flag")
auc_now = outcomes.mean_batch_auc(stream, "risk_score")
auc_v1 = outcomes.mean_batch_auc(stream, "v1_score")

# ---------------------------------------------------------------- header
st.title("Early warning for learners at risk of failing or withdrawing")
st.markdown(
    "At **day 28** of a course, the model looks at how each learner has behaved so far (activity, "
    "assignments handed in, scores) and flags those likely to **fail or withdraw**, early enough "
    "for an instructor to reach out. Data: Open University, 32,593 enrolments and 10.6 million "
    "click records. The model never sees whether a learner declared a disability; that is only "
    "used to check the model treats everyone fairly."
)

if USING_SNAPSHOT:
    st.info("Showing a saved copy of the pipeline's results from the full 2014 replay.")

st.subheader("The result")
st.caption(
    f"Measured on {len(real_batches)} course batches from 2014 that the model had never seen "
    f"({overall['students']:,} learners). Each learner was scored before their outcome was known."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "At-risk learners caught", f"{overall['recall']:.0%}",
    help="Of the learners who really went on to fail or withdraw, the share the model flagged.",
)
c2.metric(
    "Learners flagged for outreach", f"{overall['share_flagged']:.0%}",
    f"{overall['contacts_saved']:.0%} fewer than a random list",
    delta_color="off",
    help=f"A random list would have to include {overall['recall']:.0%} of all learners "
         "to reach the same share of at-risk learners.",
)
dis_y, dis_n = groups.get("Y"), groups.get("N")
c3.metric(
    "Disabled learners caught", f"{dis_y['recall']:.0%}" if dis_y else "n/a",
    f"others: {dis_n['recall']:.0%}" if dis_n else None, delta_color="off",
    help="Recall for learners who declared a disability, compared with everyone else.",
)
c4.metric(
    "Accuracy (AUC)", f"{auc_now:.2f}",
    help="How well the model ranks at-risk learners above safe ones. 0.5 is a coin flip, 1.0 is perfect.",
)
st.markdown(
    f"**In plain terms:** about **{overall['precision']:.0%}** of the learners flagged really did fail or "
    f"withdraw, compared with **{overall['base_rate']:.0%}** of all learners. "
    "So the list is a useful shortlist, not a perfect one."
)

with st.container(border=True):
    st.markdown("**What did not work as hoped (honest findings)**")
    retrains = int(sum(bool(r) for r in history["trigger_reasons"]))
    st.markdown(
        f"- **Automatic retraining did not measurably improve accuracy.** The first model, never "
        f"retrained, scored {auc_v1:.3f} AUC on the same 2014 batches; the retrained system scored "
        f"{auc_now:.3f}. The value of the pipeline is control: no model goes live unless it passes an "
        f"accuracy check and a fairness check.\n"
        f"- **The first drift alarm was too sensitive, and is now fixed.** It compared each single course with a mix of "
        f"courses and fired on all 13 batches. Comparing with the same course fixed that: it now triggers a "
        f"retrain on {retrains} of {len(history)} batches and catches the simulated outage. It was tuned on this same "
        f"replay, so treat it as a fitted design, not an independent test.\n"
        "- **A simple baseline is almost as good.** Logistic regression scored 0.782 AUC against "
        "0.785 for LightGBM on the same split, so the choice of model matters little here.\n"
        "- **Accuracy is lower on new terms.** Validation AUC on 2013 was 0.785; on unseen 2014 batches "
        f"it was {auc_now:.2f}."
    )

tab_res, tab_try, tab_how, tab_infra, tab_model = st.tabs(
    ["Results in detail", "Try it", "How it works", "Behind the scenes", "Model and monitoring"]
)

# ---------------------------------------------------------------- results
with tab_res:
    st.subheader("Retrained system vs the first model")
    compare = pd.DataFrame(
        {
            "First model, never retrained": [
                f"{auc_v1:.3f}", f"{static['share_flagged']:.0%}", f"{static['recall']:.0%}",
                f"{static['precision']:.0%}",
            ],
            "This system (auto-retrained)": [
                f"{auc_now:.3f}", f"{overall['share_flagged']:.0%}", f"{overall['recall']:.0%}",
                f"{overall['precision']:.0%}",
            ],
        },
        index=["Accuracy (AUC)", "Learners flagged", "At-risk learners caught", "Flagged who really are at risk"],
    )
    st.dataframe(compare, width="stretch")
    st.caption("Both are scored on exactly the same learners, none of whom either model had trained on.")

    st.subheader("Accuracy on each batch")
    per_batch = pd.DataFrame(
        {
            "This system": {b: outcomes.mean_batch_auc(d, "risk_score") for b, d in stream.groupby("batch")},
            "First model": {b: outcomes.mean_batch_auc(d, "v1_score") for b, d in stream.groupby("batch")},
        }
    )
    st.line_chart(per_batch)
    st.caption("Each point is one course batch. Some courses are simply harder to predict than others.")

    st.subheader("Is it fair to learners who declared a disability?")
    fair = pd.DataFrame(
        {
            "Learners": {k: v["students"] for k, v in groups.items()},
            "Really at risk": {k: v["at_risk"] for k, v in groups.items()},
            "At-risk caught": {k: f"{v['recall']:.0%}" for k, v in groups.items()},
            "Flagged for outreach": {k: f"{v['share_flagged']:.0%}" for k, v in groups.items()},
        }
    ).rename(index={"Y": "Declared a disability", "N": "No disability declared"})
    st.dataframe(fair, width="stretch")
    st.caption(
        "Disabled learners are caught at least as often as others, so the model does not silently "
        "miss them. They are also flagged more often, because their at-risk rate is higher in this data."
    )

# ---------------------------------------------------------------- try it
with tab_try:
    st.subheader("Pick a course batch and see who is flagged, and why")
    batch_id = st.selectbox("Course batch (course code and term)", history["batch"].tolist())
    if batch_id.startswith(INJECTED):
        st.warning(
            "The pipeline scored this batch after a simulated outage was injected to test the drift alarm. "
            "The learners shown here are the clean originals."
        )
    df = get_batch(batch_id).reset_index(drop=True)
    df["risk_score"] = score(champion, df)
    df["flagged"] = df["risk_score"] >= champion.threshold
    a, b = st.columns(2)
    a.metric("Learners in this batch", f"{len(df):,}")
    b.metric("Flagged for outreach", f"{int(df['flagged'].sum()):,}",
             help=f"Flagged when the risk score is at least {champion.threshold:.2f}.")

    top = df.sort_values("risk_score", ascending=False).head(50).reset_index(drop=True)
    table = pd.DataFrame(
        {
            "Learner": top["id_student"],
            "Risk score (0 to 1)": top["risk_score"].round(2),
            "Assessments missed": top["n_missed"],
            "Average score": top["mean_score"].round(0),
            "Days active": top["active_days"],
            "What actually happened": top["final_result"],
        }
    )
    st.write("**Highest-risk learners in this batch**")
    st.dataframe(table.head(15), width="stretch", hide_index=True)

    student = st.selectbox(
        "Explain one learner",
        top["id_student"].tolist(),
        format_func=lambda i: f"Learner {i} (risk {top.loc[top['id_student'] == i, 'risk_score'].iloc[0]:.2f})",
    )
    row = df[df["id_student"] == student].head(1)
    contrib = explain(champion, row)
    up = contrib[contrib > 0].sort_values(ascending=False).head(3)
    down = contrib[contrib < 0].sort_values().head(3)
    left, right = st.columns(2)
    with left:
        st.markdown("**Raises the risk**")
        for f in up.index:
            st.markdown(f"- {diagrams.nice(f)}: {show(row.iloc[0][f])}")
    with right:
        st.markdown("**Lowers the risk**")
        for f in down.index:
            st.markdown(f"- {diagrams.nice(f)}: {show(row.iloc[0][f])}")
    st.caption("These are the three biggest reasons in each direction for this one learner.")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.subheader("What happens for every batch of learners")
    st.graphviz_chart(diagrams.WORKFLOW, width="stretch")
    st.markdown(
        "- **Training data:** the first model learns from 2013. The 13 course batches of 2014 are "
        "then replayed one at a time as if they were arriving live.\n"
        "- **Who is scored:** learners still enrolled at day 28 (someone who already left cannot be helped).\n"
        "- **What counts as at risk:** the learner ends up with Fail or Withdrawn.\n"
        "- **Only the past is used:** every input comes from before day 28, so the model never peeks at the future.\n"
        "- **When it retrains** (compared with the same course's training data): many inputs shifted (30% or more), "
        "activity volume collapsed (an outage), the course is new to the model, or accuracy dropped 0.05 below the baseline.\n"
        "- **Threshold:** set to catch about 80% of at-risk learners, because missing a struggling learner "
        "costs more than sending one extra check-in message."
    )

# ---------------------------------------------------------------- infrastructure
with tab_infra:
    st.subheader("Where it runs")
    st.markdown(
        "Everything runs on **one AWS server using Kubernetes (k3s)**, with files kept in S3. "
        "k3s is a lightweight Kubernetes, so it gives the same features as the managed EKS service "
        "without the hourly fee."
    )
    st.graphviz_chart(diagrams.ARCHITECTURE, width="stretch")
    st.dataframe(diagrams.KUBERNETES_TABLE, width="stretch", hide_index=True)

    st.divider()
    st.subheader("How Docker is used")
    st.markdown(
        "Docker packages code and its libraries into an **image** that runs the same anywhere. "
        "This project builds **four images**, one per job."
    )
    st.graphviz_chart(diagrams.DOCKER_FLOW, width="stretch")
    st.dataframe(diagrams.DOCKER_IMAGES, width="stretch", hide_index=True)
    with st.expander("Docker practices used"):
        st.dataframe(diagrams.DOCKER_PRACTICES, width="stretch", hide_index=True)
    st.caption("Checked: the API container returned exactly the same scores as the non-Docker run.")

    st.divider()
    st.subheader("How a change reaches production")
    st.graphviz_chart(diagrams.DEPLOYMENT, width="stretch")
    st.markdown(
        f"- **New code:** tests run, images are built, and the server swaps pods one at a time, so users see no downtime.\n"
        f"- **New model:** the pipeline saves it to S3 and sends an event; the API pods restart and load it. "
        f"Right now the live model is **version {champion.version}**.\n"
        "- **No stored passwords:** GitHub gets short-lived AWS access for each deploy (OIDC), so there is no "
        "AWS key in the repository to leak.\n"
        "- **Undo:** copy an older model over the current one and restart, or roll back the code deployment."
    )

# ---------------------------------------------------------------- model and monitoring
with tab_model:
    st.subheader("Which model, and why")
    st.markdown(
        f"The live model (version {champion.version}) is a **LightGBM** gradient-boosted decision tree "
        "model. It is not clearly better than a simple model on this data, as the table shows."
    )
    st.dataframe(diagrams.MODEL_CHOICES, width="stretch", hide_index=True)
    st.caption("Accuracy measured on 2013 data the models had not trained on. Training LightGBM takes about 0.7 seconds.")

    imp = pd.Series(
        champion.booster.feature_importance(importance_type="gain"),
        index=champion.booster.feature_name(),
    )
    share = (imp / imp.sum()).sort_values(ascending=False)
    top6 = share.head(6)
    st.write("**What the model relies on most**")
    st.bar_chart(top6.rename(index=diagrams.nice).rename("share of the model's decisions"))
    course_share = float(share.get("code_module", 0))
    st.caption(
        f"It leans mostly on **{diagrams.nice(share.index[0]).lower()}** and "
        f"**{diagrams.nice(share.index[1]).lower()}**, then on activity. "
        f"The course itself accounts for {course_share:.0%}, so part of the signal is simply which courses are hard."
    )

    st.write("**Why day 28, and why leave demographics out**")
    st.dataframe(diagrams.EXPERIMENTS, width="stretch", hide_index=True)
    st.caption(
        "Waiting longer is more accurate but leaves less time to help. Adding demographics improves accuracy "
        "only slightly and does not consistently narrow the fairness gap (lower is fairer)."
    )

    st.divider()
    st.subheader("What happened to the model across the 13 batches")
    counts = history["outcome"].value_counts()
    st.dataframe(counts.rename("Batches"), width="stretch")
    st.caption(
        "'Blocked: not fair enough' means a new model was more accurate but would have widened the gap "
        "in how well disabled and other learners are caught, so it was rejected."
    )
    history["why"] = [", ".join(r) if r else "no trigger" for r in history["trigger_reasons"]]
    view = history[["batch", "rows", "drift_share", "why", "champion_version", "outcome"]].rename(
        columns={"batch": "Batch", "rows": "Learners", "drift_share": "Share of inputs that shifted",
                 "why": "Why retrain was triggered", "champion_version": "Model used", "outcome": "Decision"}
    )
    st.dataframe(view, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Has behaviour changed, and is a batch treated fairly?")
    d_batch = st.selectbox("Batch", history["batch"].tolist(), key="drift_batch")
    left, right = st.columns(2)
    with left:
        st.write("**Which inputs shifted most** (above 0.2 counts as shifted)")
        psi = pd.Series(get_drift(d_batch)["psi"]).sort_values(ascending=False).head(10)
        st.bar_chart(psi.rename(index=diagrams.nice).rename("shift score"))
    with right:
        st.write("**At-risk learners caught, by group**")
        bdf = get_batch(d_batch).reset_index(drop=True)
        audit = fairness_audit(bdf[LABEL], score(champion, bdf), champion.threshold, bdf["disability"])
        by_group = pd.DataFrame(audit["by_group"]).T
        by_group["Learners"] = pd.Series(audit["group_sizes"])
        by_group = by_group.rename(index={"Y": "Declared a disability", "N": "No disability declared"})
        by_group["recall"] = (by_group["recall"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(by_group[["recall", "Learners"]].rename(columns={"recall": "At-risk caught"}),
                     width="stretch")
        gap = audit["recall_gap"]
        st.metric("Gap between the two groups", f"{gap:.2f}",
                  "within the 0.10 limit" if gap <= S.fairness_gap_max else "over the 0.10 limit",
                  delta_color="off")
        st.caption("Small groups make one batch noisy; look at the group sizes.")
