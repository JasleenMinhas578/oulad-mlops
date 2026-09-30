"""Real out-of-time outcomes of the replayed 2014 stream, computed from what the pipeline saved.

Every student in a batch is scored by the champion of that moment, before that batch's outcomes
are used for any training, so these numbers are honest estimates of live performance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oulad import storage
from oulad.data import KEYS


def load_stream(data_uri: str, batch_ids: list[str]) -> pd.DataFrame:
    """One row per scored student: true label, disability, score and flag from the pipeline."""
    parts = []
    for b in batch_ids:
        proc = storage.read_parquet(storage.join(data_uri, "processed", f"{b}.parquet"))
        pred = storage.read_parquet(storage.join(data_uri, "predictions", f"{b}.parquet"))
        merged = proc.merge(pred[[*KEYS, "risk_score", "flagged"]], on=KEYS)
        parts.append(merged.assign(batch=b))
    return pd.concat(parts, ignore_index=True)


def summarize(df: pd.DataFrame, flag: str = "flagged") -> dict:
    """Headline numbers for a set of scored students."""
    n = len(df)
    at_risk = int(df["at_risk"].sum())
    flagged = int(df[flag].sum())
    caught = int(((df[flag] == 1) & (df["at_risk"] == 1)).sum())
    recall = caught / at_risk if at_risk else float("nan")
    precision = caught / flagged if flagged else float("nan")
    share = flagged / n if n else float("nan")
    # A random contact list must include a share of students equal to the recall to reach the
    # same share of at-risk learners, so the saving is 1 - share / recall.
    saved = 1 - share / recall if recall and not np.isnan(recall) else float("nan")
    return {
        "students": n, "at_risk": at_risk, "base_rate": at_risk / n if n else float("nan"),
        "flagged": flagged, "share_flagged": share, "caught": caught,
        "recall": recall, "precision": precision, "contacts_saved": saved,
    }


def summarize_by_group(df: pd.DataFrame, column: str = "disability", flag: str = "flagged") -> dict:
    return {str(g): summarize(d, flag) for g, d in df.groupby(column)}


def mean_batch_auc(df: pd.DataFrame, score: str = "risk_score") -> float:
    """Average ROC AUC over batches that contain both classes."""
    aucs = [
        roc_auc_score(d["at_risk"], d[score])
        for _, d in df.groupby("batch")
        if d["at_risk"].nunique() == 2
    ]
    return float(np.mean(aucs)) if aucs else float("nan")
