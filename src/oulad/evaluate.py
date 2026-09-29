"""Classification metrics and the fairness audit across disability groups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "n": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if two_classes else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def fairness_audit(y_true, proba, threshold: float, sensitive, min_group_size: int = 30) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    sensitive = np.asarray(sensitive).astype(str)
    mf = MetricFrame(
        metrics={
            "recall": true_positive_rate,
            "fnr": false_negative_rate,
            "fpr": false_positive_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    sizes = pd.Series(sensitive).value_counts()
    gaps = mf.difference()
    return {
        "by_group": mf.by_group.round(4).to_dict(orient="index"),
        "group_sizes": {k: int(v) for k, v in sizes.items()},
        "recall_gap": float(gaps["recall"]),
        "fpr_gap": float(gaps["fpr"]),
        "small_groups": [g for g, n in sizes.items() if n < min_group_size],
    }


def flatten_fairness(audit: dict, prefix: str = "fair") -> dict[str, float]:
    """Turn the nested audit into flat MLflow metric names, e.g. fair_recall_Y."""
    flat = {f"{prefix}_recall_gap": audit["recall_gap"], f"{prefix}_fpr_gap": audit["fpr_gap"]}
    for group, metrics in audit["by_group"].items():
        for name, value in metrics.items():
            flat[f"{prefix}_{name}_{group}"] = float(value)
    return flat


def passes_fairness(audit: dict, max_gap: float) -> bool:
    gap = audit["recall_gap"]
    return not np.isnan(gap) and gap <= max_gap
