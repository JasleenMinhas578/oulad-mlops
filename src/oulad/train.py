"""Train LightGBM with early stopping and choose the decision threshold."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit

from oulad.features import LABEL, feature_columns
from oulad.serving import prepare_frame

DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


@dataclass
class TrainedModel:
    model: lgb.LGBMClassifier
    numeric: list[str]
    categorical: list[str]
    categories: dict[str, list[str]]
    threshold: float

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_frame(df, self.numeric, self.categorical, self.categories)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.prepare(df))[:, 1]

    def metadata(self) -> dict:
        return {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "categories": self.categories,
            "threshold": self.threshold,
            "best_iteration": int(self.model.best_iteration_ or self.model.n_estimators),
        }


def choose_threshold(y_true: np.ndarray, proba: np.ndarray, target_recall: float) -> float:
    """Highest threshold that still reaches the target recall (fewest false alarms)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else 0.5


def train_model(
    df: pd.DataFrame,
    include_demographics: bool,
    target_recall: float,
    params: dict | None = None,
    seed: int = 42,
) -> tuple[TrainedModel, pd.DataFrame]:
    """Returns the model and the validation frame with a 'proba' column."""
    numeric, categorical = feature_columns(include_demographics)
    categories = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in categorical}

    # Group by student: the same person can appear in several presentations (retakes).
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(splitter.split(df, groups=df["id_student"]))
    tr, va = df.iloc[tr_idx], df.iloc[va_idx].copy()
    X_tr = prepare_frame(tr, numeric, categorical, categories)
    X_va = prepare_frame(va, numeric, categorical, categories)

    model = lgb.LGBMClassifier(**{**DEFAULT_PARAMS, **(params or {}), "random_state": seed})
    model.fit(
        X_tr, tr[LABEL],
        eval_set=[(X_va, va[LABEL])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    va["proba"] = model.predict_proba(X_va)[:, 1]
    threshold = choose_threshold(va[LABEL].to_numpy(), va["proba"].to_numpy(), target_recall)
    return TrainedModel(model, numeric, categorical, categories, threshold), va


if __name__ == "__main__":
    from pathlib import Path

    from sklearn.metrics import roc_auc_score

    from oulad.config import Settings
    from oulad.features import build_features

    s = Settings.from_env()
    df = build_features(Path(s.data_uri) / "bronze", s.cutoff_day)
    train_df = df[df["code_presentation"].str.startswith("2013")]
    trained, va = train_model(train_df, s.include_demographics, s.target_recall)
    print(f"rows={len(train_df)} at_risk_rate={train_df[LABEL].mean():.3f}")
    print(f"validation AUC={roc_auc_score(va[LABEL], va['proba']):.3f}")
    print(f"threshold for {s.target_recall:.0%} recall = {trained.threshold:.3f}")
