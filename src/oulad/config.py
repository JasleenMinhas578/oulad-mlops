"""Central configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    data_uri: str               # root for bronze/, reference/, processed/, state/, reports/
    models_uri: str             # where exported model bundles live
    cutoff_day: int             # prediction day relative to course start
    mlflow_tracking_uri: str
    model_name: str
    psi_threshold: float        # PSI above this = the feature drifted
    drift_share_threshold: float  # share of drifted features that triggers a retrain
    auc_drop_threshold: float   # AUC drop vs baseline that triggers a retrain
    fairness_gap_max: float     # max allowed recall gap between disability groups
    target_recall: float        # recall the decision threshold is tuned for
    include_demographics: bool  # experiment toggle, off by default
    github_repo: str            # owner/repo, for the redeploy trigger
    github_token: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_uri=_env("DATA_URI", "data"),
            models_uri=_env("MODELS_URI", "artifacts/models"),
            cutoff_day=int(_env("CUTOFF_DAY", "28")),
            mlflow_tracking_uri=_env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
            model_name=_env("MODEL_NAME", "oulad-at-risk"),
            psi_threshold=float(_env("PSI_THRESHOLD", "0.2")),
            drift_share_threshold=float(_env("DRIFT_SHARE_THRESHOLD", "0.3")),
            auc_drop_threshold=float(_env("AUC_DROP_THRESHOLD", "0.05")),
            fairness_gap_max=float(_env("FAIRNESS_GAP_MAX", "0.10")),
            target_recall=float(_env("TARGET_RECALL", "0.80")),
            include_demographics=_env("INCLUDE_DEMOGRAPHICS", "false").lower() == "true",
            github_repo=_env("GITHUB_REPO", ""),
            github_token=_env("GITHUB_DISPATCH_TOKEN", ""),
        )
