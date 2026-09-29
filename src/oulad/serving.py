"""Model bundle loading and the single preprocessing function shared by training and serving."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


def prepare_frame(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    categories: dict[str, list[str]],
) -> pd.DataFrame:
    """Select, order and type the model inputs. Unknown categories become NaN."""
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[numeric + categorical].copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    for col in categorical:
        X[col] = pd.Categorical(X[col].astype("string"), categories=categories[col])
    return X


@dataclass
class ModelBundle:
    booster: lgb.Booster
    metadata: dict

    @classmethod
    def load(cls, uri: str) -> ModelBundle:
        """Load model.txt + metadata.json from a local folder or an S3 prefix."""
        from oulad import storage

        local = storage.materialize_dir(uri, tempfile.mkdtemp(prefix="bundle-"))
        booster = lgb.Booster(model_file=str(local / "model.txt"))
        metadata = json.loads((local / "metadata.json").read_text())
        return cls(booster, metadata)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def feature_names(self) -> list[str]:
        return self.metadata["numeric"] + self.metadata["categorical"]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = prepare_frame(
            df, self.metadata["numeric"], self.metadata["categorical"], self.metadata["categories"]
        )
        return self.booster.predict(X)  # binary objective: probability of class 1
