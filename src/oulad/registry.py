"""MLflow helpers: log runs, register versions, manage the champion alias, export bundles."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from oulad import storage
from oulad.config import Settings
from oulad.train import TrainedModel

EXPERIMENT = "oulad-at-risk"
CHAMPION = "champion"


def setup(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def log_run(
    trained: TrainedModel,
    params: dict,
    metrics: dict[str, float],
    audit: dict,
    tags: dict[str, str],
    model_name: str,
    register: bool = True,
) -> str | None:
    """Log one training run. Returns the new registry version, or None if not registered."""
    with mlflow.start_run(run_name=tags.get("reason", "train")) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({**params, "threshold": round(trained.threshold, 4),
                           "categorical": ",".join(trained.categorical)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if not math.isnan(v)})
        mlflow.log_dict(audit, "fairness_audit.json")
        with tempfile.TemporaryDirectory() as tmp:
            trained.model.booster_.save_model(str(Path(tmp) / "model.txt"))
            (Path(tmp) / "metadata.json").write_text(json.dumps(trained.metadata(), indent=2))
            mlflow.log_artifacts(tmp, artifact_path="bundle")
        mlflow.lightgbm.log_model(trained.model, artifact_path="model")
    if not register:
        return None
    return mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name).version


def get_champion(client: MlflowClient, model_name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        return None


def promote(client: MlflowClient, model_name: str, version: str) -> None:
    client.set_registered_model_alias(model_name, CHAMPION, str(version))


def export_bundle(client: MlflowClient, model_name: str, version: str, dest_uri: str) -> None:
    """Copy a version's bundle (model.txt + metadata.json) to a local folder or S3 prefix."""
    mv = client.get_model_version(model_name, str(version))
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="bundle", dst_path=tmp))
        meta = json.loads((local / "metadata.json").read_text())
        meta["model_version"] = str(version)
        (local / "metadata.json").write_text(json.dumps(meta, indent=2))
        storage.copy_dir(str(local), dest_uri)
