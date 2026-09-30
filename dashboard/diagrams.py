"""Graphviz diagrams and explanatory tables shown in the dashboard."""
from __future__ import annotations

import pandas as pd

_STYLE = """
  rankdir=LR; bgcolor="transparent"; fontcolor="#8899aa"; fontsize=12;
  node [shape=box, style="rounded,filled", fillcolor="#dbe7f5", fontcolor="#111", color="#5b7fa6"];
  edge [color="#8899aa", fontcolor="#8899aa"];
"""

WORKFLOW = "digraph {" + _STYLE + """
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

ARCHITECTURE = "digraph {" + _STYLE + """
  user [label="Browser / LMS\\n(instructor, dashboard)", fillcolor="#efe6d0"];
  subgraph cluster_aws {
    label="AWS, ca-central-1"; color="#8899aa";
    s3 [label="S3 bucket\\ndata, models/champion,\\nreports, pipeline state", fillcolor="#e9dff2"];
    subgraph cluster_k3s {
      label="One EC2 instance running k3s (Kubernetes), namespace 'oulad'"; color="#5b7fa6";
      svc [label="Services (NodePort)\\n30080 API, 30501 dashboard"];
      api [label="Deployment: oulad-api\\n2 replicas + health probes"];
      dash [label="Deployment: oulad-dashboard"];
      mlflow [label="Deployment: oulad-mlflow\\n+ volume for its database"];
      cron [label="CronJob: oulad-pipeline\\none batch per run", fillcolor="#f7e3b5"];
    }
  }
  user -> svc; svc -> api; svc -> dash;
  api -> s3 [label=" loads champion"];
  dash -> s3 [label=" reads results"];
  cron -> mlflow [label=" logs runs,\\n moves alias"];
  cron -> s3 [label=" writes champion,\\n reports, state"];
}
"""

DEPLOYMENT = "digraph {" + _STYLE + """
  subgraph cluster_code {
    label="Code loop: new software"; color="#5b7fa6";
    push [label="git push\\nto main"];
    ci [label="GitHub Actions ci\\nruff + pytest"];
    build [label="Build 4 images\\ntagged with commit SHA"];
    ghcr [label="GHCR\\n(container registry)", fillcolor="#e9dff2"];
    push -> ci -> build -> ghcr;
  }
  subgraph cluster_model {
    label="Model loop: new model"; color="#6aa06a";
    promote [label="Pipeline promotes\\na new champion", fillcolor="#cfe8cf"];
    s3 [label="Bundle copied to\\nS3 models/champion", fillcolor="#e9dff2"];
    event [label="'model-promoted'\\nGitHub event"];
    promote -> s3 -> event;
  }
  deploy [label="deploy workflow\\n(OIDC: no stored AWS keys)", fillcolor="#f7e3b5"];
  ssm [label="AWS SSM\\nRun Command"];
  k3s [label="k3s on EC2:\\nrolling update, 2 pods,\\nno downtime", fillcolor="#cfe8cf"];
  ghcr -> deploy [label=" ci succeeded"];
  event -> deploy;
  deploy -> ssm -> k3s;
}
"""

KUBERNETES_TABLE = pd.DataFrame(
    [
        ("Deployment", "oulad-api", "Keeps 2 copies of the API running and replaces them one at a time on updates.",
         "Zero-downtime model and code updates; if a new pod cannot load its model it never becomes ready."),
        ("Service (NodePort)", "oulad-api :30080", "One stable address that spreads requests across ready pods.",
         "Pods come and go; callers use a single address."),
        ("Readiness probe", "/health", "Sends traffic to a pod only after the model has loaded.",
         "A broken new version never receives requests."),
        ("Liveness probe", "/health", "Restarts a pod that stops answering.", "Self-healing."),
        ("CronJob", "oulad-pipeline", "Runs the retraining pipeline for one batch per run.",
         "Uses what the cluster already has instead of a separate scheduler."),
        ("Deployment + volume", "oulad-mlflow", "Experiment tracker and model registry, with its own disk.",
         "Keeps run history and the champion label across restarts."),
        ("ConfigMap", "oulad-config", "Environment settings (S3 paths, thresholds).",
         "Same image runs locally and on AWS; only settings change."),
        ("Kustomize overlays", "base, local, aws", "One shared config with small per-environment changes.",
         "Same manifests tested on kind (laptop) and run on k3s (AWS)."),
    ],
    columns=["Kubernetes object", "Name here", "What it does", "Why it is used"],
)

_LGBM = (
    "Fast, accurate on tables, handles missing values natively ('never submitted' becomes "
    "a signal), and retrains in seconds, so retraining on drift is cheap."
)
_LOGREG = (
    "Simple and explainable, a useful baseline, but cannot capture "
    "interactions like 'active early but now silent'."
)

MODEL_CHOICES = pd.DataFrame(
    [
        ("LightGBM (chosen)", _LGBM),
        ("XGBoost", "Similar accuracy, slower to retrain, no benefit here."),
        ("Logistic regression", _LOGREG),
        ("Neural network", "Needs more data and tuning, and is harder to explain to instructors."),
    ],
    columns=["Model", "Why / why not"],
)

EXPERIMENTS = pd.DataFrame(
    [
        (14, "off", 0.730, 0.040), (14, "on", 0.752, 0.031),
        (28, "off", 0.785, 0.048), (28, "on", 0.798, 0.057),
        (42, "off", 0.812, 0.028), (42, "on", 0.819, 0.046),
    ],
    columns=["Cutoff day", "Demographics", "ROC AUC", "Recall gap"],
)

TOOLS = pd.DataFrame(
    [
        ("Parquet + DuckDB", "10.6M click rows summed to per-student features from disk, no cluster needed."),
        ("LightGBM", "The prediction model."),
        ("MLflow", "Records every training run and keeps the champion label."),
        ("Fairlearn", "Measures recall for disabled vs non-disabled learners and gates releases."),
        ("PSI + Evidently", "Detects when student behaviour has shifted; triggers retraining."),
        ("Prefect", "Runs the pipeline steps with retries and logs."),
        ("FastAPI + Docker", "Serves predictions; identical scores on any machine."),
        ("Kubernetes (k3s)", "Keeps services running and updates them without downtime."),
        ("GitHub Actions + OIDC + SSM", "Tests, builds and deploys with no stored AWS keys."),
        ("S3", "Durable storage for data, models and reports."),
    ],
    columns=["Tool", "Role in this project"],
)

DOCKER_FLOW = "digraph {" + _STYLE + """
  subgraph cluster_local {
    label="On my laptop"; color="#5b7fa6";
    df [label="4 Dockerfiles\napi, pipeline,\nmlflow, dashboard"];
    run [label="docker build + run\nscores identical to\nnon-Docker run", fillcolor="#e6efe0"];
    df -> run;
  }
  subgraph cluster_ci {
    label="GitHub Actions (on every push to main)"; color="#6aa06a";
    test [label="ruff + pytest\nmust pass first"];
    build [label="docker buildx\n4 images in parallel\nlayer cache reused"];
    test -> build;
  }
  ghcr [label="GHCR registry\nghcr.io/<owner>/oulad-*\ntags: commit SHA, latest", fillcolor="#e9dff2"];
  k3s [label="k3s on EC2 pulls the\nSHA-tagged image and\nstarts pods", fillcolor="#cfe8cf"];
  df -> test [label=" git push"];
  build -> ghcr -> k3s;
}
"""

DOCKER_IMAGES = pd.DataFrame(
    [
        ("oulad-api", "docker/api.Dockerfile",
         "FastAPI + LightGBM only. No MLflow or training libraries.", "8000",
         "Small and fast to start. Loads the champion from S3 at startup, so a new model needs no rebuild."),
        ("oulad-pipeline", "docker/pipeline.Dockerfile",
         "pandas, DuckDB, LightGBM, MLflow, Prefect, Fairlearn, Evidently.", "none (a job)",
         "Runs `python -m pipelines.flow` once per batch as a Kubernetes CronJob, then exits."),
        ("oulad-mlflow", "docker/mlflow.Dockerfile",
         "MLflow server + boto3, SQLAlchemy pinned below 2.1.", "5000",
         "The experiment tracker and model registry, kept separate so it can be upgraded on its own."),
        ("oulad-dashboard", "docker/dashboard.Dockerfile",
         "Streamlit + LightGBM + Fairlearn.", "8501",
         "This page. Read-only: it only reads results from S3."),
    ],
    columns=["Image", "Built from", "What is inside", "Port", "Why it is a separate image"],
)

DOCKER_PRACTICES = pd.DataFrame(
    [
        ("python:3.11-slim base", "Small image; includes libgomp1, which LightGBM needs."),
        ("Dependencies copied before code", "Docker caches layers, so a code edit rebuilds in seconds, not minutes."),
        ("Separate requirements per image", "The API image stays slim: it never installs training libraries."),
        ("Runs as non-root user", "Least privilege: a compromised container cannot act as root."),
        ("Settings from environment variables", "The same image runs on a laptop and on AWS; only settings change."),
        ("Tags are commit SHAs, not `latest`", "You always know exactly what runs, and can roll back to any commit."),
        (".dockerignore", "Keeps data, the virtual env and secrets out of every image."),
    ],
    columns=["Practice", "Why"],
)
