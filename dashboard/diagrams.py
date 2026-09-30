"""Simple diagrams and small tables shown in the dashboard. Every figure quoted here was measured."""
from __future__ import annotations

import pandas as pd

_STYLE = """
  rankdir=LR; bgcolor="transparent"; fontcolor="#8899aa"; fontsize=13; nodesep=0.4;
  node [shape=box, style="rounded,filled", fillcolor="#dbe7f5", fontcolor="#111", color="#5b7fa6"];
  edge [color="#8899aa", fontcolor="#8899aa"];
"""

WORKFLOW = "digraph {" + _STYLE + """
  a [label="1. A new course batch\\narrives"];
  b [label="2. Turn each student's\\nfirst 28 days into numbers"];
  c [label="3. Score every student\\nwith the current model"];
  d [label="4. Has behaviour\\nchanged a lot?", fillcolor="#f7e3b5"];
  keep [label="No: keep the\\ncurrent model", fillcolor="#e6efe0"];
  e [label="Yes: train a\\nnew model"];
  f [label="5. Ship it only if it is\\nmore accurate AND fair", fillcolor="#f7e3b5"];
  ok [label="Passes: new model\\ngoes live", fillcolor="#cfe8cf"];
  no [label="Fails: keep the\\nold model", fillcolor="#f2cfcf"];
  a -> b -> c -> d;
  d -> keep [label=" no"];
  d -> e [label=" yes"];
  e -> f;
  f -> ok [label=" yes"];
  f -> no [label=" no"];
}
"""

ARCHITECTURE = "digraph {" + _STYLE + """
  user [label="You / an instructor\\n(web browser)", fillcolor="#efe6d0"];
  subgraph cluster_aws {
    label="Amazon Web Services (Canada region)"; color="#8899aa";
    s3 [label="S3: file storage\\ndata, models, results", fillcolor="#e9dff2"];
    subgraph cluster_k3s {
      label="One server running Kubernetes (k3s)"; color="#5b7fa6";
      api [label="API\\n2 copies"];
      dash [label="This dashboard"];
      mlflow [label="MLflow\\nrun history"];
      cron [label="Pipeline job\\ntrains models", fillcolor="#f7e3b5"];
    }
  }
  user -> api [label=" asks for scores"];
  user -> dash;
  api -> s3 [label=" loads model"];
  dash -> s3 [label=" reads results"];
  cron -> mlflow [label=" records runs"];
  cron -> s3 [label=" saves new model"];
}
"""

DOCKER_FLOW = "digraph {" + _STYLE + """
  a [label="Dockerfile\\n(a recipe for one\\nimage)"];
  b [label="GitHub builds it\\nafter tests pass"];
  c [label="Stored in GHCR\\n(image library)\\ntagged with commit ID", fillcolor="#e9dff2"];
  d [label="The server downloads\\nthe image and\\nruns it", fillcolor="#cfe8cf"];
  a -> b -> c -> d;
}
"""

DEPLOYMENT = "digraph {" + _STYLE + """
  subgraph cluster_code {
    label="New CODE"; color="#5b7fa6";
    a [label="Push code\\nto GitHub"];
    b [label="Tests run,\\nimages are built"];
    a -> b;
  }
  subgraph cluster_model {
    label="New MODEL"; color="#6aa06a";
    m1 [label="Pipeline promotes\\na better model"];
    m2 [label="Model file saved\\nin S3"];
    m1 -> m2;
  }
  gh [label="GitHub deploy job\\n(short-lived AWS\\naccess, no stored keys)", fillcolor="#f7e3b5"];
  srv [label="Server updates the\\nAPI pods one at a time\\n(no downtime)", fillcolor="#cfe8cf"];
  b -> gh; m2 -> gh [label=" sends an event"];
  gh -> srv;
}
"""

DOCKER_IMAGES = pd.DataFrame(
    [
        ("oulad-api", "Answers score requests.", "Smallest: no training tools inside (747 MB)."),
        ("oulad-pipeline", "Trains and checks models, one batch per run.", "Has all the training tools."),
        ("oulad-mlflow", "Keeps the history of every training run.", "Separate, so it can be upgraded alone."),
        ("oulad-dashboard", "This page.", "Read-only; only reads results (1.15 GB)."),
    ],
    columns=["Image", "What it does", "Why separate"],
)

DOCKER_PRACTICES = pd.DataFrame(
    [
        ("Install libraries before copying code", "Editing code rebuilds in seconds, not minutes."),
        ("Run as a normal user, not root", "A hacked container has less power."),
        ("Tag images with the commit ID", "You know exactly what is running and can roll back."),
        ("Settings come from environment variables", "The same image runs on a laptop and on AWS."),
    ],
    columns=["Practice", "Why"],
)

KUBERNETES_TABLE = pd.DataFrame(
    [
        ("API Deployment", "Keeps 2 copies of the API running and replaces them one at a time on updates."),
        ("Health checks", "A new copy gets traffic only after its model has loaded, so a broken version never serves users."),
        ("Service", "One fixed address that spreads requests over the copies."),
        ("CronJob", "Runs the training pipeline once per batch (currently paused; I ran the 13 batches by hand)."),
        ("Kustomize", "One shared config with small differences for laptop (kind) and AWS (k3s)."),
    ],
    columns=["Kubernetes piece", "What it does here"],
)

MODEL_CHOICES = pd.DataFrame(
    [
        ("LightGBM (used)", "0.785", "Handles missing values itself and trains in under 1 second."),
        ("Logistic regression", "0.782", "Nearly as good and simpler to explain (measured on the same split)."),
        ("XGBoost, neural network", "not tested", "Not compared in this project."),
    ],
    columns=["Model", "Accuracy (AUC)", "Note"],
)

EXPERIMENTS = pd.DataFrame(
    [
        (14, "no", 0.730, 0.040), (14, "yes", 0.752, 0.031),
        (28, "no", 0.785, 0.048), (28, "yes", 0.798, 0.057),
        (42, "no", 0.812, 0.028), (42, "yes", 0.819, 0.046),
    ],
    columns=["Prediction day", "Uses demographics", "Accuracy (AUC)", "Fairness gap"],
)

FEATURE_NAMES = {
    "mean_score": "Average assessment score",
    "n_missed": "Assessments missed",
    "n_submitted": "Assessments submitted",
    "n_due": "Assessments due so far",
    "n_late": "Late submissions",
    "mean_days_early": "How early work is handed in",
    "score_trend": "Score trend",
    "active_days": "Days active",
    "clicks_total": "Total clicks",
    "clicks_last_7d": "Clicks in the last 7 days",
    "clicks_prev_7d": "Clicks in the 7 days before that",
    "click_trend": "Activity trend",
    "days_since_last_active": "Days since last active",
    "never_active": "Never active",
    "clicks_pre_start": "Clicks before the course started",
    "date_registration": "When they registered",
    "num_of_prev_attempts": "Previous attempts",
    "studied_credits": "Credits being studied",
    "code_module": "Course",
    "clicks_oucontent": "Clicks on course content",
    "clicks_forumng": "Clicks in forums",
    "clicks_quiz": "Clicks on quizzes",
    "clicks_homepage": "Clicks on the homepage",
    "clicks_resource": "Clicks on resources",
    "clicks_subpage": "Clicks on sub-pages",
    "clicks_url": "Clicks on links",
    "clicks_other": "Clicks on other pages",
}


def nice(name: str) -> str:
    return FEATURE_NAMES.get(name, name.replace("_", " ").capitalize())
