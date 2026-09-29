# OULAD Early Warning: an end-to-end MLOps pipeline

Flags learners at risk of failing or withdrawing four weeks into a course, retrains itself when
behaviour drifts, and audits every model for fairness toward learners who declared a disability.
Free, open-source tools throughout; AWS is used only for hosting.

## Why it matters

Learners who fail an online course usually go quiet weeks before a bad grade appears. Disabled
learners are more exposed to this, and a model that silently works worse for them hides the
problem. The disability flag is therefore **never a model input**. It is used only to audit recall.

## What it does

| Item | Definition |
| --- | --- |
| Unit | One student in one course presentation (module + term) |
| Prediction time | Day 28 (`CUTOFF_DAY`); features use events with `date < 28` only |
| Population | Students still enrolled at day 28 |
| Label | `at_risk` = final result is Fail or Withdrawn |
| Threshold | Tuned for 80% recall (missing a struggling learner costs more than an extra email) |
| Gate | A new model ships only if AUC >= champion **and** recall gap between disability groups <= 0.10 |

## Architecture

Model loop: Prefect pipeline -> MLflow registry (`champion` / `challenger` aliases) -> exported bundle
in `models/champion/` -> `model-promoted` event -> deploy workflow restarts the API.
Code loop: push to `main` -> CI (ruff, pytest) -> images tagged with the commit SHA on GHCR -> deploy.

| Layer | Tool |
| --- | --- |
| Data | Parquet + DuckDB |
| Model | LightGBM, grouped split, early stopping |
| Tracking / registry | MLflow 2.x (SQLite + artifacts) |
| Fairness | Fairlearn (`MetricFrame`) |
| Drift | Own PSI + Evidently HTML report |
| Orchestration | Prefect 3 |
| Serving | FastAPI + Uvicorn in Docker |
| Cluster | kind locally, k3s on one EC2 instance; Kustomize overlays |
| CI/CD | GitHub Actions, GHCR, OIDC + SSM |

## Results (produced by this repo, local run)

Training: 2013B + 2013J, 11,809 enrolled-at-day-28 rows, at-risk rate 0.444.

| Metric | Validation (champion v1) |
| --- | --- |
| ROC AUC | 0.785 |
| PR AUC | 0.769 |
| Recall at threshold (0.332) | 0.800 |
| Precision | 0.598 |
| Recall, disability = Y (n = 246) | 0.843 |
| Recall, disability = N (n = 2117) | 0.795 |
| Recall gap | 0.048 |

Replaying the 13 presentations of 2014 as a stream (drift injected at batch 5):

| Outcome | Batches |
| --- | --- |
| Retrain triggered | 13 of 13 (drift share was 0.38 to 0.89 on every batch) |
| Challenger promoted | 7 |
| Challenger did not beat the champion | 4 |
| **Blocked by the fairness gate despite higher AUC** | 2 (`05_GGG_2014B`, `10_EEE_2014J`) |
| Champion mean AUC / recall / recall gap on the stream | 0.752 / 0.771 / 0.048 |

Caveats: the drift monitor fires on every batch because each batch is a single module while the
reference mixes several (`code_module` itself is excluded from the drift share, other features still
differ by module). A per-module reference (Phase 5.5, fix 2) would be the next improvement. Small
groups make per-batch recall gaps noisy; always read them next to group sizes.
Docker check: the container returns scores identical to the local run; image size 747 MB.

## Run it locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # 3.11 or newer
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
# put the seven OULAD CSVs in data/raw/, then:
make data                                                 # CSV -> Parquet bronze
make mlflow                                               # terminal 2, UI at http://127.0.0.1:5001
make bootstrap                                            # train, audit, register champion v1
INJECT_DRIFT_AT=5 bash -c 'for i in $(seq 13); do python -m pipelines.flow; done'
jq '.history[] | {batch, drift_share, champion_auc, promoted}' data/state/stream_state.json
make api                                                  # http://localhost:8000/docs
make kind-up && make kind-deploy                          # needs kind
```

Notes: LightGBM on macOS needs `brew install libomp`. MLflow runs on port **5001** because macOS
AirPlay Receiver occupies 5000. `sqlalchemy<2.1` is pinned because MLflow 2.x does not import with 2.1.

## Deploy to AWS

Follow Phase 9 of the build guide: S3 bucket, instance role, security group, EC2 `t3.medium` with k3s,
GitHub OIDC role, `kubectl apply -k k8s/overlays/aws` (replace `__OWNER__` and `__BUCKET__` first).
Set the secrets `AWS_DEPLOY_ROLE_ARN`, `AWS_REGION`, `EC2_INSTANCE_ID`. Tear down before credits end.

## Design decisions

- **Cutoff and population**: students who already withdrew are excluded; predicting them is pointless and inflates accuracy.
- **Fairness as a gate**: enforced in code (`passes_fairness`), not a slide.
- **One preprocessing function** (`prepare_frame`) for training and serving, to prevent skew.
- **Bundle, not pyfunc**: the API loads `model.txt` + `metadata.json` and never imports MLflow.
- **k3s instead of EKS**: same Kubernetes API without the control-plane fee.

## Limitations and ethics

Adult distance learners, not children. The disability flag is a coarse self-declared Y/N. Data is from
2013-2014, so exact coefficients would not transfer. The stream is simulated: real labels arrive weeks
later, here they are read immediately. Clicks measure activity, not understanding. This model supports
outreach decisions by people; it must never be used to penalise a learner.

## Dataset

Kuzilek, Hlosta and Zdrahal, "Open University Learning Analytics dataset", *Scientific Data*, 2017.
Licence: CC BY 4.0.
