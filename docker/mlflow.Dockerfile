FROM python:3.11-slim
RUN pip install --no-cache-dir "mlflow>=2.16,<3" "sqlalchemy<2.1" boto3 && mkdir -p /mlflow
ENV MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts
EXPOSE 5000
# Shell form so the env var is expanded at start-up.
CMD mlflow server --host 0.0.0.0 --port 5000 --workers 2 \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION"
