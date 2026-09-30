.PHONY: dashboard data mlflow bootstrap api test lint run-batch images kind-up kind-deploy

dashboard:
	streamlit run dashboard/app.py

data:
	python -m oulad.ingest --raw data/raw --out data/bronze

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001

bootstrap:
	python -m oulad.bootstrap

api:
	uvicorn api.main:app --reload --port 8000

test:
	pytest -q

lint:
	ruff check .

run-batch:
	python -m pipelines.flow

images:
	docker build -f docker/api.Dockerfile -t oulad-api:dev .
	docker build -f docker/pipeline.Dockerfile -t oulad-pipeline:dev .

kind-up:
	kind create cluster --name oulad --config k8s/kind-config.yaml

kind-deploy: images
	kind load docker-image oulad-api:dev --name oulad
	kubectl apply -k k8s/overlays/local
