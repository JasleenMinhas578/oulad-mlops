"""FastAPI service that scores students with the current champion model."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from oulad.serving import ModelBundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("oulad.api")


class Student(BaseModel):
    id_student: int
    features: dict[str, float | str | None] = Field(
        ..., description="Feature name to value, exactly as produced by oulad.features"
    )


class PredictRequest(BaseModel):
    students: list[Student] = Field(..., min_length=1, max_length=5000)


class Prediction(BaseModel):
    id_student: int
    risk_score: float
    at_risk: bool


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup. If this fails the container exits, and Kubernetes
    # keeps the old pods serving: failing fast is the safe behaviour.
    uri = os.getenv("MODEL_URI", "artifacts/models/champion")
    app.state.bundle = ModelBundle.load(uri)
    log.info("loaded model version %s from %s", app.state.bundle.version, uri)
    yield


app = FastAPI(title="OULAD early warning API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": app.state.bundle.version}


@app.get("/model")
def model_info() -> dict:
    b = app.state.bundle
    return {"model_version": b.version, "threshold": b.threshold, "features": b.feature_names}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    b = app.state.bundle
    df = pd.DataFrame([s.features for s in req.students])
    missing = [f for f in b.feature_names if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing}")
    start = time.perf_counter()
    proba = b.predict_proba(df)
    log.info("scored %d students in %.1f ms (model v%s)",
             len(df), (time.perf_counter() - start) * 1000, b.version)
    return PredictResponse(
        model_version=b.version,
        threshold=b.threshold,
        predictions=[
            Prediction(id_student=s.id_student, risk_score=round(float(p), 4),
                       at_risk=bool(p >= b.threshold))
            for s, p in zip(req.students, proba)
        ],
    )
