import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "f1": rng.normal(size=300),
        "f2": rng.normal(size=300),
        "code_module": pd.Categorical(rng.choice(["AAA", "BBB"], 300), categories=["AAA", "BBB"]),
    })
    y = (X["f1"] > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1).fit(X, y)
    model.booster_.save_model(str(tmp_path / "model.txt"))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "numeric": ["f1", "f2"], "categorical": ["code_module"],
        "categories": {"code_module": ["AAA", "BBB"]},
        "threshold": 0.5, "model_version": "test",
    }))
    monkeypatch.setenv("MODEL_URI", str(tmp_path))
    from api.main import app

    with TestClient(app) as c:   # the with-block runs the lifespan, which loads the model
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "test"}


def test_predict(client):
    body = {"students": [
        {"id_student": 1, "features": {"f1": 2.5, "f2": 0.0, "code_module": "AAA"}},
        {"id_student": 2, "features": {"f1": -2.5, "f2": 0.0, "code_module": "ZZZ"}},  # unseen module
    ]}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert [p["id_student"] for p in preds] == [1, 2]
    assert all(0.0 <= p["risk_score"] <= 1.0 for p in preds)
    assert preds[0]["at_risk"] is True and preds[1]["at_risk"] is False


def test_missing_feature_is_rejected(client):
    body = {"students": [{"id_student": 1, "features": {"f1": 1.0}}]}
    assert client.post("/predict", json=body).status_code == 422


def test_empty_request_is_rejected(client):
    assert client.post("/predict", json={"students": []}).status_code == 422
