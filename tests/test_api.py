"""API tests for the MP308 service (no CSV or joblib model needed)."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app


class _StubModel:
    """Minimal stand-in for the champion pipeline."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 250.0)


VALID_PAYLOAD = {
    "items_in_order": 18,
    "aisle_distance_m": 210.5,
    "picker_exp_months": 24,
    "shelf_height_m": 1.4,
    "shift_hour": 10,
    "order_weight_kg": 12.5,
    "zone_type": "ambient",
    "shift": "Morning",
}


def _client() -> TestClient:
    return TestClient(create_app(model=_StubModel()))


def test_health_returns_ok():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_number():
    response = _client().post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)


def test_order_weight_above_45_is_rejected():
    payload = {**VALID_PAYLOAD, "order_weight_kg": 46.0}
    assert _client().post("/predict", json=payload).status_code == 422


def test_negative_experience_is_rejected():
    payload = {**VALID_PAYLOAD, "picker_exp_months": -1}
    assert _client().post("/predict", json=payload).status_code == 422


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "zone_type"}
    assert _client().post("/predict", json=payload).status_code == 422
