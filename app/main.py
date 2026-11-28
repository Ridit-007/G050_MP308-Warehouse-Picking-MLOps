"""FastAPI prediction service for Group G050 (MP308 Warehouse Picking Time)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from warehouse_mlops.preprocessing import clean_dataframe, get_feature_columns, load_params

LOGGER_NAME = "mp308.api"


class JsonFormatter(logging.Formatter):
    """Emit exactly one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    """Send JSON formatted logs to stdout only (never to a file)."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class PredictRequest(BaseModel):
    """Request body: exactly the fields the client can supply at prediction time."""

    model_config = ConfigDict(extra="forbid")

    items_in_order: int = Field(..., ge=1, le=200, description="Items in the order")
    aisle_distance_m: float = Field(..., ge=0, le=5000, description="Walking distance in metres")
    picker_exp_months: int = Field(..., ge=0, le=600, description="Picker experience (client rule: no negatives)")
    shelf_height_m: float = Field(..., gt=0, le=10, description="Average shelf height in metres")
    shift_hour: int = Field(..., ge=0, le=23, description="Hour of picking")
    order_weight_kg: float = Field(..., gt=0, le=45, description="Order weight (client rule: max 45 kg)")
    zone_type: str = Field(..., description="ambient, chilled or bulky")
    shift: str = Field(..., description="Morning, Evening or Night")


class PredictResponse(BaseModel):
    prediction: float
    model: str


class ServiceState:
    """Holds the loaded champion model and the project configuration."""

    def __init__(self, model: Any | None = None, params: dict | None = None) -> None:
        self.params = params or load_params()
        self.model_path = os.getenv("MODEL_PATH", self.params["output"]["model_path"])
        self._model = model
        self.logger = configure_logging()

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = joblib.load(self.model_path)
            self.logger.info(
                "model_loaded",
                extra={"event": {"event": "model_loaded", "path": str(self.model_path)}},
            )
        return self._model


def create_app(model: Any | None = None, params: dict | None = None) -> FastAPI:
    """Build the FastAPI application (a model can be injected for tests)."""
    state = ServiceState(model=model, params=params)
    app = FastAPI(title="MP308 Warehouse Picking Time API", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_ready": Path(state.model_path).exists() or state._model is not None}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()
        payload = request.model_dump()
        frame = pd.DataFrame([payload])
        cleaned = clean_dataframe(frame, state.params, is_training=False)
        features = cleaned[get_feature_columns(state.params)]
        prediction = float(state.model.predict(features)[0])
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        state.logger.info(
            "prediction",
            extra={
                "event": {
                    "event": "prediction",
                    "items_in_order": payload["items_in_order"],
                    "zone_type": payload["zone_type"],
                    "shift": payload["shift"],
                    "prediction": round(prediction, 2),
                    "latency_ms": latency_ms,
                }
            },
        )
        return PredictResponse(prediction=round(prediction, 2), model=Path(state.model_path).name)

    return app


app = create_app()
