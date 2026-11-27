"""Data cleaning and sklearn preprocessing pipeline for Group G050 (MP308)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_params(path: str | Path = "params.yaml") -> dict:
    """Load project configuration from params.yaml using yaml.safe_load."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_dataframe(
    df: pd.DataFrame, params: dict, *, is_training: bool = True
) -> pd.DataFrame:
    """Apply reproducible cleaning rules derived from Stage 1 audit."""
    cleaned = df.copy()
    cleaning_cfg = params["cleaning"]

    if is_training and cleaning_cfg.get("drop_duplicates", True):
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Remove post-event leaking columns (e.g. pick_time_from_scanner)
    for col in cleaning_cfg.get("leaking_columns", []):
        if col in cleaned.columns:
            cleaned = cleaned.drop(columns=[col])

    # Normalize spelling variants in zone_type and shift
    if "zone_type" in cleaned.columns:
        aliases = cleaning_cfg.get("zone_type_aliases", {})
        cleaned["zone_type"] = (
            cleaned["zone_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(aliases)
        )

    if "shift" in cleaned.columns:
        cleaned["shift"] = cleaned["shift"].astype(str).str.strip().str.title()

    # Replace sentinel placeholders (e.g. shift_hour == -1) with NaN for imputation
    for col, sentinel in cleaning_cfg.get("sentinel_values", {}).items():
        if col in cleaned.columns:
            cleaned.loc[cleaned[col] == sentinel, col] = np.nan

    # Mask stuck sensor readings in the affected date window so imputer fills them
    stuck_cfg = cleaning_cfg.get("stuck_sensor") or {}
    stuck_col = stuck_cfg.get("column")
    if (
        is_training
        and stuck_col
        and stuck_col in cleaned.columns
        and "date" in cleaned.columns
    ):
        start_d = stuck_cfg["start_date"]
        end_d = stuck_cfg["end_date"]
        mask = (cleaned["date"] >= start_d) & (cleaned["date"] <= end_d)
        cleaned.loc[mask, stuck_col] = np.nan

    return cleaned


def get_feature_columns(params: dict) -> list[str]:
    """Return ordered list of model input features."""
    return params["features"]["numeric"] + params["features"]["categorical"]


def build_preprocessor(params: dict) -> ColumnTransformer:
    """Create numeric + categorical preprocessing ColumnTransformer."""
    numeric_cols = params["features"]["numeric"]
    categorical_cols = params["features"]["categorical"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
