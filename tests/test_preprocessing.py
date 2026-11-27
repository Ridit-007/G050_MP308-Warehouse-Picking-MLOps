"""Unit tests for Stage 2 cleaning and sklearn preprocessing (no CSV reads)."""

import numpy as np
import pandas as pd

from warehouse_mlops.preprocessing import (
    build_preprocessor,
    clean_dataframe,
    get_feature_columns,
    load_params,
)


def test_clean_dataframe_fixes_all_issues():
    params = load_params("params.yaml")
    raw = pd.DataFrame(
        [
            {
                "record_id": "R1",
                "date": "2026-05-10",
                "items_in_order": 5,
                "aisle_distance_m": 80.0,
                "picker_exp_months": 12,
                "shelf_height_m": 1.35,
                "shift_hour": -1,
                "order_weight_kg": 4.2,
                "zone_type": "bul.",
                "shift": "morning ",
                "pick_time_from_scanner": 110.0,
                "pick_seconds": 111.0,
            },
            {
                "record_id": "R1",
                "date": "2026-05-10",
                "items_in_order": 5,
                "aisle_distance_m": 80.0,
                "picker_exp_months": 12,
                "shelf_height_m": 1.35,
                "shift_hour": -1,
                "order_weight_kg": 4.2,
                "zone_type": "bul.",
                "shift": "morning ",
                "pick_time_from_scanner": 110.0,
                "pick_seconds": 111.0,
            },
            {
                "record_id": "R2",
                "date": "2026-06-01",
                "items_in_order": 15,
                "aisle_distance_m": 160.0,
                "picker_exp_months": 24,
                "shelf_height_m": 2.1,
                "shift_hour": 14,
                "order_weight_kg": 10.0,
                "zone_type": " Amb. ",
                "shift": "Night",
                "pick_time_from_scanner": 240.0,
                "pick_seconds": 241.0,
            },
        ]
    )

    cleaned = clean_dataframe(raw, params, is_training=True)
    assert len(cleaned) == 2
    assert "pick_time_from_scanner" not in cleaned.columns
    assert cleaned["zone_type"].tolist() == ["bulky", "ambient"]
    assert cleaned["shift"].tolist() == ["Morning", "Night"]
    assert np.isnan(cleaned.loc[0, "shift_hour"])
    assert np.isnan(cleaned.loc[0, "shelf_height_m"])
    assert cleaned.loc[1, "shelf_height_m"] == 2.1


def test_preprocessor_handles_missing_and_transforms():
    params = load_params("params.yaml")
    df = pd.DataFrame(
        [
            {
                "items_in_order": 10,
                "aisle_distance_m": 100.0,
                "picker_exp_months": 12,
                "shelf_height_m": np.nan,
                "shift_hour": np.nan,
                "order_weight_kg": 5.0,
                "zone_type": "ambient",
                "shift": "Morning",
            },
            {
                "items_in_order": 20,
                "aisle_distance_m": 200.0,
                "picker_exp_months": 24,
                "shelf_height_m": 1.6,
                "shift_hour": 15.0,
                "order_weight_kg": 12.0,
                "zone_type": "bulky",
                "shift": "Evening",
            },
        ]
    )
    pre = build_preprocessor(params)
    X_t = pre.fit_transform(df[get_feature_columns(params)])
    assert X_t.shape[0] == 2
    assert not np.isnan(X_t).any()
