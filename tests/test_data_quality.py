"""Unit tests for Stage 1 data quality checks (synthetic data, no CSV reads)."""

import pandas as pd

from warehouse_mlops.data_quality import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
)


def _sample_clean_later() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["L1", "L2", "L3"],
            "date": ["2026-07-01", "2026-07-02", "2026-07-03"],
            "items_in_order": [10, 20, 30],
            "aisle_distance_m": [100, 200, 300],
            "picker_exp_months": [12, 24, 36],
            "shelf_height_m": [1.0, 1.5, 2.0],
            "shift_hour": [8, 14, 22],
            "order_weight_kg": [5.0, 10.0, 15.0],
            "zone_type": ["ambient", "bulky", "chilled"],
            "shift": ["Morning", "Evening", "Night"],
            "pick_time_from_scanner": [120, 220, 320],
            "pick_seconds": [121, 219, 321],
        }
    )


def test_detects_leaking_and_duplicates_and_sentinels():
    later_df = _sample_clean_later()
    holdout_df = later_df.drop(columns=["pick_time_from_scanner", "pick_seconds"])

    train_df = later_df.copy()
    train_df["shift_hour"] = [-1, -1, -1]
    train_df["zone_type"] = ["bul.", "Bulky", "amb."]
    # Add one exact extra copy of the first record (duplicate_rows = 1)
    train_df = pd.concat([train_df, train_df.iloc[[0]]], ignore_index=True)

    leak = check_leaking_column(train_df, holdout_df)
    assert leak["present"] is True
    assert "pick_time_from_scanner" in leak["extra_columns"]

    dups = check_duplicate_rows(train_df)
    assert dups["present"] is True
    assert dups["extra_copies"] == 1

    sent = check_sentinel_value(train_df, later_df)
    assert sent["present"] is True
    assert sent["hits"][0]["column"] == "shift_hour"
    assert sent["hits"][0]["value"] == -1

    spell = check_spelling_variants(train_df, later_df)
    assert spell["present"] is True
    assert spell["issues"][0]["true_levels"] == 3


def test_detects_stuck_sensor_and_clean_checks():
    later_df = _sample_clean_later()
    rows = []
    for day in range(1, 21):
        rows.append(
            {
                "record_id": f"T{day}",
                "date": f"2026-05-{day:02d}",
                "items_in_order": 10 + day % 5,
                "aisle_distance_m": 100 + day * 2,
                "picker_exp_months": 12 + day % 7,
                "shelf_height_m": 1.35,
                "shift_hour": day % 24,
                "order_weight_kg": 8.0 + (day % 5) * 0.5,
                "zone_type": "ambient",
                "shift": ["Morning", "Evening", "Night"][day % 3],
                "pick_seconds": 150 + day,
            }
        )
    train_df = pd.DataFrame(rows)

    stuck = check_stuck_sensor(train_df)
    assert stuck["present"] is True
    assert stuck["candidates"][0]["column"] == "shelf_height_m"
    assert stuck["candidates"][0]["start_date"] == "2026-05-01"
    assert stuck["candidates"][0]["end_date"] == "2026-05-20"

    assert check_unit_mixup(train_df, later_df)["present"] is False
    assert check_target_typos(train_df, later_df)["present"] is False
