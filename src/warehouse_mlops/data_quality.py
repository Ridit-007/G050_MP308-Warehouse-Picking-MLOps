"""Stage 1 data quality inspection for Group G050 (MP308)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

TARGET_COL = "pick_seconds"
GROUP_COL = "shift"
ID_COLS = {"record_id", "date"}
KNOWN_SENTINELS = {-9999, -999, -99, -1, 999, 9999, 99999}


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_leaking_column(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame
) -> dict:
    """Identify columns in train that are absent from holdout (excluding target)."""
    extra_cols = [
        c for c in train_df.columns
        if c not in holdout_df.columns and c != TARGET_COL
    ]
    correlations = {}
    for col in extra_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            correlations[col] = float(train_df[col].corr(train_df[TARGET_COL]))
    return {
        "present": len(extra_cols) > 0,
        "extra_columns": extra_cols,
        "correlations_with_target": correlations,
    }


def check_unit_mixup(
    train_df: pd.DataFrame, later_df: pd.DataFrame
) -> dict:
    """Compare per-shift numeric summaries to spot 10x/12x/60x/100x/1000x unit mixups."""
    clean_shift = train_df[GROUP_COL].astype(str).str.strip().str.title()
    work = train_df.copy()
    work["_shift_clean"] = clean_shift

    numeric_cols = [
        c for c in train_df.select_dtypes(include="number").columns
        if c != TARGET_COL
    ]
    findings = []
    for col in numeric_cols:
        grp = work.groupby("_shift_clean")[col].agg(["min", "mean", "median", "max"])
        later_mean = float(later_df[col].mean())
        medians = grp["median"]
        max_med = float(medians.max())
        min_med = float(medians.min())
        ratio = max_med / min_med if min_med > 0 else float("inf")
        if ratio >= 5.0:
            findings.append(
                {
                    "column": col,
                    "group_stats": grp.round(3).to_dict(orient="index"),
                    "later_mean": round(later_mean, 3),
                    "median_ratio": round(ratio, 2),
                }
            )
    return {"present": len(findings) > 0, "candidates": findings}


def check_sentinel_value(
    train_df: pd.DataFrame, later_df: pd.DataFrame
) -> dict:
    """Detect repeated impossible placeholder values (e.g., -1, -999, 999, 9999)."""
    numeric_cols = [
        c for c in train_df.select_dtypes(include="number").columns
        if c != TARGET_COL
    ]
    hits = []
    for col in numeric_cols:
        s = train_df[col]
        later_min, later_max = float(later_df[col].min()), float(later_df[col].max())
        vc = s.value_counts()
        for val, cnt in vc.items():
            is_known = float(val) in KNOWN_SENTINELS
            is_neg_impossible = float(val) < 0 and later_min >= 0
            if (is_known or is_neg_impossible) and cnt >= 3:
                hits.append(
                    {
                        "column": col,
                        "value": float(val) if not float(val).is_integer() else int(val),
                        "count": int(cnt),
                        "later_range": [later_min, later_max],
                    }
                )
    return {"present": len(hits) > 0, "hits": hits}


def check_duplicate_rows(train_df: pd.DataFrame) -> dict:
    """Count exact extra duplicate rows in train.csv."""
    extra_copies = int(train_df.duplicated().sum())
    dup_ids = int(train_df["record_id"].duplicated().sum())
    return {
        "present": extra_copies > 0,
        "extra_copies": extra_copies,
        "duplicate_record_ids": dup_ids,
    }


def check_stuck_sensor(train_df: pd.DataFrame) -> dict:
    """Sort by date and look for long consecutive runs of an identical reading."""
    ordered = train_df.sort_values(["date", "record_id"]).reset_index(drop=True)
    numeric_cols = [
        c for c in ordered.select_dtypes(include="number").columns
        if c != TARGET_COL
    ]
    candidates = []
    for col in numeric_cols:
        vals = ordered[col]
        run_ids = (vals != vals.shift()).cumsum()
        runs = (
            ordered.assign(_run=run_ids)
            .groupby("_run")
            .agg(
                value=(col, "first"),
                rows=(col, "size"),
                distinct_days=("date", "nunique"),
                start_date=("date", "min"),
                end_date=("date", "max"),
            )
        )
        longest = runs.sort_values(["rows", "distinct_days"], ascending=False).iloc[0]
        if int(longest["rows"]) >= 15 and int(longest["distinct_days"]) >= 5:
            candidates.append(
                {
                    "column": col,
                    "stuck_value": float(longest["value"]),
                    "rows": int(longest["rows"]),
                    "distinct_days": int(longest["distinct_days"]),
                    "start_date": str(longest["start_date"]),
                    "end_date": str(longest["end_date"]),
                }
            )
    return {"present": len(candidates) > 0, "candidates": candidates}


def check_target_typos(
    train_df: pd.DataFrame, later_df: pd.DataFrame
) -> dict:
    """Look for extreme high outliers in pick_seconds (~10x above normal range)."""
    s = train_df[TARGET_COL].sort_values(ascending=False).reset_index(drop=True)
    later_max = float(later_df[TARGET_COL].max())
    # Look for a jump > 2x between adjacent sorted target values above later_max
    ratios = s / s.shift(-1)
    jump_idx = ratios[(s > later_max) & (ratios > 2.0)].index.tolist()
    if jump_idx:
        cutoff_idx = jump_idx[0]
        typo_vals = s.iloc[: cutoff_idx + 1]
        return {
            "present": True,
            "count": int(len(typo_vals)),
            "min_typo_value": float(typo_vals.min()),
            "max_normal_value": float(s.iloc[cutoff_idx + 1]),
            "later_max": later_max,
        }
    return {
        "present": False,
        "count": 0,
        "top_5_train": s.head(5).tolist(),
        "later_max": later_max,
    }


def check_spelling_variants(
    train_df: pd.DataFrame, later_df: pd.DataFrame
) -> dict:
    """Compare categorical column values in train vs clean later.csv."""
    text_cols = [
        c for c in train_df.select_dtypes(include=["object", "string"]).columns
        if c not in ID_COLS
    ]
    issues = []
    for col in text_cols:
        train_counts = train_df[col].value_counts(dropna=False).to_dict()
        clean_levels = sorted(later_df[col].dropna().unique().tolist())
        unexpected = {
            k: int(v) for k, v in train_counts.items() if k not in clean_levels
        }
        if unexpected:
            issues.append(
                {
                    "column": col,
                    "clean_levels": clean_levels,
                    "true_levels": len(clean_levels),
                    "unexpected_values": unexpected,
                    "all_train_counts": {str(k): int(v) for k, v in train_counts.items()},
                }
            )
    return {"present": len(issues) > 0, "issues": issues}


def run_audit(raw_dir: str | Path = "data/raw") -> dict:
    """Run all seven data quality checks on raw CSVs."""
    raw_dir = Path(raw_dir)
    train_df = pd.read_csv(raw_dir / "train.csv")
    later_df = pd.read_csv(raw_dir / "later.csv")
    holdout_df = pd.read_csv(raw_dir / "holdout_features.csv")

    return {
        "train_sha256": file_sha256(raw_dir / "train.csv"),
        "train_rows": int(len(train_df)),
        "leaking_column": check_leaking_column(train_df, holdout_df),
        "unit_mixup": check_unit_mixup(train_df, later_df),
        "sentinel_value": check_sentinel_value(train_df, later_df),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(train_df),
        "target_typos": check_target_typos(train_df, later_df),
        "spelling_variants": check_spelling_variants(train_df, later_df),
    }


if __name__ == "__main__":
    import pprint

    results = run_audit()
    pprint.pprint(results, sort_dicts=False)
