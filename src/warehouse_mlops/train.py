"""Stage 2 training, MLflow tracking, and model registry for Group G050 (MP308)."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from warehouse_mlops.data_quality import file_sha256
from warehouse_mlops.preprocessing import (
    build_preprocessor,
    clean_dataframe,
    get_feature_columns,
    load_params,
)

MODEL_REGISTRY = {
    "Ridge": Ridge,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
}


def build_estimator(name: str, model_params: dict):
    """Instantiate a scikit-learn regressor from config."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {name}")
    return MODEL_REGISTRY[name](**model_params)


def update_findings_tracking(
    findings_path: str | Path,
    experiment_name: str,
    best_run_id: str,
    registered_model_name: str,
    champion_version: int,
) -> None:
    """Update the tracking section in reports/findings.yaml."""
    with open(findings_path, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    findings["tracking"] = {
        "experiment_name": experiment_name,
        "best_run_id": best_run_id,
        "registered_model_name": registered_model_name,
        "champion_version": int(champion_version),
    }

    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)


def train_v1(params_path: str | Path = "params.yaml") -> dict:
    """Train candidate v1 models, track with MLflow, and register the champion."""
    params = load_params(params_path)
    raw_train_path = Path(params["data"]["raw_dir"]) / params["data"]["train_file"]
    data_sha256 = file_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    cleaned_df = clean_dataframe(raw_df, params, is_training=True)

    feature_cols = get_feature_columns(params)
    target_col = params["features"]["target"]
    X = cleaned_df[feature_cols]
    y = cleaned_df[target_col]

    mlflow_cfg = params["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    cv = KFold(
        n_splits=params["training"]["cv_folds"],
        shuffle=True,
        random_state=params["training"]["random_state"],
    )

    best_result = None
    run_summaries = []

    for candidate in params["training"]["candidates"]:
        model_name = candidate["name"]
        model_params = candidate.get("params", {})

        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(params)),
                ("model", build_estimator(model_name, model_params)),
            ]
        )

        cv_scores = cross_val_score(
            pipe, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        mae_cv = float(-cv_scores.mean())

        pipe.fit(X, y)

        with mlflow.start_run(run_name=f"v1_{model_name}") as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("model_stage", "v1")
            mlflow.log_param("train_rows_cleaned", len(cleaned_df))
            for k, v in model_params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("mae_cv", mae_cv)

            mlflow.sklearn.log_model(
                sk_model=pipe,
                name="model",
                serialization_format="cloudpickle",
            )

            summary = {
                "run_id": run.info.run_id,
                "model_type": model_name,
                "mae_cv": round(mae_cv, 4),
            }
            run_summaries.append(summary)

            if best_result is None or mae_cv < best_result["mae_cv"]:
                best_result = {
                    **summary,
                    "pipeline": pipe,
                }

    registered_name = mlflow_cfg["registered_model_name"]
    model_uri = f"runs:/{best_result['run_id']}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registered_name)

    client = MlflowClient()
    alias = mlflow_cfg["champion_alias"]
    existing = None
    try:
        existing = client.get_model_version_by_alias(registered_name, alias)
    except Exception:
        existing = None

    if existing is not None:
        # A champion is already promoted (for example a retrained v2): never
        # overwrite it by re-running this script.
        champion_version = int(existing.version)
    else:
        client.set_registered_model_alias(registered_name, alias, str(mv.version))
        champion_version = int(mv.version)

        model_out = Path(params["output"]["model_path"])
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_result["pipeline"], model_out)

    update_findings_tracking(
        findings_path=params["output"]["findings_path"],
        experiment_name=mlflow_cfg["experiment_name"],
        best_run_id=best_result["run_id"],
        registered_model_name=registered_name,
        champion_version=champion_version,
    )

    return {
        "data_sha256": data_sha256,
        "runs": run_summaries,
        "best_run_id": best_result["run_id"],
        "best_model": best_result["model_type"],
        "best_mae_cv": best_result["mae_cv"],
        "champion_version": champion_version,
    }


if __name__ == "__main__":
    import pprint

    result = train_v1()
    pprint.pprint(result, sort_dicts=False)
