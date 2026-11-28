# SCSE3040 MLOps Mini Project — Group G050, Problem MP308

**Problem:** MP308 — Warehouse Picking Time (Dehradun Fulfilment Centre)
**Faculty:** Dr. Avantika Saklani | **Batch:** 73

## Group members

| Enrolment No. | Name |
|---|---|
| S24CSEU2805 | Ridit Saxena |
| S24CSEU2818 | Shikhar Verma |
| S24CSEU2807 | Md Amaan Haque |
| S24CSEU0368 | Navnidh Malik |

## Problem

The fulfilment centre wants to predict how long an order takes to pick so slow
orders can be started early enough to catch the evening truck. This is a
supervised regression task. The target is **`pick_seconds`**, and models are
measured with mean absolute error (MAE), in seconds.

## Work by assignment stage

| Stage | Work |
|---|---|
| 1 — Data audit | Added DVC tracking files and code to check the seven data-quality issues. Recorded the findings in `reports/findings.yaml`. |
| 2 — Cleaning and training | Added configurable cleaning and preprocessing, evaluated four v1 model candidates with MLflow, registered the model, and added data-quality and preprocessing tests. |
| 3 — Service and deployment | Added the FastAPI service, request validation, JSON logging to stdout, Dockerfile, GitHub Actions workflow, and Kubernetes manifests. |
| 4 — Monitoring and predictions | Calculated feature PSI and per-shift MAE, retrained and evaluated v2 using the promotion rule, and generated holdout predictions. |

## Repository layout

```text
src/warehouse_mlops/   data audit, preprocessing, training, monitoring, prediction
app/main.py            FastAPI service: GET /health and POST /predict
tests/                 pytest tests, including API tests with TestClient
data/raw/              DVC pointer files and holdout_features.csv
models/model.joblib    champion model loaded by the API
reports/findings.yaml  data, tracking, and monitoring findings
predictions/           holdout_predictions.csv
k8s/                   Kubernetes Deployment and Service
```

## Setup and data

```cmd
git clone https://github.com/Ridit-007/G050_MP308-Warehouse-Picking-MLOps.git
cd G050_MP308-Warehouse-Picking-MLOps
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

`train.csv` and `later.csv` are supplied in the original group archive and are
not committed to this public repository. Copy them into `data/raw/` before
running the audit, training, or monitoring commands. The DVC pointer files
record their expected content hashes. A shared DVC remote is not configured;
the local DVC remote used by the maintainer is machine-specific, so `dvc pull`
will not restore the files for a fresh clone. Keep the raw CSVs and local
`dvc-store/` out of Git, and do not share the data or code with other groups.

Do not open and save the CSVs in Excel; that can change their contents and
break the data hash check.

## Pipeline commands

Run these from the repository root after setup and placing the supplied data
in `data/raw/`.

| Step | Command | Result |
|---|---|---|
| Data audit | `python -m warehouse_mlops.data_quality` | Prints checks for the seven data issues. |
| Tests | `python -m pytest -v` | Runs the project tests. |
| Train v1 | `python -m warehouse_mlops.train` | Evaluates four candidates and records runs in MLflow. |
| Drift and retraining | `python -m warehouse_mlops.monitoring` | Computes PSI and per-shift MAE, evaluates v2, and applies the promotion rule. |
| Holdout prediction | `python -m warehouse_mlops.predict` | Writes `predictions/holdout_predictions.csv`. |
| Run the API | `uvicorn app.main:app --host 0.0.0.0 --port 8000` | Serves `/health` and `/predict`. |
| Build the container | `docker build -t mp308-api:1.0 .` | Builds the required API image. |

MLflow uses the SQLite tracking database `mlflow.db`. The database contains
the recorded runs and registered model versions; the `champion` alias points
to the model selected by the promotion rule.

## Data problems found in `train.csv`

| Kind | Present | Detail |
|---|---|---|
| `leaking_column` | Yes | `pick_time_from_scanner`, which is absent from the holdout features. |
| `unit_mixup` | No | No group was found with a unit-scale difference. |
| `sentinel_value` | Yes | `shift_hour = -1` appears in 55 rows. |
| `duplicate_rows` | Yes | 31 extra duplicate rows. |
| `stuck_sensor` | Yes | `shelf_height_m = 1.35` from 2026-05-09 through 2026-05-24. |
| `target_typos` | No | No 10x target typo cluster was found. |
| `spelling_variants` | Yes | `zone_type` contains spelling and formatting variants; there are 3 real levels. |

The checks are implemented in `src/warehouse_mlops/data_quality.py`.

## API request rules

`POST /predict` accepts exactly these fields: `items_in_order`,
`aisle_distance_m`, `picker_exp_months`, `shelf_height_m`, `shift_hour`,
`order_weight_kg`, `zone_type`, and `shift`. The schema rejects weights above
45 kg and negative picker experience with HTTP 422. `GET /health` returns
`{"status": "ok"}`. Each prediction emits one JSON log line to stdout.

## Monitoring results

The monitoring code compares cleaned train and later data with 10-bin PSI,
measures v1 MAE by shift, and evaluates v2 on the last 30% of later data. The
promotion rule requires v2 MAE to be at least 10% lower on that check set.
The computed values and decision are recorded in `reports/findings.yaml`.
## Stage 3 — Service and deployment

- **API:** `app/main.py` provides `GET /health` and `POST /predict`.
- **Request validation:** `/predict` accepts the eight required fields, rejects extra fields, rejects `order_weight_kg` above 45, and rejects negative `picker_exp_months`.
- **Logging:** Each prediction writes a JSON log line to stdout.
- **Docker:** The Dockerfile builds the `mp308-api:1.0` image and serves the API on port 8000.
- **CI:** `.github/workflows/ci.yml` runs the pytest suite on pushes. Check the repository’s Actions tab for the latest run.
- **Kubernetes:** `mp308-api` runs four replicas; `mp308-svc` exposes port 80 and forwards to port 8000. Readiness and liveness probes use `/health`.
- **Tests:** The local pytest run passed all nine tests.
