# Weather Forecasting with PySpark

End-to-end workflow for training and evaluating weather temperature forecasting models on Spark.  
The project builds feature pipelines, performs time-aware cross-validation, trains multiple regressors (Random Forest, Gradient-Boosted Trees, Elastic Net), and submits Dataproc jobs that stream results and plots back to Google Cloud Storage (GCS).

## Repository Layout
- `src/` &nbsp;Reusable Spark pipeline utilities (feature assembly, cross-validation, plotting helpers).
- `scripts/` &nbsp;Command-line utilities for data extraction, feature engineering, and Dataproc submission (`submit_main.sh`, `submit_visualize.sh`).
- `test/` &nbsp;Entry points executed on Dataproc (`main.py` for training, `visualize.py` for parameter scans, `main_gpu.py` experimental).
- `models/` *(created at runtime)* &nbsp;Persisted Spark pipelines; mirrored to GCS.

## Prerequisites
- Python 3.10+ with `pyspark`, `google-cloud-storage`, `matplotlib`, `pandas`.
- Google Cloud CLI (`gcloud`) and `gsutil`.
- Access to a GCS bucket containing the processed parquet datasets (defaults assume `gs://spark-result-lyx`).
- A Dataproc cluster (or permissions to create one) in `asia-southeast1`.

Install Python packages locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyspark google-cloud-storage pandas matplotlib
```

## Data Locations
Default paths are managed in `scripts/common.py`:

```
TRAIN_DATA_PATH=gs://spark-result-lyx/train_withds
TEST_DATA_PATH=gs://spark-result-lyx/test_withds
```

Override them via environment variables when running locally:

```bash
export TRAIN_DATA_PATH=gs://my-bucket/train
export TEST_DATA_PATH=gs://my-bucket/test
```

or pass explicit `--train-path / --test-path` flags to the entry script.

## Running Locally
The local runner mirrors the Dataproc job but uses your active PySpark installation:

```bash
python test/main.py \
  --model elastic \
  --train-path gs://my-bucket/train_withds \
  --test-path gs://my-bucket/test_withds \
  --sample-fraction 0.001 \
  --num-folds 4 \
  --bucket my-bucket
```

Outputs:
- Metrics logged to stdout (RMSE, MAE, R²).
- Diagnostic plots uploaded to `gs://<bucket>/plots/<model>/`.
- Trained pipeline stored under `models/<model>_model/` locally and mirrored to `gs://<bucket>/models/<model>_model/`.

## Submitting to Dataproc

1. **Create (or reuse) a cluster** – the helper script has the command commented near the top:

```bash
gcloud dataproc clusters create my-cluster \
  --region=asia-southeast1 \
  --num-workers=2 \
  --worker-machine-type=n2-standard-4 \
  --master-machine-type=n2-standard-4 \
  --image-version=2.2-debian12 \
  --optional-components=JUPYTER \
  --enable-component-gateway
```

2. **Run the training job** for Random Forest (`rf`), Gradient-Boosted Trees (`gbrt`), or Elastic Net (`elastic`):

```bash
chmod +x scripts/submit_main.sh
./scripts/submit_main.sh elastic
```

The script:
- Packages `src/` into `src.zip`.
- Submits `test/main.py` as a PySpark job.
- Streams logs to `/tmp/job_<model>.log` and copies them to `gs://spark-result-lyx/logs/`.
- Uploads the trained pipeline artifacts to `gs://spark-result-lyx/models/<model>_model/`.
- Deletes the cluster at the end (remove the command if you plan to keep the cluster running).

Use `scripts/submit_visualize.sh` to run parameter scans and push plots to the bucket.

## Key Components

- **Feature Engineering** (`scripts/features.py`, `scripts/data_processing*.py`): transforms raw weather observations into time-aware sin/cos features, lag variables, and normalization-ready columns.
- **Time-aware Cross-Validation** (`src/time_cv.py`): builds prefix-based folds that respect chronological order.
- **Model Selection** (`src/model_selection.py`): implements grid search with prefix folds, collecting average RMSE per parameter set.
- **Visualization** (`src/plot.py`, `test/visualize.py`): residuals, prediction vs. actual, time series comparisons, and parameter effect plots saved to GCS.

## Customising Parameter Grids
`test/main.py` contains the search grids for each model type. Adjust the `param_grid` dictionaries to probe additional hyper-parameters (e.g., deeper GBRT trees, finer Elastic Net regularization). The results table printed after cross-validation summarizes average RMSE for each candidate.

## Troubleshooting
- `ModuleNotFoundError: No module named 'pyspark'` → install PySpark in the active environment (`pip install pyspark`).
- `NOT_FOUND: Cluster ...` → create the Dataproc cluster or update the cluster name in `scripts/submit_main.sh`.
- Rate limit messages while writing to `dataproc-temp-*` → informational; Dataproc flushes logs once the job finishes.

## Cleaning Up
Remove temporary job artifacts to avoid extra storage costs:

```bash
rm -rf src.zip models/
gsutil rm -r gs://spark-result-lyx/logs/
```

Delete Dataproc clusters when idle:

```bash
gcloud dataproc clusters delete my-cluster --region=asia-southeast1
```

---

This README captures the current workflow; keep bucket names, regions, and parameter grids in sync with your team’s environment as needed.
