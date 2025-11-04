#!/bin/bash
# chmod +x submit.sh
# ./submit.sh

set -e

CLUSTER_NAME="my-cluster"
REGION="asia-southeast1"
BUCKET1="weather-2024"
BUCKET2="spark-result"
PY_FILES="gs://$BUCKET1/scripts/common.py,gs://$BUCKET1/scripts/features.py,gs://$BUCKET1/scripts/model_selection.py,gs://$BUCKET1/scripts/time_cv.py"

TRAIN_GBRT_PATH="gs://$BUCKET1/scripts/train_gbrt.py"
TRAIN_GBRT_LOG_PATH="gs://$BUCKET2/logs/train_gbrt.log"

TRAIN_RF_PATH="gs://$BUCKET1/scripts/train_random_forest.py"
TRAIN_RF_LOG_PATH="gs://$BUCKET2/logs/train_random_forest.log"

TRAIN_EN_PATH="gs://$BUCKET1/scripts/train_elastic_net.py"
TRAIN_EN_LOG_PATH="gs://$BUCKET2/logs/train_elastic_net.log"

MASTER_MACHINE_TYPE="n2-standard-2"
WORKER_MACHINE_TYPE="n2-standard-2"
NUM_WORKERS=3
MASTER_DISK=100
WORKER_DISK=100
TRAIN_DATA_PATH="gs://$BUCKET2/train_withds"
TEST_DATA_PATH="gs://$BUCKET2/test_withds"

CLUSTER_CREATED=0

cleanup() {
  if [[ $CLUSTER_CREATED -eq 1 ]]; then
    echo "Deleting cluster $CLUSTER_NAME..."
    gcloud dataproc clusters delete "$CLUSTER_NAME" \
      --region="$REGION" \
      --quiet || echo "Warning: failed to delete cluster $CLUSTER_NAME"
  fi
}

trap cleanup EXIT

if gcloud dataproc clusters describe "$CLUSTER_NAME" --region="$REGION" > /dev/null 2>&1; then
  echo "Cluster $CLUSTER_NAME already exists. Please delete it or choose a new CLUSTER_NAME."
  exit 1
fi

echo "Creating cluster $CLUSTER_NAME..."
gcloud dataproc clusters create "$CLUSTER_NAME" \
  --region="$REGION" \
  --num-workers="$NUM_WORKERS" \
  --worker-machine-type="$WORKER_MACHINE_TYPE" \
  --worker-boot-disk-size="$WORKER_DISK" \
  --master-machine-type="$MASTER_MACHINE_TYPE" \
  --master-boot-disk-size="$MASTER_DISK"

CLUSTER_CREATED=1

submit_job() {
  local script_path="$1"
  local log_dest="$2"
  local local_log="$3"

  gcloud dataproc jobs submit pyspark "$script_path" \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --py-files="$PY_FILES" \
    --properties="spark.driverEnv.TRAIN_DATA_PATH=$TRAIN_DATA_PATH,spark.executorEnv.TRAIN_DATA_PATH=$TRAIN_DATA_PATH,spark.driverEnv.TEST_DATA_PATH=$TEST_DATA_PATH,spark.executorEnv.TEST_DATA_PATH=$TEST_DATA_PATH" \
    --jars="gs://hadoop-lib/gcs/gcs-connector-latest-hadoop3.jar" \
    > >(tee "$local_log") 2>&1

  gsutil cp "$local_log" "$log_dest"
  echo "Logs saved to: $log_dest"
}


submit_job "$TRAIN_GBRT_PATH" "$TRAIN_GBRT_LOG_PATH" "/tmp/job_train_gbrt.log"

if gsutil ls "$TRAIN_RF_PATH" > /dev/null 2>&1; then
  submit_job "$TRAIN_RF_PATH" "$TRAIN_RF_LOG_PATH" "/tmp/job_train_rf.log"
else
  echo "Skipping Random Forest training, script not found at $TRAIN_RF_PATH"
fi

if gsutil ls "$TRAIN_EN_PATH" > /dev/null 2>&1; then
  submit_job "$TRAIN_EN_PATH" "$TRAIN_EN_LOG_PATH" "/tmp/job_train_elastic.log"
else
  echo "Skipping Elastic Net training, script not found at $TRAIN_EN_PATH"
fi

echo "All jobs submitted. Cluster $CLUSTER_NAME will be cleaned up automatically."
