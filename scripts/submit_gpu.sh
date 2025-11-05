#!/bin/bash
# Usage:
# chmod +x ./scripts/submit_gpu.sh
# ./scripts/submit_gpu.sh gbrt

set -e

MODEL="$1"  # Param: rf, gbrt or elastic
if [[ -z "$MODEL" || ! "$MODEL" =~ ^(rf|gbrt|elastic)$ ]]; then
  echo "Usage: $0 <rf|gbrt|elastic>"
  exit 1
fi

CLUSTER_NAME="gpu-cluster"
REGION="asia-southeast1"
NUM_GPUS=1
NUM_WORKERS=2
DISK_SIZE=100
IS=gs://goog-dataproc-initialization-actions-${REGION}/spark-rapids/spark-rapids.sh
PKG="spark-rapids-ml==25.08.0,cuml-cu12==25.10.0,cuvs-cu12==25.10.0,ucx-py-cu12==0.45.0"
ENV="CUPY_CACHE_DIR=/tmp/cupy_cache"
gcloud dataproc clusters create $CLUSTER_NAME \
--region $REGION --image-version 2.3-debian12 \
--public-ip-address --enable-component-gateway \
--master-machine-type n2-standard-4 \
--master-boot-disk-size $DISK_SIZE \
--num-workers $NUM_WORKERS \
--worker-accelerator type=nvidia-tesla-t4,count=$NUM_GPUS \
--worker-machine-type n2-standard-4 \
--worker-boot-disk-size $DISK_SIZE \
--initialization-actions $IS --initialization-action-timeout 40m \
--no-shielded-secure-boot --metadata rapids-runtime=SPARK \
--properties="^#^dataproc:pip.packages=$PKG#spark:spark.executorEnv.$ENV"



REGION="asia-southeast1"
BUCKET2="spark-resulttt"
LOG_PATH="gs://$BUCKET2/logs/main_${MODEL}.log"

TRAINSET_PATH="gs://$BUCKET2/train_withds"
TESTSET_PATH="gs://$BUCKET2/test_withds"


zip -r src.zip ./src

gcloud dataproc jobs submit pyspark \
    test/main_gpu.py \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --py-files=src.zip \
  --jars="gs://hadoop-lib/gcs/gcs-connector-latest-hadoop3.jar" \
  -- \
  --model $MODEL \
  --train-path $TRAINSET_PATH \
  --test-path $TESTSET_PATH \
  --bucket=$BUCKET2 \
  > >(tee /tmp/job_${MODEL}.log) 2>&1

gsutil cp /tmp/job_${MODEL}.log "$LOG_PATH"
echo "Logs saved to: $LOG_PATH"

MODEL="$1"
LOCAL_PATH="models/${MODEL}_model"

gsutil -m cp -r $LOCAL_PATH gs://$BUCKET2/models/


gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
echo "✅ Cluster deleted."