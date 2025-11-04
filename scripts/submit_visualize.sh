set -e

CLUSTER_NAME="my-cluster"
NUM_WORKERS=2
MASTER_DISK=100
WORKER_DISK=100


gcloud dataproc clusters create $CLUSTER_NAME \
  --region=asia-southeast1 \
  --num-workers=$NUM_WORKERS \
  --worker-machine-type=n2-standard-4 \
  --master-machine-type=n2-standard-4 \
  --master-boot-disk-size=$MASTER_DISK \
  --worker-boot-disk-size=$WORKER_DISK \
  --image-version="2.2-debian12" \
  --optional-components=JUPYTER \
  --enable-component-gateway \

gcloud dataproc jobs submit pyspark \
    test/visualize.py \
    --cluster=mycluster \
    --region=asia-southeast1 \
    --py-files="src/" \
    -- \
    --train-path="gs://spark-result/train_withds1/" \
    --test-path="gs://spark-result/test_withds1/" \
    --sample-fraction=0.01 \
    --num-folds=4 \
    --bucket="spark-result"

gsutil cp /tmp/job_${MODEL}.log "$LOG_PATH"
echo "Logs saved to: $LOG_PATH"


gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
echo "✅ Cluster deleted."