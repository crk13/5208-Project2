gcloud dataproc jobs submit pyspark \
    test/visualize.py \
    --cluster=mycluster \
    --region=asia-southeast1 \
    --py-files=src/time_cv.py,src/utils.py \
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