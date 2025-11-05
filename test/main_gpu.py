import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.evaluation import RegressionEvaluator

# ✅ GPU version model
from spark_rapids_ml.regression import RandomForestRegressor as GPU_RF
from spark_rapids_ml.regression import GBTRegressor as GPU_GBT
from spark_rapids_ml.regression import LinearRegression as GPU_LR

import time


numeric_features = [
        "dt_min", "ELEVATION", "DEW", "VIS", "CIG", "SLP", "WND_Speed",
        "TMP_lag1", "time_speed", "time_a", "WND_sin", "WND_cos",
        "month_sin", "month_cos", "day_sin", "day_cos",
        "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        "lat_sin", "lat_cos", "lon_sin", "lon_cos"
    ]
LABEL = "TMP"
TIMESTAMP_COL = "DATE_TS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "gbrt", "elastic"], required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--bucket", default="spark-result")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("WeatherGPU")
        .config("spark.rapids.sql.enabled", "true")
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.executor.resource.gpu.amount", "1")
        .config("spark.task.resource.gpu.amount", "0.1")
        .getOrCreate()
    )

    train_df = spark.read.parquet(args.train_path).drop("STATION")
    test_df  = spark.read.parquet(args.test_path).drop("STATION")
    train_df = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(DoubleType()))
    test_df  = test_df.withColumn("ELEVATION", test_df["ELEVATION"].cast(DoubleType()))

    assembler = VectorAssembler(inputCols=numeric_features, outputCol="features_raw")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")

    # ✅ Fill in the model best params
    if args.model == "rf":
        best_params = {
            "numTrees": 200,
            "maxDepth": 10,
            "minInstancesPerNode": 5,
        }
        estimator_builder = GPU_RF(labelCol=LABEL, featuresCol="features", **best_params)

    elif args.model == "gbrt":
        best_params = { 
            "maxDepth": 4,
            "maxIter": 70,
            "stepSize": 0.2,
            "subsamplingRate": 1.0,
            "minInstancesPerNode": 5
        }
        estimator_builder = GPU_GBT(labelCol=LABEL, featuresCol="features", **best_params)

    elif args.model == "elastic":
        best_params = {
            "regParam": 0.00025,
            "elasticNetParam": 0.1,
            "maxIter": 300,
        }
        estimator_builder = GPU_LR(labelCol=LABEL, featuresCol="features", **best_params)


    pipeline = Pipeline(stages=[assembler, scaler, estimator_builder])

    # ✅ GPU time cost
    start_time = time.time()
    model = pipeline.fit(train_df)
    end_time = time.time()
    print(f"[GPU Train] time: {end_time - start_time:.2f} seconds")

    start_time1 = time.time()
    preds = model.transform(test_df)
    end_time1 = time.time()
    print(f"[GPU Infer] time: {end_time1 - start_time1:.2f} seconds")

    evaluator = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse")
    evaluator_mae  = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="mae")
    evaluator_r2   = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="r2")

    ### ===== Train metrics =====
    train_preds = final_model.transform(train_df)
    train_rmse = evaluator.evaluate(train_preds)
    train_mae  = evaluator_mae.evaluate(train_preds)
    train_r2   = evaluator_r2.evaluate(train_preds)
    print(f"[GPU Train] RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")

    ### ===== Test metrics =====
    preds = final_model.transform(test_df)
    test_rmse = evaluator.evaluate(preds)
    test_mae  = evaluator_mae.evaluate(preds)
    test_r2   = evaluator_r2.evaluate(preds)
    print(f"[GPU Test] RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")


if __name__ == "__main__":
    main()
