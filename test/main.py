import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.time_cv import prefix_folds
from src.model_selection import grid_search_prefix_cv

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
    parser.add_argument("--model", choices=["rf", "gbrt"], required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--num-folds", type=int, default=4)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WeatherForecast").getOrCreate()
    train_df = spark.read.parquet(args.train_path).drop("STATION") 
    train_df = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(DoubleType()))
    test_df = spark.read.parquet(args.test_path).drop("STATION") 
    test_df = test_df.withColumn("ELEVATION", test_df["ELEVATION"].cast(DoubleType())) 

    train_sample = train_df.sample(fraction=args.sample_fraction, seed=42)
    train_sample = train_sample.orderBy(TIMESTAMP_COL).cache()
    folds = prefix_folds(train_sample, TIMESTAMP_COL, num_folds=args.num_folds)
    evaluator = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse")


    assembler = VectorAssembler(inputCols=numeric_features, outputCol="features_raw")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
    # scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    base_stages = [assembler, scaler]

    if args.model == "rf":
        param_grid = [
            {"numTrees":100, "maxDepth":10, "subsamplingRate":0.8, "featureSubsetStrategy":"auto", "minInstancesPerNode":5},
            {"numTrees":200, "maxDepth":12, "subsamplingRate":0.9, "featureSubsetStrategy":"auto", "minInstancesPerNode":5},
        ]
        estimator_builder = lambda **p: RandomForestRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p)
    else:
        param_grid = [
            {"maxDepth":5, "maxIter":80, "stepSize":0.1, "maxBins":32, "subsamplingRate":1.0, "minInstancesPerNode":5},
            {"maxDepth":7, "maxIter":120, "stepSize":0.1, "maxBins":64, "subsamplingRate":0.8, "minInstancesPerNode":5},
        ]
        estimator_builder = lambda **p: GBTRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p)

    best_params, grid_results = grid_search_prefix_cv(folds, base_stages, estimator_builder, param_grid, evaluator)
    final_pipeline = Pipeline(stages=base_stages + [estimator_builder(**best_params)])
    final_model = final_pipeline.fit(train_df)
    test_rmse = evaluator.evaluate(final_model.transform(test_df))
    print(f"Test RMSE: {test_rmse:.4f}")
    final_model.write().overwrite().save(f"models/{args.model}_model")


if __name__ == "__main__":
    main()
