# test/visualize.py
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.time_cv import prefix_folds
from src.utils import single_param_scan, plot_and_upload

LABEL = "TMP"
TIMESTAMP_COL = "DATE_TS"
NUMERIC_FEATURES = [
    "dt_min","ELEVATION","DEW","VIS","CIG","SLP","WND_Speed",
    "TMP_lag1","time_speed","time_a","WND_sin","WND_cos",
    "month_sin","month_cos","day_sin","day_cos",
    "hour_sin","hour_cos","minute_sin","minute_cos",
    "lat_sin","lat_cos","lon_sin","lon_cos"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--bucket", default="spark-result")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("VisualizeParamEffect").getOrCreate()

    train_df = spark.read.parquet(args.train_path).drop("STATION")
    train_df = train_df.withColumn("ELEVATION", train_df["ELEVATION"].cast(DoubleType()))
    test_df = spark.read.parquet(args.test_path).drop("STATION")
    test_df = test_df.withColumn("ELEVATION", test_df["ELEVATION"].cast(DoubleType()))

    train_sample = train_df.sample(fraction=args.sample_fraction, seed=42)
    print("Sampled rows:", train_sample.count())
    train_sample = train_sample.orderBy(TIMESTAMP_COL).cache()
    folds = prefix_folds(train_sample, TIMESTAMP_COL, num_folds=args.num_folds)
    evaluator = RegressionEvaluator(labelCol=LABEL, predictionCol="prediction", metricName="rmse")


    assembler = VectorAssembler(inputCols=NUMERIC_FEATURES, outputCol="features_raw")
    scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
    # scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    base_stages = [assembler, scaler]

    # 定义模型和参数网格
    models = {
        "gbrt": lambda **p: GBTRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p),
        #"rf": lambda **p: RandomForestRegressor(labelCol=LABEL, featuresCol="features", seed=42, **p),
        #"elasticnet": lambda **p: LinearRegression(labelCol=LABEL, featuresCol="features", **p)
    }

    param_grids = {
        "gbrt": {
        "maxDepth": [3,5,7],
        "maxIter": [50,100,150],
        "stepSize": [0.05,0.1,0.2],
        "maxBins": [16,32,64],
        "subsamplingRate": [0.7,0.8,1.0],
        "minInstancesPerNode": [5,10]
    },
        #"rf": {"numTrees":[50,100,200], "maxDepth":[5,10,15]},
        #"elasticnet": {"regParam":[0.01,0.1,1.0], "elasticNetParam":[0.0,0.5,1.0]}
    }

    # 对每个模型进行单参数扫描 + 可视化
    for model_name, estimator_builder in models.items():
        print(f"=== Processing {model_name} ===")
        for param_name, param_values in param_grids[model_name].items():
            x, y = single_param_scan(folds, base_stages, estimator_builder, param_name, param_values)
            plot_and_upload(x, y, param_name, model_name, bucket_name=args.bucket)
            print("Using param:", param_name, "with values:", param_values)



if __name__ == "__main__":
    main()
