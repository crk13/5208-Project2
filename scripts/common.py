import os

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

DEFAULT_DATASET_PATH = os.environ.get("DATASET_PATH", "../spark-result/clean")
DEFAULT_TRAIN_PATH = os.environ.get("TRAIN_DATA_PATH", "gs://spark-result/train_withds")
DEFAULT_TEST_PATH = os.environ.get("TEST_DATA_PATH", "gs://spark-result/test_withds")


def create_spark(app_name="WeatherForecast"):
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.eventLog.enabled", "false")
            .getOrCreate())

def load_dataset(spark, path=None):
    dataset_path = path or DEFAULT_DATASET_PATH
    return spark.read.parquet(dataset_path)

def load_train_test(spark, train_path=None, test_path=None):
    train_dataset_path = train_path or DEFAULT_TRAIN_PATH
    test_dataset_path = test_path or DEFAULT_TEST_PATH
    train_df = spark.read.parquet(train_dataset_path)
    test_df = spark.read.parquet(test_dataset_path)
    return train_df, test_df

def build_feature_pipeline(features, label, normalizer="minmax"):
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    if normalizer == "standard":
        from pyspark.ml.feature import StandardScaler
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    else:
        scaler = MinMaxScaler(inputCol="features_raw", outputCol="features")
    return Pipeline(stages=[assembler, scaler]), label
