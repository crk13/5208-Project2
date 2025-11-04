from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, DoubleType,
)
from pyspark.storagelevel import StorageLevel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
import os


input_path = "gs://spark-result/clean"
output_bucket = "gs://spark-result"

spark = SparkSession.builder.appName("WeatherProcessing").getOrCreate()

print("Step 0: Reading files from", input_path)
df_filtered = spark.read.parquet(input_path)

# ---------
# Step 6: Train/test split
# ---------
min_date, max_date = df_filtered.agg(
    F.min("DATE_TS").alias("min"),
    F.max("DATE_TS").alias("max")
).first()

cutoff = min_date + (max_date - min_date) * 0.7

train_df = df_filtered.filter(F.col("DATE_TS") <= cutoff)
test_df  = df_filtered.filter(F.col("DATE_TS") > cutoff)
print("Step 8: Train count:", train_df.count(), "Test count:", test_df.count())

train_df.write.mode("overwrite").parquet(os.path.join(output_bucket, "train_withds"))
test_df.write.mode("overwrite").parquet(os.path.join(output_bucket, "test_withds"))
print("All steps completed. CSVs saved to", output_bucket)

