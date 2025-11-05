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


input_path = "gs://weather-2024/csv/*.csv"
output_bucket = "gs://spark-result"

spark = (
    SparkSession.builder
    .appName("WeatherProcessing")
    .config("spark.eventLog.enabled", "false")
    .getOrCreate()
)


print("Step 0: Reading files from", input_path)

df = spark.read.option("header", True).csv(input_path)
print("Step 0: Total rows read:", df.count())



# ---------
# Step 1: Delete stations with too few rows
# ---------
station_record_counts = df.groupBy("STATION").count().persist()  # Count rows per station and cache the result

counts = station_record_counts.select("count").rdd.map(lambda r: r[0]).collect()
local_path = "/tmp/station_record_hist.png"
local_path1 = "/tmp/station_record_hist_10000.png"

plt.figure(figsize=(12,6))
plt.hist(counts, bins=5000, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.xlabel("Number of rows per station")
plt.ylabel("Number of stations")
plt.title("Distribution of rows per NOAA Station")
plt.tight_layout()
plt.savefig(local_path)
plt.close()

plt.figure(figsize=(12,6))
plt.hist(counts, bins=200, range=(0,10000), color='skyblue', edgecolor='black')
plt.xlabel("Number of rows per Station (0-10000)")
plt.ylabel("Number of stations")
plt.title("Distribution of rows per NOAA Station (0-10000 rows)")
plt.tight_layout()
plt.savefig(local_path1)
plt.close()

thres = station_record_counts.approxQuantile("count", [0.05], 0.01)[0] 
keep_stations_df = station_record_counts.filter(F.col("count") > thres).select("STATION")
df_filtered = df.join(keep_stations_df, on="STATION", how="inner")
print("Step 1: threshold for station rows (5% quantile):", thres)

num_rows_total = station_record_counts.agg(F.sum("count")).first()[0]
num_rows_after = df_filtered.count()
num_total_csv = station_record_counts.count()
num_removed_csv = num_total_csv - keep_stations_df.count()
log_text = (
    f"Number of rows: {num_rows_total}\n"
    f"Number of rows after filtering: {num_rows_after}\n"
    f"Total number of CSV files: {num_total_csv}\n"
    f"Number of removed CSV files: {num_removed_csv}\n"
)
print(log_text)

station_record_counts.unpersist()
df.unpersist()

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("spark-result")
blob = bucket.blob("station_record_hist.png")
blob1 = bucket.blob("station_record_hist_10000.png")
blob.upload_from_filename(local_path)
blob1.upload_from_filename(local_path1)


# ---------
# Step 2: Select relevant columns 
# ---------
cols = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "TMP", "DEW", "WND", "VIS", "CIG", "SLP"]
df_filtered = df_filtered.select(*cols)
print("Step 2: Selected columns:", df_filtered.columns)

df_filtered = (
    df_filtered
    .withColumn("TMP", F.regexp_replace(F.split(F.col("TMP"), ",")[0], "[+]", "").cast("double") / 10)
    .withColumn("DEW", F.regexp_replace(F.split(F.col("DEW"), ",")[0], "[+]", "").cast("double") / 10)
    .withColumn("WND_Dir", F.split(F.col("WND"), ",")[0].cast("double"))
    .withColumn("WND_Speed", F.split(F.col("WND"), ",")[3].cast("double") / 10)
    .withColumn("VIS", F.split(F.col("VIS"), ",")[0].cast("double"))
    .withColumn("CIG", F.split(F.col("CIG"), ",")[0].cast("double"))
    .withColumn("SLP", F.split(F.col("SLP"), ",")[0].cast("double") / 10)
    .drop("WND")
)


# ---------
# Step 3: Delete anomaly
# ---------
abnormal_counts = df_filtered.select([
    F.sum((F.col("TMP") == 9999/10).cast("int")).alias("TMP"),
    F.sum((F.col("DEW") == 9999/10).cast("int")).alias("DEW"),
    F.sum((F.col("WND_Dir") == 999).cast("int")).alias("WND_Dir"),
    F.sum((F.col("WND_Speed") == 9999/10).cast("int")).alias("WND_Speed"),
    F.sum((F.col("VIS") == 999999).cast("int")).alias("VIS"),
    F.sum((F.col("CIG") == 99999).cast("int")).alias("CIG"),
    F.sum((F.col("SLP") == 99999/10).cast("int")).alias("SLP"),
    F.sum((F.abs(F.col("LATITUDE")) == 999.999).cast("int")).alias("LATITUDE"),
    F.sum((F.abs(F.col("LONGITUDE")) == 999.999).cast("int")).alias("LONGITUDE"),
    F.sum((F.col("ELEVATION") == 9999.9).cast("int")).alias("ELEVATION")
])
print("Step 3: Before anomaly filter, anomaly summary: ")
abnormal_counts.show(truncate=False)


# Replace sentinel values with null
df_filtered = (
    df_filtered.withColumn("TMP", F.expr("CASE WHEN TMP = 999.9 THEN NULL ELSE TMP END"))
      .withColumn("DEW", F.expr("CASE WHEN DEW = 999.9 THEN NULL ELSE DEW END"))
      .withColumn("WND_Dir", F.expr("CASE WHEN WND_Dir = 999 THEN NULL ELSE WND_Dir END"))
      .withColumn("WND_Speed", F.expr("CASE WHEN WND_Speed = 999.9 THEN NULL ELSE WND_Speed END"))
      .withColumn("VIS", F.expr("CASE WHEN VIS = 999999 THEN NULL ELSE VIS END"))
      .withColumn("CIG", F.expr("CASE WHEN CIG = 99999 THEN NULL ELSE CIG END"))
      .withColumn("SLP", F.expr("CASE WHEN SLP = 9999.9 THEN NULL ELSE SLP END"))
      .withColumn("LATITUDE", F.expr("CASE WHEN abs(LATITUDE) = 999.999 THEN NULL ELSE LATITUDE END"))
      .withColumn("LONGITUDE", F.expr("CASE WHEN abs(LONGITUDE) = 999.999 THEN NULL ELSE LONGITUDE END"))
)

# Drop rows where all major weather features are null
df_filtered = df_filtered.filter(
    ~(F.col("TMP").isNull() |
      F.col("DEW").isNull() &
      F.col("WND_Dir").isNull() &
      F.col("WND_Speed").isNull() &
      F.col("VIS").isNull() &
      F.col("CIG").isNull() &
      F.col("SLP").isNull()
     )
)

# Inspect the remaining row count
print("Remaining rows after cleaning:", df_filtered.count())


# Inspect null counts after filtering
abnormal_counts = df_filtered.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df_filtered.columns])
abnormal_counts.show()

# Summarize null percentages for each column
total_count = df_filtered.count()
abnormal_counts = df_filtered.select([((F.sum(F.col(c).isNull().cast("int")) / total_count) * 100).alias(c) for c in df_filtered.columns])
abnormal_counts.show()


# ---------
# Step 4: Timestamp
# ---------
df_filtered = df_filtered.withColumn("DATE_TS", F.to_timestamp("DATE", "yyyy-MM-dd'T'HH:mm:ss")) 
print("Step 4: DATE_TS column added")

df_filtered.write.mode("overwrite").option("header", True).parquet(os.path.join(output_bucket, "filter1"))
