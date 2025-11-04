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

spark = SparkSession.builder.appName("WeatherProcessing").getOrCreate()

print("Step 0: Reading files from", input_path)
df = spark.read.option("header", True).csv(input_path)
print("Step 0: Total rows read:", df.count())


# ---------
# Step 1: Filter stations by row counts 
# ---------
station_record_counts = df.groupBy("STATION").count().toPandas()
local_path = "/tmp/station_record_hist.png"
local_path1 = "/tmp/station_record_hist_10000.png"

plt.figure(figsize=(12,6))
plt.hist(station_record_counts['count'], bins=5000, color='skyblue', edgecolor='black')
plt.yscale('log')
plt.xlabel("Number of rows per station")
plt.ylabel("Number of stations")
plt.title("Distribution of rows per NOAA Station")
plt.tight_layout()
plt.savefig(local_path)
plt.close()

plt.figure(figsize=(12,6))
plt.hist(station_record_counts['count'], bins=200, range=(0,10000), color='skyblue', edgecolor='black')
plt.xlabel("Number of rows per Station (0-10000)")
plt.ylabel("Number of stations")
plt.title("Distribution of rows per NOAA Station (0-10000 rows)")
plt.tight_layout()
plt.savefig(local_path1)
plt.close()

# 上传到 GCS
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("spark-result")
blob = bucket.blob("station_record_hist.png")
blob1 = bucket.blob("station_record_hist_10000.png")
blob.upload_from_filename(local_path)
blob1.upload_from_filename(local_path1)

thres = station_record_counts['count'].quantile(0.05)
keep_files = station_record_counts[station_record_counts['count'] > thres]['STATION'].tolist()
removed_csv_len = len(station_record_counts) - len(keep_files)
df_filtered = df.where(F.col("STATION").isin(keep_files))

print("Step 1: Filter stations with threshld {thres}")
log_text = (
    f"Number of rows: {station_record_counts['count'].sum()}\n"
    f"Number of rows after filtering: {df_filtered.count()}\n"
    f"Total number of CSV files: {len(station_record_counts)}\n"
    f"Number of removed CSV files: {removed_csv_len}\n"
)
print(log_text)

df_filtered.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "filter_station1"))


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
# Step 3: clean anomalies
# ---------
abnormal_counts = df_filtered.select([
    sum((F.col("TMP") == 9999/10).cast("int")).alias("TMP"),
    sum((F.col("DEW") == 9999/10).cast("int")).alias("DEW"),
    sum((F.col("WND_Dir") == 999).cast("int")).alias("WND_Dir"),
    sum((F.col("WND_Speed") == 9999/10).cast("int")).alias("WND_Speed"),
    sum((F.col("VIS") == 999999).cast("int")).alias("VIS"),
    sum((F.col("CIG") == 99999).cast("int")).alias("CIG"),
    sum((F.col("SLP") == 99999/10).cast("int")).alias("SLP"),
    sum((F.abs(F.col("LATITUDE")) == 999.999).cast("int")).alias("LATITUDE"),
    sum((F.abs(F.col("LONGITUDE")) == 999.999).cast("int")).alias("LONGITUDE"),
    sum((F.col("ELEVATION") == 9999.9).cast("int")).alias("ELEVATION")
])
print("Step 3: Before anomaly filter, anomaly summary: ")
abnormal_counts.show(truncate=False)


df_filtered = df_filtered.withColumn("TMP", F.when(F.col("TMP") == 9999/10, None).otherwise(F.col("TMP"))) \
       .withColumn("DEW", F.when(F.col("DEW") == 9999/10, None).otherwise(F.col("DEW"))) \
       .withColumn("WND_Dir", F.when(F.col("WND_Dir") == 999, None).otherwise(F.col("WND_Dir"))) \
       .withColumn("WND_Speed", F.when(F.col("WND_Speed") == 9999/10, None).otherwise(F.col("WND_Speed"))) \
       .withColumn("VIS", F.when(F.col("VIS") == 999999, None).otherwise(F.col("VIS"))) \
       .withColumn("CIG", F.when(F.col("CIG") == 99999, None).otherwise(F.col("CIG"))) \
       .withColumn("SLP", F.when(F.col("SLP") == 99999/10, None).otherwise(F.col("SLP"))) \
       .withColumn("LATITUDE", F.when(F.abs(F.col("LATITUDE")) == 999.999, None).otherwise(F.col("LATITUDE"))) \
       .withColumn("LONGITUDE", F.when(F.abs(F.col("LONGITUDE")) == 999.999, None).otherwise(F.col("LONGITUDE"))) \
       .withColumn("ELEVATION", F.when(F.col("ELEVATION") == 9999.9, None).otherwise(F.col("ELEVATION")))

# drop null rows   
df_filtered = df_filtered.dropna(subset=["TMP"])
print("Step 3: After cleaning TMP, remaining rows:", df_filtered.count())
cols_to_check = ["DEW","WND_Dir","WND_Speed","VIS","CIG","SLP"]
df_filtered = df_filtered.dropna(how="all", subset=cols_to_check)
print("Step 3: After cleaning attributes with all null, remaining rows:", df_filtered.count())


abnormal_counts = df_filtered.select([
    sum(F.col("TMP").isNull().cast("int")).alias("TMP_null"),
    sum(F.col("DEW").isNull().cast("int")).alias("DEW_null"),
    sum(F.col("WND_Dir").isNull().cast("int")).alias("WND_Dir_null"),
    sum(F.col("WND_Speed").isNull().cast("int")).alias("WND_Speed_null"),
    sum(F.col("VIS").isNull().cast("int")).alias("VIS_null"),
    sum(F.col("CIG").isNull().cast("int")).alias("CIG_null"),
    sum(F.col("SLP").isNull().cast("int")).alias("SLP_null"),
    sum(F.col("LATITUDE").isNull().cast("int")).alias("LAT_null"),
    sum(F.col("LONGITUDE").isNull().cast("int")).alias("LON_null"),
    sum(F.col("ELEVATION").isNull().cast("int")).alias("ELEV_null")
])
print("Step 3: After anomaly filter, anomaly summary: ")
abnormal_counts.show(truncate=False)

df_filtered.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "filter_anomaly3"))


# ---------
# Step 4: Timestamp
# ---------
df_filtered = df_filtered.withColumn("DATE_TS", F.to_timestamp("DATE", "yyyy-MM-dd'T'HH:mm:ss"))
print("Step 4: DATE_TS column added")


# ---------
# Step 5: Numeric Col & linear interpolation
# ---------
cols_to_interp = ["WND_Speed", "VIS", "CIG", "SLP", "DEW"]
partition_cols = ["STATION"]
time_col = "DATE_TS"

for c in cols_to_interp:
    # 定义窗口：同一组内按时间排序
    base_w = Window.partitionBy(*partition_cols).orderBy(time_col)

    # 各点最近的“前一个”非空取值及时间
    prev_w = base_w.rowsBetween(Window.unboundedPreceding, 0)
    prev_val = F.last(c, ignorenulls=True).over(prev_w)
    prev_ts = F.last(F.when(F.col(c).isNotNull(), F.col(time_col)), ignorenulls=True).over(prev_w)

    # 各点最近的“后一个”非空取值及时间
    next_w = base_w.rowsBetween(0, Window.unboundedFollowing)
    next_val = F.first(c, ignorenulls=True).over(next_w)
    next_ts = F.first(F.when(F.col(c).isNotNull(), F.col(time_col)), ignorenulls=True).over(next_w)

    # 时间差（秒）
    total_dt = next_ts.cast("long") - prev_ts.cast("long")
    curr_dt = F.col(time_col).cast("long") - prev_ts.cast("long")


    interp = (
        F.when(
            F.col(c).isNull() &
            prev_val.isNotNull() &
            next_val.isNotNull() &
            (total_dt != 0),
            prev_val + (next_val - prev_val) * curr_dt / total_dt
        )
        .when(F.col(c).isNull(), prev_val)  # 如果只有前向值可用，就做 forward fill
        .otherwise(F.col(c))
    )

    df_filtered = df_filtered.withColumn(f"{c}_interp", interp)

print("Step 5: Linear interpolation done")
df_filtered.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "filter_numeric5"))


# ---------
# Step 6: Periodic columns sin, cos
# ---------
df_filtered = df_filtered.withColumn("WND_sin", F.sin(F.radians(F.col("WND_Dir")))) \
                         .withColumn("WND_cos", F.cos(F.radians(F.col("WND_Dir")))) \
                         .drop("WND_Dir") \
                         .withColumn("year", F.year("DATE_TS")) \
                         .withColumn("month", F.month("DATE_TS")) \
                         .withColumn("day", F.dayofmonth("DATE_TS")) \
                         .withColumn("hour", F.hour("DATE_TS")) \
                         .withColumn("minute", F.minute("DATE_TS")) \
                         .withColumn("month_sin", F.sin(2 * F.pi() * F.col("month") / 12)) \
                         .withColumn("month_cos", F.cos(2 * F.pi() * F.col("month") / 12)) \
                         .withColumn("day_sin", F.sin(2 * F.pi() * F.col("day") / 31)) \
                         .withColumn("day_cos", F.cos(2 * F.pi() * F.col("day") / 31)) \
                         .withColumn("hour_sin", F.sin(2 * F.pi() * F.col("hour") / 24)) \
                         .withColumn("hour_cos", F.cos(2 * F.pi() * F.col("hour") / 24)) \
                         .withColumn("minute_sin", F.sin(2 * F.pi() * F.col("minute") / 60)) \
                         .withColumn("minute_cos", F.cos(2 * F.pi() * F.col("minute") / 60)) \
                         .drop("year", "month", "day", "hour", "minute") \
                         .withColumn("lat_sin", F.sin(F.radians(F.col("LATITUDE")))) \
                         .withColumn("lat_cos", F.cos(F.radians(F.col("LATITUDE")))) \
                         .withColumn("lon_sin", F.sin(F.radians(F.col("LONGITUDE")))) \
                         .withColumn("lon_cos", F.cos(F.radians(F.col("LONGITUDE")))) \
                         .drop("LATITUDE", "LONGITUDE") \
                         .na.drop()
print("Step 6 done")
df_filtered.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "filtered_perodic6"))


# ---------
# Step 7: Periodic columns WIN_DIR Fourier Interp
# ---------
periodic_cols = ["WND_sin", "WND_cos"]

# 0) 预先缓存主数据，避免重复扫描；同时触发一次 action
df_filtered = (
    df_filtered
    .repartition("STATION")
    .persist(StorageLevel.MEMORY_AND_DISK)
)
df_filtered.count()

# 1) 找出缺失率很低的站点，直接跳过 UDF
null_stats = (
    df_filtered
    .select("STATION", *periodic_cols)
    .groupBy("STATION")
    .agg(
        *[
            F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_null")
            for c in periodic_cols
        ],
        F.count("*").alias("total_cnt")
    )
)

null_stats = null_stats.withColumn(
    "null_total",
    reduce(lambda a, b: a + b, [F.col(f"{c}_null") for c in periodic_cols])
).withColumn(
    "null_ratio",
    F.col("null_total") / F.col("total_cnt")
)

stations_to_interp = (
    null_stats
    .filter(F.col("null_ratio") >= F.lit(0.001))  # 0.1% 以上缺失才进入 UDF
    .select("STATION")
)

df_interp = df_filtered.join(stations_to_interp, on="STATION", how="inner")
df_skip = df_filtered.join(stations_to_interp, on="STATION", how="leftanti")

# 2) 定义自适应窗口 + 降 TOP_K 的傅立叶插值
schema = StructType([
    StructField("STATION", StringType(), False),
    StructField("DATE_TS", TimestampType(), False),
] + [StructField(c, DoubleType(), True) for c in periodic_cols])

def _fourier_reconstruct(values: np.ndarray, top_k: int) -> np.ndarray:
    mask = np.isnan(values)
    if not mask.any() or mask.all():
        return values
    mean_val = np.nanmean(values)
    centered = np.where(mask, 0.0, values - mean_val)

    fft_vals = np.fft.rfft(centered)
    keep = min(max(top_k, 1), len(fft_vals) - 1)
    filtered = np.zeros_like(fft_vals)
    filtered[: keep + 1] = fft_vals[: keep + 1]

    reconstructed = np.fft.irfft(filtered, n=len(centered)) + mean_val
    filled = values.copy()
    filled[mask] = reconstructed[mask]
    return filled

def fourier_fill(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("DATE_TS")
    if pdf.empty:
        return pdf[["STATION", "DATE_TS", *periodic_cols]]

    mask_any = pdf[periodic_cols].isnull().any(axis=1)
    if not mask_any.any():
        return pdf[["STATION", "DATE_TS", *periodic_cols]]

    dt = pdf["DATE_TS"].diff().median()
    if pd.isna(dt) or dt <= pd.Timedelta(0):
        dt = pd.Timedelta(minutes=30)
    samples_per_day = max(int(round(pd.Timedelta(days=1) / dt)), 1)

    groups = mask_any.ne(mask_any.shift()).cumsum()
    max_gap = mask_any.groupby(groups).sum().max()
    max_gap = 0 if pd.isna(max_gap) else max_gap
    gap_days = max_gap / samples_per_day

    if gap_days <= 1:
        window_days, top_k = 3, 3
    elif gap_days <= 3:
        window_days, top_k = 5, 4
    elif gap_days <= 7:
        window_days, top_k = 10, 5
    else:
        window_days, top_k = 21, 6

    missing_times = pdf.loc[mask_any, "DATE_TS"]
    buffer = pd.Timedelta(days=window_days / 2)
    start = missing_times.min() - buffer
    end = missing_times.max() + buffer

    recent_mask = (pdf["DATE_TS"] >= start) & (pdf["DATE_TS"] <= end)
    if recent_mask.sum() < 4:
        recent_mask = slice(None)

    recent = pdf.loc[recent_mask].copy()
    for c in periodic_cols:
        recent[c] = _fourier_reconstruct(
            recent[c].to_numpy(dtype=float),
            top_k=top_k
        )

    pdf.loc[recent_mask, periodic_cols] = recent[periodic_cols].to_numpy()
    return pdf[["STATION", "DATE_TS", *periodic_cols]]

filled_periodic = (
    df_interp
    .select("STATION", "DATE_TS", *periodic_cols)
    .repartition("STATION")
    .groupby("STATION")
    .applyInPandas(fourier_fill, schema=schema)
)

df_interp = (
    df_interp.drop(*periodic_cols)
             .join(filled_periodic, on=["STATION", "DATE_TS"], how="left")
)

df_filtered = (
    df_interp.unionByName(df_skip, allowMissingColumns=True)
              .persist(StorageLevel.MEMORY_AND_DISK)
)

print("Step 7: After Fourier interp, num of rows:", df_filtered.count())  # 触发缓存，也验证最终行数
df_filtered.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "filtered_fourier7"))


# ---------
# Step 8: Train/test split
# ---------
df_filtered = df_filtered.orderBy("DATE_TS")
min_date, max_date = df_filtered.agg(F.min("DATE_TS"), F.max("DATE_TS")).first()
cutoff = min_date + (max_date - min_date) * 0.7
train_df = df_filtered.filter(F.col("DATE_TS") <= cutoff)
test_df  = df_filtered.filter(F.col("DATE_TS") > cutoff)
print("Step 8: Train/Test split done")
print("Train count:", train_df.count(), "Test count:", test_df.count())

train_df.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "train"))
test_df.write.mode("overwrite").option("header", True).csv(os.path.join(output_bucket, "test"))
print("All steps completed. CSVs saved to", output_bucket)

