from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, regexp_replace, abs as abs_, when, sum as sum_
import matplotlib.pyplot as plt
from google.cloud import storage


spark = SparkSession.builder.appName("WeatherPreprocess").getOrCreate()

# ------------------------
# 1. Loading
# ------------------------
input_path = "gs://weather-2024/csv/*.csv"
output_bucket = "spark-result"

print("Reading CSV files from:", input_path)
df = spark.read.option("header", True).csv(input_path)

# ------------------------
# 2. Filter stations
# ------------------------
station_record_counts = df.groupBy("STATION").count().toPandas()

plt.figure(figsize=(12,6))
plt.hist(station_record_counts['count'], bins=5000, color='skyblue', edgecolor='black')
plt.xlabel("Number of rows per station")
plt.ylabel("Number of stations")
plt.title("Distribution of rows per NOAA Station")
plt.yscale('log')
plt.tight_layout()
plt.savefig("/tmp/station_record_hist.png")
plt.close()

plt.figure(figsize=(12,6))
plt.hist(station_record_counts['count'], bins=200, range=(0,10000), color='skyblue', edgecolor='black')
plt.xlabel("Number of rows per Station (0-10000)")
plt.ylabel("Number of stations")
plt.title("Distribution of rows per NOAA Station (0-10000 rows)")
plt.tight_layout()
plt.savefig("/tmp/station_record_hist_0_10000.png")
plt.close()

thres = station_record_counts['count'].quantile(0.05)
keep_files = station_record_counts[station_record_counts['count'] > thres]['STATION'].tolist()
removed_csv_len = len(station_record_counts) - len(keep_files)
print(f"Threshold (5% quantile): {thres}")

df_filtered = df.filter(col("STATION").isin(keep_files))
print("After filtering:", df_filtered.count())

log_text = (
    f"Number of rows: {station_record_counts['count'].sum()}\n"
    f"Number of rows after filtering: {df_filtered.count()}\n"
    f"Total stations: {len(station_record_counts)}\n"
    f"Removed stations: {removed_csv_len}\n"
)
print(log_text)

# ------------------------
# save csv, log and pics
# ------------------------

output_path = f"gs://{output_bucket}/filtered_data/cleaned_csv"
print("Writing cleaned data to:", output_path)
df_filtered.write.mode("overwrite").option("header", True).csv(output_path)

bucket = storage.Client().bucket(output_bucket)

for local_path, gcs_path in [
    ("/tmp/filtered_count.txt", "filtered_data/filtered_count.txt"),
    ("/tmp/station_record_hist.png", "filtered_data/station_record_hist.png"),
    ("/tmp/station_record_hist_0_10000.png", "filtered_data/station_record_hist_0_10000.png"),
]:
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print("Uploaded:", gcs_path)

# ------------------------
# 3. Select columns
# ------------------------
cols = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION",
        "TMP", "DEW", "WND", "VIS", "CIG", "SLP"]

df_filtered = df_filtered.select(cols)

df_filtered = (
    df_filtered.withColumn("TMP", (regexp_replace(split(col("TMP"), ",")[0], "[+]", "").cast("double") / 10))
    .withColumn("DEW", (regexp_replace(split(col("DEW"), ",")[0], "[+]", "").cast("double") / 10))
    .withColumn("WND_Dir", split(col("WND"), ",")[0].cast("double"))
    .withColumn("WND_Speed", split(col("WND"), ",")[3].cast("double") / 10)
    .withColumn("VIS", split(col("VIS"), ",")[0].cast("double"))
    .withColumn("CIG", split(col("CIG"), ",")[0].cast("double"))
    .withColumn("SLP", split(col("SLP"), ",")[0].cast("double") / 10)
    .drop("WND")
)

# ------------------------
# save csv
# ------------------------
output_path = f"gs://{output_bucket}/filtered_data/cleaned_csv"
print("Writing cleaned data to:", output_path)
df_filtered.write.mode("overwrite").option("header", True).csv(output_path)

# ------------------------
# 4. Anomaly
# ------------------------
abnormal_counts = df_filtered.select([
    sum((col("TMP") == 9999/10).cast("int")).alias("TMP"),
    sum((col("DEW") == 9999/10).cast("int")).alias("DEW"),
    sum((col("WND_Dir") == 999).cast("int")).alias("WND_Dir"),
    sum((col("WND_Speed") == 9999/10).cast("int")).alias("WND_Speed"),
    sum((col("VIS") == 999999).cast("int")).alias("VIS"),
    sum((col("CIG") == 99999).cast("int")).alias("CIG"),
    sum((col("SLP") == 99999/10).cast("int")).alias("SLP"),
    sum((abs_(col("LATITUDE")) == 999.999).cast("int")).alias("LATITUDE"),
    sum((abs_(col("LONGITUDE")) == 999.999).cast("int")).alias("LONGITUDE"),
    sum((col("ELEVATION") == 9999.9).cast("int")).alias("ELEVATION")
])
print(f"abnormal_counts: {abnormal_counts}")

df_filtered = df_filtered.withColumn("TMP", when(col("TMP") == 999.9, None).otherwise(col("TMP"))) \
       .withColumn("DEW", when(col("DEW") == 999.9, None).otherwise(col("DEW"))) \
       .withColumn("WND_Dir", when(col("WND_Dir") == 999, None).otherwise(col("WND_Dir"))) \
       .withColumn("WND_Speed", when(col("WND_Speed") == 999.9, None).otherwise(col("WND_Speed"))) \
       .withColumn("VIS", when(col("VIS") == 999999, None).otherwise(col("VIS"))) \
       .withColumn("CIG", when(col("CIG") == 99999, None).otherwise(col("CIG"))) \
       .withColumn("SLP", when(col("SLP") == 9999.9, None).otherwise(col("SLP"))) \
       .withColumn("LATITUDE", when(abs_(col("LATITUDE")) == 999.999, None).otherwise(col("LATITUDE"))) \
       .withColumn("LONGITUDE", when(abs_(col("LONGITUDE")) == 999.999, None).otherwise(col("LONGITUDE"))) \
       .withColumn("ELEVATION", when(col("ELEVATION") == 9999.9, None).otherwise(col("ELEVATION")))

df_filtered = df_filtered.dropna(subset=["TMP"])
print("TMP cleaned data count:", df_filtered.count())

cols_to_check = ["DEW","WND_Dir","WND_Speed","VIS","CIG","SLP"]
df_filtered = df_filtered.dropna(how="all", subset=cols_to_check)

print("Final cleaned data count:", df_filtered.count())

# ------------------------
# save csv
# ------------------------
output_path = f"gs://{output_bucket}/filtered_data/cleaned_csv"
print("Writing cleaned data to:", output_path)
df_filtered.write.mode("overwrite").option("header", True).csv(output_path)

print("✅ Preprocessing completed successfully.")
spark.stop()
