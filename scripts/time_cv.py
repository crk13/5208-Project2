from pyspark.sql import functions as F
from pyspark.sql import types as T

def chronological_split(df, timestamp_col, train_ratio=0.7):
    df = df.withColumn("ts_long", F.col(timestamp_col).cast("long"))
    cutoff = df.approxQuantile("ts_long", [train_ratio], 0.0)[0]
    train = df.filter(F.col("ts_long") <= cutoff)
    test  = df.filter(F.col("ts_long") > cutoff)
    return train.drop("ts_long"), test.drop("ts_long")

def prefix_folds(train_df, timestamp_col, num_folds=4):
    """
    Create prefix-style folds by ordering on the timestamp and assigning a
    global index via zipWithIndex, which avoids an unpartitioned window
    operation.
    """
    sorted_df = train_df.orderBy(timestamp_col).cache()
    indexed_rdd = sorted_df.rdd.zipWithIndex().map(
        lambda x: tuple(x[0]) + (int(x[1]),)
    )
    schema = sorted_df.schema.add("_row_idx", T.LongType())
    indexed_df = sorted_df.sparkSession.createDataFrame(indexed_rdd, schema).cache()

    total = indexed_df.count()
    if total == 0:
        indexed_df.unpersist()
        sorted_df.unpersist()
        return []

    fold_size = max(total // num_folds, 1)
    folds = []
    for k in range(1, num_folds + 1):
        val_start = fold_size * (k - 1)
        val_end = fold_size * k if k < num_folds else total

        val_df = indexed_df.filter(
            (F.col("_row_idx") >= val_start) & (F.col("_row_idx") < val_end)
        ).drop("_row_idx")
        train_slice = indexed_df.filter(
            F.col("_row_idx") < val_start
        ).drop("_row_idx")

        if val_df.rdd.isEmpty():
            continue
        folds.append((train_slice, val_df))

    indexed_df.unpersist()
    sorted_df.unpersist()
    return folds
