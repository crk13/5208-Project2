from pyspark.sql.types import NumericType

LABEL = "TMP"        # Target column
TIMESTAMP_COL = "DATE_TS"


def infer_numeric_features(df):
    """
    Return the sorted list of numeric feature columns, excluding the label and
    timestamp fields. Sorting keeps the output deterministic for debugging.
    """
    excluded = {LABEL, TIMESTAMP_COL}
    numeric_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NumericType) and field.name not in excluded
    ]
    return sorted(numeric_cols)
