# scripts/features.py
from pyspark.sql.types import NumericType

LABEL = "TMP"        # 或者你的目标字段名
TIMESTAMP_COL = "DATE_TS"


def infer_numeric_features(df):
    """
    根据 DataFrame 的 schema 自动筛选数值型字段，排除标签列和时间戳列。
    返回值按列名排序，便于调试和复现。
    """
    excluded = {LABEL, TIMESTAMP_COL}
    numeric_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NumericType) and field.name not in excluded
    ]
    return sorted(numeric_cols)
