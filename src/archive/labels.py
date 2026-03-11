# src/spark/features/labels.py
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def add_binary_label_cols(
    df: DataFrame,
    labels: list[str],
    *,
    source_col: str = "zs_labels",
    prefix: str = "y_",
    output_dtype: str = "int",
) -> DataFrame:
    """Given a DataFrame with a column of list labels (e.g. from zero-shot labeling), add binary indicator columns for each label.  
    """
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found in DataFrame")
    
    out = df
    for label in labels:
        col_name = f"{prefix}{label}"
        out = out.withColumn(
            col_name,
            F.when(F.array_contains(F.col(source_col), label), F.lit(1)).otherwise(F.lit(0)).cast(output_dtype)
        )
    
    return out

def get_label_cols(df: DataFrame, prefix: str = "y_") -> list[str]:
    """Get a list of all columns in the DataFrame that start with the given prefix (e.g. "y_")."""
    return [c for c in df.columns if c.startswith(prefix)]