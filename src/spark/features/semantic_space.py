# src/spark/features/semantic_space.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union

from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

logger = logging.getLogger(__name__)

# --- SPECS ---
@dataclass(frozen=True)
class PrototypeSpec:
    split_col: str = "split"
    train_split_value: str = "train"
    features_col: str = "features"
    label_prefix: str = "y_"
    out_tag_col: str = "tag"
    out_centroid_col: str = "centroid"
    out_count_col: str = "pos_count"

# --- MATH HELPERS ---
def _native_cosine_sim(col_a: Union[str, Column], col_b: Union[str, Column]) -> Column:
    """Calculates cosine similarity natively using arrays."""
    c1 = F.col(col_a) if isinstance(col_a, str) else col_a
    c2 = F.col(col_b) if isinstance(col_b, str) else col_b

    def square_sum(col):
        return F.aggregate(col, F.lit(0.0), lambda acc, x: acc + (x * x))

    dot_product = F.aggregate(
        F.arrays_zip(c1.alias("a"), c2.alias("b")),
        F.lit(0.0),
        lambda acc, x: acc + (x["a"] * x["b"])
    )
    
    norm_a, norm_b = F.sqrt(square_sum(c1)), F.sqrt(square_sum(c2))
    return F.when((norm_a == 0.0) | (norm_b == 0.0), 0.0).otherwise(dot_product / (norm_a * norm_b))

# --- CENTROIDS & CALIBRATION ---
def build_tag_centroids(df: DataFrame, *, spec: PrototypeSpec, labels: Optional[Iterable[str]] = None) -> DataFrame:
    """Calculates the mean vector (centroid) for each tag based on the Silver standard labels."""
    y_cols = [f"{spec.label_prefix}{t}" for t in labels] if labels else [c for c in df.columns if c.startswith(spec.label_prefix)]

    train_arr = df.filter(F.col(spec.split_col) == F.lit(spec.train_split_value)).withColumn("_vec_arr", vector_to_array(F.col(spec.features_col)))
    
    first_row = train_arr.select("_vec_arr").where(F.col("_vec_arr").isNotNull()).first()
    if not first_row: raise ValueError("No non-null feature vectors found.")
    dim = len(first_row["_vec_arr"])

    stack_expr = ", ".join([f"'{y[len(spec.label_prefix):]}', {y}" for y in y_cols])
    unpivoted = train_arr.select("_vec_arr", F.expr(f"stack({len(y_cols)}, {stack_expr}) as ({spec.out_tag_col}, _is_pos)")).where(F.col("_is_pos") == 1)

    return unpivoted.groupBy(spec.out_tag_col).agg(
        F.count("*").alias(spec.out_count_col),
        F.array([F.avg(F.col("_vec_arr")[i]) for i in range(dim)]).alias(spec.out_centroid_col)
    )

def calculate_diagonal_thresholds(labeled_df: DataFrame, centroids_df: DataFrame, *, spec: PrototypeSpec) -> Dict[str, float]:
    """Extracts diagonal thresholding map natively."""
    train_labeled = labeled_df.filter(F.col(spec.split_col) == F.lit(spec.train_split_value)).withColumn("_r_arr", vector_to_array(F.col(spec.features_col)))
    c_small = centroids_df.select(F.col(spec.out_tag_col).alias("_tag_id"), F.col(spec.out_centroid_col).alias("_c_arr"))

    label_cols = [c for c in labeled_df.columns if c.startswith(spec.label_prefix)]
    stack_expr = ", ".join([f"'{c[len(spec.label_prefix):]}', {c}" for c in label_cols])
    
    unpivoted = train_labeled.select("_r_arr", F.expr(f"stack({len(label_cols)}, {stack_expr}) as (_tag_id, _is_pos)")).where(F.col("_is_pos") == 1)

    thresholds_df = (
        unpivoted.join(F.broadcast(c_small), on="_tag_id")
        .withColumn("_sim", _native_cosine_sim("_r_arr", "_c_arr"))
        .groupBy("_tag_id").agg(F.avg("_sim").alias("threshold"))
    )

    threshold_map = {r["_tag_id"]: float(r["threshold"]) for r in thresholds_df.collect()}
    logger.info("Generated automated thresholds for %d tags", len(threshold_map))
    return threshold_map