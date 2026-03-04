# src/spark/labeling/postprocess.py
from __future__ import annotations

import logging
from typing import Dict, List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

from src.spark.features.similarity import _native_cosine_sim
from src.spark.features.prototypes import PrototypeSpec

logger = logging.getLogger(__name__)

def calculate_diagonal_thresholds(
    labeled_df: DataFrame,
    centroids_df: DataFrame,
    *,
    spec: PrototypeSpec,
    label_prefix: str = "y_"
) -> Dict[str, float]:
    """
    Finding Thresholds from datasets by calculating the average similarity of reviews in a category to their own centroid.
    """
    # Prepare labeled data: convert Vector to Array for native sim
    train_labeled = (
        labeled_df.filter(F.col(spec.split_col) == F.lit(spec.train_split_value))
        .withColumn("_r_arr", vector_to_array(F.col(spec.features_col)))
    )

    # Extract Tag IDs and Centroid Arrays
    c_small = centroids_df.select(
        F.col(spec.out_tag_col).alias("_tag_id"),
        F.col(spec.out_centroid_col).alias("_c_arr")
    )

    # Join Reviews to their specific Centroids
    label_cols = [c for c in labeled_df.columns if c.startswith(label_prefix)]
    stack_expr = ", ".join([f"'{c[len(label_prefix):]}', {c}" for c in label_cols])
    
    unpivoted = train_labeled.select(
        "_r_arr",
        F.expr(f"stack({len(label_cols)}, {stack_expr}) as (_tag_id, _is_pos)")
    ).where(F.col("_is_pos") == 1)

    # Calculate Similarity and Aggregate
    thresholds_df = (
        unpivoted.join(F.broadcast(c_small), on="_tag_id")
        .withColumn("_sim", _native_cosine_sim("_r_arr", "_c_arr"))
        .groupBy("_tag_id")
        .agg(F.avg("_sim").alias("threshold"))
    )

    # Collect as a Dictionary
    rows = thresholds_df.collect()
    threshold_map = {r["_tag_id"]: float(r["threshold"]) for r in rows}
    
    logger.info("Generated automated thresholds for %d tags", len(threshold_map))
    return threshold_map