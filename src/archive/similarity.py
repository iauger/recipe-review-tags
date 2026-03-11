# src/spark/features/similarity.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.ml.functions import vector_to_array

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SimilaritySpec:
    review_id_col: str = "review_key"
    split_col: str = "split"
    review_vec_col: str = "features"
    centroid_tag_col: str = "tag"
    centroid_vec_col: str = "centroid"
    out_score_col: str = "sim"


def _native_cosine_sim(col_a: Union[str, Column], col_b: Union[str, Column]) -> Column:
    """
    Calculates cosine similarity using native Spark SQL functions with explicit naming.
    Matches the logic: dot(a, b) / (norm(a) * norm(b))
    """
    # 1. Resolve names/columns to fix Pylance/AnalysisException resolution issues
    c1 = F.col(col_a) if isinstance(col_a, str) else col_a
    c2 = F.col(col_b) if isinstance(col_b, str) else col_b

    def square_sum(col):
        return F.aggregate(col, F.lit(0.0), lambda acc, x: acc + (x * x))

    # 2. Name zipped fields explicitly to fix [INVALID_EXTRACT_FIELD_TYPE]
    # This prevents Spark from defaulting to numeric indices "0" and "1"
    
    dot_product = F.aggregate(
        F.arrays_zip(c1.alias("a"), c2.alias("b")),
        F.lit(0.0),
        lambda acc, x: acc + (x["a"] * x["b"])
    )
    
    norm_a = F.sqrt(square_sum(c1))
    norm_b = F.sqrt(square_sum(c2))
    
    return F.when((norm_a == 0.0) | (norm_b == 0.0), 0.0).otherwise(dot_product / (norm_a * norm_b))

def score_review_tag_similarity_long(
    reviews_df: DataFrame,
    centroids_df: DataFrame,
    *,
    spec: SimilaritySpec,
    filter_split: Optional[str] = None,
    broadcast_centroids: bool = True,
) -> DataFrame:
    r = reviews_df
    c = centroids_df

    if filter_split is not None:
        r = r.filter(F.col(spec.split_col) == F.lit(filter_split))

    # Convert review vectors to arrays (Native SQL works on arrays, not Vectors)
    r = r.withColumn("_r_arr", vector_to_array(F.col(spec.review_vec_col)))
    
    # centroids_df is already arrays from the previous step
    c_small = c.select(
        F.col(spec.centroid_tag_col).alias("_tag"), 
        F.col(spec.centroid_vec_col).alias("_c_arr")
    )

    if broadcast_centroids:
        c_small = F.broadcast(c_small)

    # Cross join and calculate similarity natively
    scored = (
        r.crossJoin(c_small)
        .withColumn(
            spec.out_score_col, 
            _native_cosine_sim("_r_arr", "_c_arr")
        )
        .select(
            spec.review_id_col,
            spec.split_col,
            F.col("_tag").alias("tag"),
            spec.out_score_col
        )
    )

    return scored

def pivot_similarity_wide(
    sim_long_df: DataFrame,
    *,
    tags: list[str],
    review_id_col: str = "review_key",
    split_col: str = "split",
    tag_col: str = "tag",
    score_col: str = "sim",
    prefix: str = "sim_",
) -> DataFrame:
    # This part is already quite efficient! 
    # Providing the list of tags to .pivot() is the best way to avoid an extra scan.
    return (
        sim_long_df
        .groupBy(review_id_col, split_col)
        .pivot(tag_col, sorted(list(tags)))
        .agg(F.first(score_col))
        # Mass-rename columns using select
        .select(
            review_id_col,
            split_col,
            *[F.col(t).alias(f"{prefix}{t}") for t in sorted(list(tags))]
        )
    )