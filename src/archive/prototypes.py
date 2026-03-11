# src/spark/features/prototypes.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PrototypeSpec:
    split_col: str = "split"
    train_split_value: str = "train"
    features_col: str = "features"
    label_prefix: str = "y_"
    out_tag_col: str = "tag"
    out_centroid_col: str = "centroid"
    out_count_col: str = "pos_count"

def _label_cols(df: DataFrame, *, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def build_tag_centroids(
    df: DataFrame,
    *,
    spec: PrototypeSpec,
    labels: Optional[Iterable[str]] = None,
) -> DataFrame:
    if spec.features_col not in df.columns:
        raise ValueError(f"Missing features_col '{spec.features_col}' in input df")

    # 1. Resolve label columns
    if labels is None:
        y_cols = _label_cols(df, prefix=spec.label_prefix)
        if not y_cols:
            raise ValueError(f"No label columns found with prefix '{spec.label_prefix}'")
    else:
        y_cols = [f"{spec.label_prefix}{t}" for t in labels]

    # 2. Filter for training and convert Vector -> Array
    # This replaces your manual vector_struct_to_array function
    train_arr = (
        df.filter(F.col(spec.split_col) == F.lit(spec.train_split_value))
        .withColumn("_vec_arr", vector_to_array(F.col(spec.features_col)))
    )

    # 3. Determine embedding dimension
    first_row = train_arr.select("_vec_arr").where(F.col("_vec_arr").isNotNull()).first()
    if not first_row:
        raise ValueError("No non-null feature vectors found in training split.")
    dim = len(first_row["_vec_arr"])

    # 4. The "Stack" approach: Unpivot labels so we can group by tag
    # This avoids the Python 'for' loop and multiple Spark jobs
    stack_expr = ", ".join([f"'{y[len(spec.label_prefix):]}', {y}" for y in y_cols])
    unpivoted = train_arr.select(
        "_vec_arr",
        F.expr(f"stack({len(y_cols)}, {stack_expr}) as ({spec.out_tag_col}, _is_pos)")
    ).where(F.col("_is_pos") == 1)

    # 5. Aggregate: Calculate mean for every element in the array at once
    # We group by tag and compute the average for each index in the embedding
    centroids = (
        unpivoted.groupBy(spec.out_tag_col)
        .agg(
            F.count("*").alias(spec.out_count_col),
            F.array([F.avg(F.col("_vec_arr")[i]) for i in range(dim)]).alias(spec.out_centroid_col)
        )
    )

    return centroids