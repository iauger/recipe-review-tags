# src/spark/features/dataset_prep.py
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def assign_recipe_splits(
    df: DataFrame,
    *,
    recipe_id_col: str = "recipe_id",
    split_col: str = "split",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> DataFrame:
    """Deterministically assign recipes to train/val/test splits."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1")

    recipes = df.select(F.col(recipe_id_col)).where(F.col(recipe_id_col).isNotNull()).distinct()

    MOD = 10_000_000
    u = (
        F.pmod(
            F.xxhash64(F.concat_ws("::", F.lit(str(seed)), F.col(recipe_id_col))),
            F.lit(MOD),
        ).cast("double") / F.lit(float(MOD))
    )

    recipes = recipes.withColumn("_u", u)
    train_cut = float(train_frac)
    val_cut = float(train_frac + val_frac)

    recipes = (
        recipes.withColumn(
            split_col,
            F.when(F.col("_u") < F.lit(train_cut), F.lit("train"))
             .when(F.col("_u") < F.lit(val_cut), F.lit("val"))
             .otherwise(F.lit("test"))
        ).drop("_u")
    )

    out = df.join(recipes, on=recipe_id_col, how="left")
    return out.withColumn(split_col, F.when(F.col(split_col).isNull(), F.lit("train")).otherwise(F.col(split_col)))


def add_binary_label_cols(
    df: DataFrame,
    labels: list[str],
    *,
    source_col: str = "zs_labels",
    prefix: str = "y_",
    output_dtype: str = "int",
) -> DataFrame:
    """Explode array of zero-shot labels into binary indicator columns."""
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
    return [c for c in df.columns if c.startswith(prefix)]