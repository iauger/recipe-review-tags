# src/spark/features/splits.py
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
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac, val_frac, and test_frac must sum to 1")

    recipes = (
        df.select(F.col(recipe_id_col))
        .where(F.col(recipe_id_col).isNotNull())
        .distinct()
    )

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
        )
        .drop("_u")
    )

    out = df.join(recipes, on=recipe_id_col, how="left")

    out = out.withColumn(
        split_col,
        F.when(F.col(split_col).isNull(), F.lit("train")).otherwise(F.col(split_col))
    )

    return out


def build_splits_table(
    df_with_splits: DataFrame,
    *,
    review_key_col: str = "review_key",
    recipe_id_col: str = "recipe_id",
    split_col: str = "split",
) -> DataFrame:
    return (
        df_with_splits
        .select(review_key_col, recipe_id_col, split_col)
        .where(F.col(review_key_col).isNotNull())
        .dropDuplicates([review_key_col])
    )