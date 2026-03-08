from __future__ import annotations

import logging
from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# Column contracts

REVIEW_REQUIRED: List[str] = [
    "user_id",
    "recipe_id",
    "date",
    "rating",
    "review_clean",
]

RECIPE_REQUIRED: List[str] = [
    "id",  # will be aliased to recipe_id in gold_recipes
    "name",
    "ingredients_clean",
    "steps_clean",
    "tags_clean",
    "description_clean",
    "minutes",
    "n_steps",
    "n_ingredients",
    "calories",
    "fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbs",
]


# Helpers

def _assert_has_columns(df: DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}. Columns: {df.columns}")


def _add_liked(df: DataFrame, rating_col: str = "rating") -> DataFrame:
    return df.withColumn("liked", (F.col(rating_col) >= F.lit(4.0)).cast("int"))


def add_review_key(
    df: DataFrame,
    user_col: str = "user_id",
    recipe_col: str = "recipe_id",
    date_col: str = "date",
    text_col: str = "review_clean",
    key_col: str = "review_key",
) -> DataFrame:
    """
    Deterministic key used to join zero-shot labels back to Spark rows later.
    """
    required = [user_col, recipe_col, date_col, text_col]
    _assert_has_columns(df, required, "add_review_key input")

    return df.withColumn(
        key_col,
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col(user_col).cast("string"), F.lit("")),
                F.coalesce(F.col(recipe_col).cast("string"), F.lit("")),
                F.coalesce(F.col(date_col).cast("string"), F.lit("")),
                F.coalesce(F.col(text_col).cast("string"), F.lit("")),
            ),
            256,
        ),
    )


# Gold Reviews

def build_gold_reviews(interactions_df: DataFrame) -> DataFrame:
    """
    Gold reviews = review-level document table used for labeling + ML.
    """
    _assert_has_columns(interactions_df, REVIEW_REQUIRED, "interactions_df")

    df = interactions_df.select(*REVIEW_REQUIRED)
    df = _add_liked(df, rating_col="rating")

    # Drop nulls and duplicates on the columns we truly need for labeling + joining labels back
    df = df.dropna(subset=["user_id", "recipe_id", "date", "rating", "review_clean"])
    df = df.dropDuplicates(["user_id", "recipe_id", "date", "rating", "review_clean"])

    # Add deterministic join key for labeling outputs
    df = add_review_key(df)

    # Stable column order
    ordered = [
        "review_key",
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "liked",
        "review_clean",
    ]
    return df.select(*ordered)


# Gold Recipes

def build_gold_recipes(recipes_df: DataFrame) -> DataFrame:
    """
    Gold recipes = recipe dimension table (one row per recipe).
    """
    _assert_has_columns(recipes_df, RECIPE_REQUIRED, "recipes_df")

    df = recipes_df.select(*RECIPE_REQUIRED).withColumnRenamed("id", "recipe_id")

    # Drop nulls on core fields required for enrichment
    df = df.dropna(subset=[
        "recipe_id",
        "name",
        "ingredients_clean",
        "steps_clean",
        "tags_clean",
        "description_clean",
        "minutes",
        "n_steps",
        "n_ingredients",
        "calories",
        "fat",
        "sugar",
        "sodium",
        "protein",
        "saturated_fat",
        "carbs",
    ])

    ordered = [
        "recipe_id",
        "name",
        "ingredients_clean",
        "steps_clean",
        "tags_clean",
        "description_clean",
        "minutes",
        "n_steps",
        "n_ingredients",
        "calories",
        "fat",
        "sugar",
        "sodium",
        "protein",
        "saturated_fat",
        "carbs",
    ]
    return df.select(*ordered)


# Build Gold VE

def build_gold_ve(gold_reviews: DataFrame, gold_recipes: DataFrame) -> DataFrame:
    """
    Optional wide table (reviews enriched with recipe fields).
    Only run when you truly need the join (can be expensive on local Spark).
    """
    _assert_has_columns(gold_reviews, ["recipe_id"], "gold_reviews")
    _assert_has_columns(gold_recipes, ["recipe_id"], "gold_recipes")

    merged = gold_reviews.join(gold_recipes, on="recipe_id", how="inner")

    # If you want a stable VE ordering:
    ordered = [
        "recipe_id",
        "name",
        "user_id",
        "date",
        "rating",
        "liked",
        "review_key",
        "review_clean",
        "ingredients_clean",
        "steps_clean",
        "tags_clean",
        "description_clean",
        "minutes",
        "n_steps",
        "n_ingredients",
        "calories",
        "fat",
        "sugar",
        "sodium",
        "protein",
        "saturated_fat",
        "carbs",
    ]
    existing = [c for c in ordered if c in merged.columns]
    return merged.select(*existing)