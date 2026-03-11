from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def merge_datasets(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
    merged = interactions_df.merge(
        recipes_df,
        how="inner",
        left_on="recipe_id",
        right_on="id",
    )
    merged["liked"] = (merged["rating"] >= 4).astype("int8")
    return merged


def validate_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
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
        "rating",
    ]
    return df.dropna(subset=required_columns)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "recipe_id",
        "name",
        "rating",
        "liked",
        "date",
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
    existing_cols = [c for c in ordered_columns if c in df.columns]
    return df[existing_cols]


def clone_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ve = df.copy(deep=True)
    df_cf = df.copy(deep=True)

    drop_ve_cols = [
        "id",
        "user_id",
        "contributor_id",
        "submitted",
        "description",
        "steps",
        "ingredients",
        "tags",
        "nutrition",
        "review",
    ]
    keep_cf_cols = ["user_id", "recipe_id", "rating"]

    for col in drop_ve_cols:
        if col in df_ve.columns:
            df_ve = df_ve.drop(columns=col)

    df_cf = df_cf[[c for c in keep_cf_cols if c in df_cf.columns]]
    return df_ve, df_cf


def prepare_modeling_data(
    recipes_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_df = merge_datasets(recipes_df, interactions_df)
    validated_df = validate_merged_dataset(merged_df)
    df_ve, df_cf = clone_data(validated_df)
    
    # Guardrail 
    required_cf = {"user_id", "recipe_id", "rating"}
    missing = required_cf - set(df_cf.columns)
    if missing:
        raise ValueError(f"CF dataset missing required columns: {sorted(missing)}")
    
    df_ve = reorder_columns(df_ve)
    return df_ve, df_cf