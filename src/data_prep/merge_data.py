# src/data_prep/merge_recipes_interactions.py

import pandas as pd

def merge_datasets(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned recipes and interactions into a single modeling DataFrame using interactions as the main table and employing inner join to remove recipes without interactions.
    """

    # Perform inner join on recipe_id
    merged = interactions_df.merge(
        recipes_df,
        how="inner",
        left_on="recipe_id",
        right_on="id"
    )

    # Create binary target
    merged["liked"] = (merged["rating"] >= 4).astype("int8")

    return merged

def validate_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing essential features.
    """
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

    # Drop rows missing any essential feature
    df_clean = df.dropna(subset=required_columns)

    return df_clean

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns to match the specified order.
    """
    ordered_columns = [
    # Identifiers
    "recipe_id",
    "name",

    # Targets
    "rating",
    "liked",

    # Interaction metadata
    "date",
    "review_clean",

    # Cleaned recipe text features
    "ingredients_clean",
    "steps_clean",
    "tags_clean",
    "description_clean",

    # Numeric recipe metadata
    "minutes",
    "n_steps",
    "n_ingredients",

    # Nutrition features
    "calories",
    "fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbs",
]
    existing_cols = [col for col in ordered_columns if col in df.columns]
    return df[existing_cols]

# Note: clone_data()  creates two deep copies of the DataFrame for different modeling purposes
# I made this choice when I was entertaining the idea of having separate dataframes for vectorization and collaborative filtering.
# I did not pursue collaborative filtering in the end due to scope, but I am keeping this function in case I want to explore it later.
def clone_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two dataframes that are deep copies of the input DataFrame, one for vectorization and one for collaborative filtering.
    """
    df_ve = df.copy(deep=True)
    df_cf = df.copy(deep=True)
    
    # Remove columns we no longer need
    drop_ve_cols = [
        "id",               # duplicate of recipe_id
        "user_id",          # drop user_id to prevent data leakage
        "contributor_id",   # drop contributor_id to prevent data leakage 
        "submitted",
        "description",
        "steps",
        "ingredients",
        "tags",
        "nutrition",
        "review"
    ]
    
    keep_cf_cols = [
        "user_id",
        "recipe_id",
        "rating"
    ]
    
    for col in drop_ve_cols:
        if col in df_ve.columns:
            df_ve = df_ve.drop(columns=col)
    
    for col in df_cf.columns:
        if col not in keep_cf_cols:
            df_cf = df_cf.drop(columns=col)
    
    return df_ve, df_cf

def prepare_modeling_data(recipes_df: pd.DataFrame, interactions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the final modeling DataFrame by merging cleaned recipes and interactions, and validating the merged dataset.
    """
    merged_df = merge_datasets(recipes_df, interactions_df)
    validated_df = validate_merged_dataset(merged_df)
    reordered_df = reorder_columns(validated_df)
    df_ve, df_cf = clone_data(reordered_df)
    return df_ve, df_cf

