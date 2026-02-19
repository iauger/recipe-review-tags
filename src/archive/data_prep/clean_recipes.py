# src/data_prep/clean_recipes.py


import pandas as pd
import ast
from src.utils.text_cleaning import normalize_text

def dtype_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types for recipes DataFrame.
    """
    df = df.copy()
    df['id'] = df['id'].astype('string')
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').astype('float32')
    df['contributor_id'] = df['contributor_id'].astype('string')
    df['submitted'] = pd.to_datetime(df['submitted'], errors='coerce')
    df['n_steps'] = df['n_steps'].astype('float32')
    df['n_ingredients'] = df['n_ingredients'].astype('float32')
    
    return df

def clean_ingredients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the ingredients list into a clean, joined string suitable for TF-IDF vectorization.
    Preserves multi-word ingredients by replacing spaces with underscores.
    """

    if "ingredients_clean" in df.columns and df["ingredients_clean"].notna().any():
        print("Ingredients already cleaned. Skipping.")
        return df

    
    cleaned_lists = []

    for raw_list in df['ingredients']:
        # Parse stringified Python list
        ing_list = ast.literal_eval(raw_list)

        # Normalize each ingredient and convert spaces to underscores
        cleaned_ing_list = [
            normalize_text(ing).replace(" ", "_")
            for ing in ing_list
        ]

        # Join into one TF-IDF-friendly string
        cleaned_lists.append(" ".join(cleaned_ing_list))

    df = df.copy()  
    df["ingredients_clean"] = cleaned_lists
    
    return df

def clean_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the list of step strings into a single cleaned string for TF-IDF vectorization.
    """
    if "steps_clean" in df.columns and df["steps_clean"].notna().any():
        print("Steps already cleaned. Skipping.")
        return df

    
    cleaned_steps = []

    for raw_steps in df['steps']:
        step_list = ast.literal_eval(raw_steps)
        joined_steps = " ".join(step_list)
        normalized_steps = normalize_text(joined_steps)
        cleaned_steps.append(normalized_steps)

    df = df.copy() 
    df["steps_clean"] = cleaned_steps
    
    return df

def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the tags by normalizing text and replacing spaces with underscores.
    """
    if "tags_clean" in df.columns and df["tags_clean"].notna().any():
        print("Tags already cleaned. Skipping.")
        return df

    cleaned_tags = []

    for raw_tags in df['tags']:
        tag_list = ast.literal_eval(raw_tags)
        cleaned_tag_list = [
            normalize_text(tag)
            for tag in tag_list
        ]
        cleaned_tags.append(" ".join(cleaned_tag_list))

    df = df.copy()  
    df["tags_clean"] = cleaned_tags
    
    return df

def extract_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the 7-number nutrition list into separate numeric columns.
    """

    def parse_nutrition(nutrition_list):
        lst = ast.literal_eval(nutrition_list)
        return pd.Series({
            "calories": lst[0],
            "fat": lst[1],
            "sugar": lst[2],
            "sodium": lst[3],
            "protein": lst[4],
            "saturated_fat": lst[5],
            "carbs": lst[6]
        })
    
    nutrition_cols = ["calories", "fat", "sugar", "sodium", "protein", "saturated_fat", "carbs"]

    # Skip only if ALL columns exist
    if all(col in df.columns for col in nutrition_cols):
        print("Nutrition already extracted. Skipping.")
        return df

    nutrition_df = df["nutrition"].apply(parse_nutrition)

    df = df.copy()
    df = pd.concat([df, nutrition_df], axis=1)

    return df

def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the description text.
    """
    if "description_clean" in df.columns and df["description_clean"].notna().any():
        print("Descriptions already cleaned. Skipping.")
        return df

    cleaned_desc = []

    for desc in df["description"]:
        # Handle missing entries
        if pd.isna(desc):
            cleaned_desc.append("")
            continue

        normalized = normalize_text(desc)
        cleaned_desc.append(normalized)

    df = df.copy()
    df["description_clean"] = cleaned_desc
    return df

def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning functions to the recipes DataFrame.
    """
    df = dtype_corrections(df)
    df = clean_ingredients(df)
    df = clean_steps(df)
    df = clean_tags(df)
    df = extract_nutrition(df)
    df = clean_description(df)
    return df