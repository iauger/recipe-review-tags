from __future__ import annotations

import ast
import logging
import pandas as pd
from src.utils.text_cleaning import normalize_text

logger = logging.getLogger(__name__)


def dtype_corrections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id"] = df["id"].astype("string")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").astype("float32")
    df["contributor_id"] = df["contributor_id"].astype("string")
    df["submitted"] = pd.to_datetime(df["submitted"], errors="coerce")
    df["n_steps"] = df["n_steps"].astype("float32")
    df["n_ingredients"] = df["n_ingredients"].astype("float32")
    return df


def clean_ingredients(df: pd.DataFrame) -> pd.DataFrame:
    if "ingredients_clean" in df.columns and df["ingredients_clean"].notna().any():
        logger.info("ingredients_clean already present; skipping.")
        return df

    df = df.copy()
    cleaned_lists = []

    for raw_list in df["ingredients"]:
        ing_list = ast.literal_eval(raw_list)
        cleaned_ing_list = [normalize_text(ing).replace(" ", "_") for ing in ing_list]
        cleaned_lists.append(" ".join(cleaned_ing_list))

    df["ingredients_clean"] = cleaned_lists
    return df


def clean_steps(df: pd.DataFrame) -> pd.DataFrame:
    if "steps_clean" in df.columns and df["steps_clean"].notna().any():
        logger.info("steps_clean already present; skipping.")
        return df

    df = df.copy()
    cleaned_steps = []

    for raw_steps in df["steps"]:
        step_list = ast.literal_eval(raw_steps)
        joined_steps = " ".join(step_list)
        cleaned_steps.append(normalize_text(joined_steps))

    df["steps_clean"] = cleaned_steps
    return df


def clean_tags(df: pd.DataFrame) -> pd.DataFrame:
    if "tags_clean" in df.columns and df["tags_clean"].notna().any():
        logger.info("tags_clean already present; skipping.")
        return df

    df = df.copy()
    cleaned_tags = []

    for raw_tags in df["tags"]:
        tag_list = ast.literal_eval(raw_tags)
        cleaned_tag_list = [normalize_text(tag) for tag in tag_list]
        cleaned_tags.append(" ".join(cleaned_tag_list))

    df["tags_clean"] = cleaned_tags
    return df


def extract_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    nutrition_cols = ["calories", "fat", "sugar", "sodium", "protein", "saturated_fat", "carbs"]
    if all(col in df.columns for col in nutrition_cols):
        logger.info("Nutrition columns already present; skipping.")
        return df

    def parse_nutrition(nutrition_list: str) -> pd.Series:
        lst = ast.literal_eval(nutrition_list)
        return pd.Series(
            {
                "calories": lst[0],
                "fat": lst[1],
                "sugar": lst[2],
                "sodium": lst[3],
                "protein": lst[4],
                "saturated_fat": lst[5],
                "carbs": lst[6],
            }
        )

    nutrition_df = df["nutrition"].apply(parse_nutrition)
    df = df.copy()
    df = pd.concat([df, nutrition_df], axis=1)
    return df


def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    if "description_clean" in df.columns and df["description_clean"].notna().any():
        logger.info("description_clean already present; skipping.")
        return df

    df = df.copy()
    df["description_clean"] = [
        "" if pd.isna(desc) else normalize_text(desc)
        for desc in df["description"]
    ]
    return df


def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    df = dtype_corrections(df)
    df = clean_ingredients(df)
    df = clean_steps(df)
    df = clean_tags(df)
    df = extract_nutrition(df)
    df = clean_description(df)
    return df