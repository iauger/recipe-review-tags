from __future__ import annotations

import logging
import pandas as pd
from src.utils.text_cleaning import normalize_text

logger = logging.getLogger(__name__)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if "review_clean" in df.columns and df["review_clean"].notna().any():
        logger.info("review_clean already present; skipping review normalization.")
        return df

    df = df.copy()
    df["review_clean"] = [
        "" if pd.isna(r) else normalize_text(r)
        for r in df["review"]
    ]
    return df


def dtype_corrections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype("string")
    df["recipe_id"] = df["recipe_id"].astype("string")
    df["rating"] = df["rating"].astype("float32")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["user_id", "recipe_id", "rating"])
    df = df[df["rating"] != 0]  # rating 0 = placeholder null in this dataset
    return df


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_reviews(df)
    df = dtype_corrections(df)
    return df