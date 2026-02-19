# src/data_prep/clean_interactions.py

import pandas as pd
from src.utils.text_cleaning import normalize_text

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the review text for better analysis.
    """
    if "review_clean" in df.columns and df["review_clean"].notna().any():
        print("Reviews already cleaned. Skipping.")
        return df

    cleaned_reviews = []

    for review in df['review']:
        if pd.isna(review):
            cleaned_reviews.append("")
        else:
            normalized_review = normalize_text(review)
            cleaned_reviews.append(normalized_review)

    df = df.copy()  
    df["review_clean"] = cleaned_reviews
    
    return df

def dtype_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types for interaction DataFrame.
    """
    df = df.copy()
    df['user_id'] = df['user_id'].astype('string')
    df['recipe_id'] = df['recipe_id'].astype('string')
    df['rating'] = df['rating'].astype('float32')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop invalid rows if necessary
    df = df.dropna(subset=["user_id", "recipe_id", "rating"])
    
    df = df[df['rating']!=0]  # Remove rows with rating 0 which is a null placeholder
    
    return df

def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the interactions DataFrame by normalizing reviews and correcting data types.
    """
    df = clean_reviews(df)
    df = dtype_corrections(df)
    return df