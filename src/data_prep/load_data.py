# src/data_prep/load_data.py

import pandas as pd
import os
from pathlib import Path
import kaggle

# Set up directory paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
recipes_path = RAW_DATA_DIR / "RAW_recipes.csv"
interactions_path = RAW_DATA_DIR / "RAW_interactions.csv"

def build_data_paths():
    """Ensure that all necessary data directories exist."""
    print(f"Working in {BASE_DIR}")
    
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

def download_data(dest: Path):
    """Download data from kaggle using kaggle API."""
    
    # Prevent re-downloading if files already exist
    if recipes_path.exists() and interactions_path.exists():
        print("Data already exists. Skipping download.")
        return
    
    dataset = "shuyangli94/food-com-recipes-and-user-interactions"  
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=dest, unzip=True)

def load_data():
    """Load recipes and interactions data from CSV."""
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError("Raw data directory does not exist. Re-run build_data_paths()?")

    if not recipes_path.exists() or not interactions_path.exists():
        raise FileNotFoundError(f"{recipes_path} or {interactions_path} does not exist. Re-run download_data().")
    

    recipes = pd.read_csv(
        recipes_path, 
        dtype={
        "name": "string",
        "id": "string",
        "minutes": "float32",
        "contributor_id": "string",
        "submitted": "string",
        "tags": "string",
        "nutrition": "string",
        "n_steps": "float32",
        "steps": "string",
        "description": "string",
        "ingredients": "string",
        "n_ingredients": "float32",}
    )
    interactions = pd.read_csv(
        interactions_path, 
        dtype={
        "user_id": "string",
        "recipe_id": "string",
        "date": "string",
        "rating": "float32",
        "review": "string",}
    )
    
    print(f"Loaded recipes data with shape: {recipes.shape}")
    print(f"Loaded interactions data with shape: {interactions.shape}")
    
    return recipes, interactions

if __name__ == "__main__":
    build_data_paths()
    download_data(RAW_DATA_DIR)
    recipes, interactions = load_data()