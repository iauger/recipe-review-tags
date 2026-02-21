from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple
import kaggle  # type: ignore
import pandas as pd

from src.config import Settings  # adjust import if needed

logger = logging.getLogger(__name__)


def download_kaggle_dataset(
    s: Settings,
    dataset: str = "shuyangli94/food-com-recipes-and-user-interactions",
    force: bool = False,
) -> None:
    """
    Download + unzip a Kaggle dataset into s.raw_dir.
    """
    dest_dir = Path(s.raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(dest_dir.glob("RAW_*.csv")):
        logger.info("Raw CSVs already exist in %s; skipping Kaggle download.", dest_dir)
        return

    logger.info("Authenticating Kaggle API...")
    kaggle.api.authenticate()

    logger.info("Downloading Kaggle dataset %s to %s (unzip=True)...", dataset, dest_dir)
    kaggle.api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)


def read_raw_data(s: Settings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recipes_path = Path(s.raw_recipes_path)
    interactions_path = Path(s.raw_interactions_path)

    for p in (recipes_path, interactions_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

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
            "n_ingredients": "float32",
        },
    )

    interactions = pd.read_csv(
        interactions_path,
        dtype={
            "user_id": "string",
            "recipe_id": "string",
            "date": "string",
            "rating": "float32",
            "review": "string",
        },
    )

    logger.info("Loaded recipes=%s interactions=%s", recipes.shape, interactions.shape)
    return recipes, interactions


def write_processed_parquet(df: pd.DataFrame, s: Settings, rel_path: str, force: bool = False) -> Path:
    """
    Write df to s.processed_dir / rel_path.
    Returns the written path.
    """
    out_path = (Path(s.processed_dir) / rel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        logger.info("Processed file exists (%s); skipping write.", out_path)
        return out_path

    logger.info("Writing %s rows to %s", len(df), out_path)

    try:
        df.to_parquet(out_path, index=False)
    except ImportError as e:
        raise ImportError(
            "Parquet support missing. Install one of:\n"
            "  pip install pyarrow\n"
            "  pip install fastparquet\n"
            f"Original error: {e}"
        ) from e

    return out_path