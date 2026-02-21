from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.spark.ingestion.pandas.io_pandas import (
    download_kaggle_dataset,
    read_raw_data,
    write_processed_parquet,
)

from src.spark.ingestion.pandas.transforms.recipes_pandas import clean_recipes
from src.spark.ingestion.pandas.transforms.interactions_pandas import clean_interactions
from src.spark.ingestion.pandas.transforms.merge_pandas import prepare_modeling_data

logger = logging.getLogger(__name__)


from src.config import Settings, load_settings

@dataclass(frozen=True)
class PipelineConfig:
    settings: Settings
    download_if_missing: bool = True
    kaggle_dataset: str = "shuyangli94/food-com-recipes-and-user-interactions"
    force_download: bool = False
    force_write: bool = False


def run_pipeline(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = cfg.settings
    
    logger.info(
        "Resolved paths | raw_dir=%s processed_dir=%s raw_recipes=%s raw_interactions=%s",
        s.raw_dir, s.processed_dir, s.raw_recipes_path, s.raw_interactions_path
    )

    # Download if missing
    recipes_path = Path(s.raw_recipes_path)
    interactions_path = Path(s.raw_interactions_path)

    if cfg.download_if_missing and (not recipes_path.exists() or not interactions_path.exists()):
        download_kaggle_dataset(s, dataset=cfg.kaggle_dataset, force=cfg.force_download)

    logger.info("Reading raw data from %s and %s", recipes_path, interactions_path)
    recipes_raw, interactions_raw = read_raw_data(s)

    logger.info("cleaning recipes")
    recipes_clean = clean_recipes(recipes_raw)

    logger.info("cleaning interactions")
    interactions_clean = clean_interactions(interactions_raw)

    logger.info("Merging and Validating modeling datasets")
    df_ve, df_cf = prepare_modeling_data(recipes_clean, interactions_clean)

    # Persist
    logger.info("Writing processed data to parquet in %s", s.processed_dir)
    write_processed_parquet(df_ve, s, "modeling_vectorization.parquet", force=cfg.force_write)
    write_processed_parquet(df_cf, s, "modeling_collab_filter.parquet", force=cfg.force_write)

    logger.info("Pipeline complete. df_ve=%s df_cf=%s", df_ve.shape, df_cf.shape)
    return df_ve, df_cf


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    
if __name__ == "__main__":
    configure_logging()
    s = load_settings()
    cfg = PipelineConfig(settings=s)
    run_pipeline(cfg)