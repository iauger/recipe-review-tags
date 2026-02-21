# src/spark/ingestion/spark/pipeline_spark.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from src.config import Settings, load_settings
from src.spark.session import get_spark  # assumes you have this
from src.spark.ingestion.spark.io_spark import (
    download_kaggle_dataset,
    read_raw_data,
    write_parquet,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    settings: Settings
    download_if_missing: bool = True
    kaggle_dataset: str = "shuyangli94/food-com-recipes-and-user-interactions"
    force_download: bool = False
    mode: str = "overwrite"  # parquet write mode


def run_pipeline(cfg: PipelineConfig, spark: SparkSession | None = None) -> tuple[int, int]:
    """
    Bronze-stage Spark pipeline:
      - optionally download raw CSVs (Kaggle)
      - read raw CSVs with explicit schemas
      - write bronze parquet outputs
      - return row counts for quick validation
    """
    s = cfg.settings

    # Create SparkSession if not provided
    if spark is None:
        spark = get_spark()

    # Optional download if missing
    recipes_path = Path(s.raw_recipes_path)
    interactions_path = Path(s.raw_interactions_path)

    if cfg.download_if_missing and (not recipes_path.exists() or not interactions_path.exists()):
        logger.info("Raw CSV(s) missing. Downloading from Kaggle into %s", s.raw_dir)
        download_kaggle_dataset(s, dataset=cfg.kaggle_dataset, force=cfg.force_download)

    # Read raw data
    logger.info("Reading raw CSVs with strict schemas...")
    recipes_raw_df, interactions_raw_df = read_raw_data(spark, s)

    # Basic sanity logs (no actions yet beyond schema)
    logger.info("recipes_raw_df schema:\n%s", recipes_raw_df.schema)
    logger.info("interactions_raw_df schema:\n%s", interactions_raw_df.schema)

    # Force actions (counts) to validate load success
    recipes_n = recipes_raw_df.count()
    interactions_n = interactions_raw_df.count()

    logger.info("Row counts: recipes=%s interactions=%s", recipes_n, interactions_n)

    # Write bronze parquet
    # Note: writing under processed/bronze/ keeps bronze separate from your final outputs
    write_parquet(recipes_raw_df, s, "bronze/recipes_raw.parquet", mode=cfg.mode)
    write_parquet(interactions_raw_df, s, "bronze/interactions_raw.parquet", mode=cfg.mode)

    logger.info("Bronze write complete: %s", s.processed_dir)
    return recipes_n, interactions_n


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