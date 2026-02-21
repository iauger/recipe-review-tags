# src/spark/ingestion/spark/io_spark.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession

from src.config import Settings
from src.spark.ingestion.spark.schemas_spark import RAW_RECIPES_SCHEMA, RAW_INTERACTIONS_SCHEMA

logger = logging.getLogger(__name__)


def download_kaggle_dataset(
    s: Settings,
    dataset: str = "shuyangli94/food-com-recipes-and-user-interactions",
    force: bool = False,
) -> None:
    """
    Download + unzip a Kaggle dataset into s.raw_dir.

    Note: keep kaggle import inside function so environments without kaggle can still import this module.
    """
    dest_dir = Path(s.raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(dest_dir.glob("RAW_*.csv")):
        logger.info("Raw CSVs already exist in %s; skipping Kaggle download.", dest_dir)
        return

    try:
        import kaggle  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Kaggle package not available. Install it with `pip install kaggle` and ensure credentials are set."
        ) from e

    logger.info("Authenticating Kaggle API...")
    kaggle.api.authenticate()

    logger.info("Downloading Kaggle dataset %s to %s (unzip=True)...", dataset, dest_dir)
    kaggle.api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)


def _validate_required_files(paths: Sequence[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n- " + "\n- ".join(missing))


def read_raw_data(
    spark: SparkSession,
    s: Settings,
) -> Tuple[DataFrame, DataFrame]:
    """
    Read raw CSVs into Spark DataFrames using strict schemas.

    Returns: (recipes_raw_df, interactions_raw_df)
    """
    recipes_path = Path(s.raw_recipes_path)
    interactions_path = Path(s.raw_interactions_path)

    _validate_required_files([recipes_path, interactions_path])

    logger.info("Reading RAW_recipes from %s", recipes_path)
    recipes_df = (
        spark.read
        .option("header", True)
        .option("escape", '"')   # defensive for quoted fields
        .schema(RAW_RECIPES_SCHEMA)
        .csv(str(recipes_path))
    )

    logger.info("Reading RAW_interactions from %s", interactions_path)
    interactions_df = (
        spark.read
        .option("header", True)
        .option("escape", '"')
        .schema(RAW_INTERACTIONS_SCHEMA)
        .csv(str(interactions_path))
    )

    return recipes_df, interactions_df


def write_parquet(
    df: DataFrame,
    s: Settings,
    rel_path: str,
    mode: str = "overwrite",
    partition_cols: Optional[list[str]] = None,
) -> Path:
    """
    Write a Spark DataFrame to parquet under s.processed_dir / rel_path.
    Returns the output path.
    """
    out_path = (Path(s.processed_dir) / rel_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = df.write.mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    logger.info("Writing parquet to %s (mode=%s)", out_path, mode)
    writer.parquet(str(out_path))
    return out_path