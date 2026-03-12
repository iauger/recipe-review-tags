# src/spark/ingestion/bronze_ingest.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType

from src.config import Settings, load_settings
from src.spark.session import get_spark

logger = logging.getLogger(__name__)

# data schemas
RAW_RECIPES_SCHEMA = StructType([
    StructField("name", StringType(), True),
    StructField("id", LongType(), True),
    StructField("minutes", LongType(), True),
    StructField("contributor_id", LongType(), True),
    StructField("submitted", StringType(), True),
    StructField("tags", StringType(), True),
    StructField("nutrition", StringType(), True),
    StructField("n_steps", LongType(), True),
    StructField("steps", StringType(), True),
    StructField("description", StringType(), True),
    StructField("ingredients", StringType(), True),
    StructField("n_ingredients", LongType(), True),
])

RAW_INTERACTIONS_SCHEMA = StructType([
    StructField("user_id", LongType(), True),
    StructField("recipe_id", LongType(), True),
    StructField("date", StringType(), True),
    StructField("rating", LongType(), True),
    StructField("review", StringType(), True),
])

# i/o helpers
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
        .option("quote", '"')    # ensure quotes are properly handled
        .option("multiLine", True)  # allow multi-line fields
        .option("mode", "PERMISSIVE")  # handle malformed lines gracefully
        .option("columnNameOfCorruptRecord", "_corrupt_record")  # capture corrupt records
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

# bronze pipeline
@dataclass(frozen=True)
class PipelineConfig:
    settings: Settings
    download_if_missing: bool = True
    kaggle_dataset: str = "shuyangli94/food-com-recipes-and-user-interactions"
    force_download: bool = False
    mode: str = "overwrite"

def run_bronze_pipeline(cfg: PipelineConfig, spark: SparkSession | None = None) -> tuple[int, int]:
    s = cfg.settings
    if spark is None:
        spark = get_spark()

    if cfg.download_if_missing:
        download_kaggle_dataset(s, dataset=cfg.kaggle_dataset, force=cfg.force_download)

    recipes_raw_df, interactions_raw_df = read_raw_data(spark, s)
    recipes_n, interactions_n = recipes_raw_df.count(), interactions_raw_df.count()
    
    write_parquet(recipes_raw_df, s, "bronze/recipes_raw.parquet", mode=cfg.mode)
    write_parquet(interactions_raw_df, s, "bronze/interactions_raw.parquet", mode=cfg.mode)
    
    logger.info("Bronze write complete. Rows: recipes=%s interactions=%s", recipes_n, interactions_n)
    return recipes_n, interactions_n