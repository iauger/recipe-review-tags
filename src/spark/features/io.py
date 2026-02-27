# src/features/io.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import Settings  

logger = logging.getLogger(__name__)


# -------------------------
# Directory + path helpers
# -------------------------

def ensure_dirs(s: Settings) -> None:
    """
    Ensure feature run directories exist.
    Settings.validate_settings already mkdirs, but this is safe to call from feature jobs too.
    """
    for d in [s.features_dir, s.features_run_dir, s.features_pipeline_model_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _maybe_repartition(df: DataFrame, n: Optional[int]) -> DataFrame:
    if n is None or n <= 0:
        return df
    return df.repartition(n)


# -------------------------
# Reads
# -------------------------

def read_labeled_reviews(spark: SparkSession, s: Settings) -> DataFrame:
    """
    Read the labeled gold reviews parquet used for training baseline models.
    """
    path = s.labeled_gold_reviews_path
    logger.info("Reading labeled reviews from %s", path)
    df = spark.read.parquet(path)

    required = ["review_key", "recipe_id", "review_clean", "zs_labels"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Labeled reviews missing required columns: {missing}")

    df = df.filter(F.col("review_key").isNotNull()).filter(F.col("recipe_id").isNotNull())

    return df


def read_feature_dataset(spark: SparkSession, s: Settings) -> DataFrame:
    logger.info("Reading feature dataset from %s", s.features_dataset_path)
    return spark.read.parquet(s.features_dataset_path)


def read_splits(spark: SparkSession, s: Settings) -> DataFrame:
    logger.info("Reading splits from %s", s.features_splits_path)
    return spark.read.parquet(s.features_splits_path)


# Writes

def write_parquet(
    df: DataFrame,
    path: str,
    *,
    mode: str = "overwrite",
    partition_cols: Optional[Iterable[str]] = None,
    repartition_n: Optional[int] = None,
) -> None:
    """
    Central parquet writer with optional repartitioning + partitioning.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    out = _maybe_repartition(df, repartition_n)

    writer = out.write.mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    logger.info("Writing parquet to %s (mode=%s, partitions=%s, repartition_n=%s)",
                str(p), mode, list(partition_cols) if partition_cols else None, repartition_n)
    writer.parquet(str(p))


def write_manifest(s: Settings, payload: dict[str, Any]) -> None:
    """
    Write run manifest JSON into s.features_manifest_path.
    Manifest should include config snapshot + key run metadata.
    """
    path = Path(s.features_manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # include settings snapshot so runs are self-describing
    base = {
        "features_run_id": s.features_run_id,
        "features_run_dir": s.features_run_dir,
        "inputs": {
            "labeled_gold_reviews_path": s.labeled_gold_reviews_path,
        },
        "outputs": {
            "features_dataset_path": s.features_dataset_path,
            "features_splits_path": s.features_splits_path,
            "features_pipeline_model_dir": s.features_pipeline_model_dir,
            "features_manifest_path": s.features_manifest_path,
            "features_metrics_path": s.features_metrics_path,
        },
        "settings_snapshot": asdict(s),
    }

    merged = {**base, **payload}

    logger.info("Writing manifest to %s", str(path))
    path.write_text(json.dumps(merged, indent=2, sort_keys=True))


def write_metrics(s: Settings, metrics: dict[str, Any]) -> None:
    """
    Write metrics JSON into s.features_metrics_path.
    """
    path = Path(s.features_metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing metrics to %s", str(path))
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


# Convenience utilities

def summarize_splits(df: DataFrame, split_col: str = "split") -> dict[str, int]:
    """
    Return split counts as a Python dict. Assumes df has a split column.
    """
    rows = df.groupBy(split_col).count().collect()
    return {r[split_col]: int(r["count"]) for r in rows}


def label_prevalence(df: DataFrame, label_cols: list[str]) -> dict[str, float]:
    """
    Compute prevalence (mean) for binary label columns on the provided DataFrame.
    Returns {col: prevalence}.
    """
    agg_exprs = [F.avg(F.col(c).cast("double")).alias(c) for c in label_cols]
    row = df.agg(*agg_exprs).collect()[0].asDict()
    return {k: float(v) if v is not None else 0.0 for k, v in row.items()}