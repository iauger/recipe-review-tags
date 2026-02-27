from __future__ import annotations

import datetime
from datetime import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import platform
import re
import logging

logger = logging.getLogger(__name__)

def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()

def _is_windows_path(p: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:\\", p)) or p.startswith("\\\\")

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "requirements.txt").exists(): # crude heuristic for repo root
            return p
    # fallback: current behavior
    return here.parents[3]


def resolve_path(path: str | None, default: str) -> str:
    root = repo_root()

    if not path:
        path = default

    path = path.strip()

    if path.startswith("gs://"):
        return path
    
    if _is_wsl() and _is_windows_path(path):
        logger.warning("Detected Windows-style path in WSL environment: %s. Attempting to resolve to WSL path.", path)
        path = default  # fallback to default path in WSL if Windows path is detected

    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str((root / p).resolve())


@dataclass(frozen=True)
class Settings:
    env: str

    raw_recipes_path: str
    raw_interactions_path: str

    features_run_id: str
    
    raw_dir: str
    processed_dir: str
    features_dir: str
    features_run_dir: str
    features_pipeline_model_dir: str
    models_dir: str
    
    bronze_dir: str
    silver_dir: str
    gold_dir: str

    bronze_recipes_path: str
    bronze_interactions_path: str
    silver_recipes_path: str
    silver_interactions_path: str
    gold_recipe_path: str
    gold_reviews_path: str
    gold_ve_path: str
    gold_cf_path: str
    
    labeled_gold_reviews_path: str
    
    features_dataset_path: str
    features_splits_path: str
    features_manifest_path: str
    features_metrics_path: str
    features_tag_centroids_path: str

    spark_master: str
    spark_app_name: str
    spark_driver_memory: str

    pyspark_python: str
    pyspark_driver_python: str

    java_home: str | None

    zero_shot_model_id: str
    zero_shot_batch_size: int
    zero_shot_max_length: int
    zero_shot_label_threshold: float


def validate_settings(s: Settings) -> None:
    require_raw = os.getenv("REQUIRE_RAW_INPUTS", "0").strip() == "1"

    if s.env == "local" and require_raw:
        for p in [s.raw_recipes_path, s.raw_interactions_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required file: {p}")

    # still ensure directories exist
    for d in [s.raw_dir, s.processed_dir, s.models_dir, s.bronze_dir, s.silver_dir, s.gold_dir, s.features_dir, s.features_run_dir, s.features_pipeline_model_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if not (0.0 <= s.zero_shot_label_threshold <= 1.0):
        raise ValueError("ZERO_SHOT_LABEL_THRESHOLD must be between 0 and 1.")


def load_settings(*, prefer_latest_run: bool = True) -> Settings:
    load_dotenv(override=False)

    env = os.getenv("ENV", "local").strip().lower()

    raw_recipes_path = resolve_path(os.getenv("RAW_RECIPES_PATH"), "./data/raw/RAW_recipes.csv")
    raw_interactions_path = resolve_path(os.getenv("RAW_INTERACTIONS_PATH"), "./data/raw/RAW_interactions.csv")

    raw_dir = resolve_path(os.getenv("RAW_DIR"), "./data/raw")
    processed_dir = resolve_path(os.getenv("PROCESSED_DIR"), "./data/processed")
    features_dir = resolve_path(os.getenv("FEATURES_DIR"), "./data/processed/features")
    env_run = os.getenv("FEATURES_RUN_ID")

    if env_run:
        features_run_id = env_run
    else:
        latest_file = str(Path(features_dir) / "LATEST_RUN")

        if prefer_latest_run and Path(latest_file).exists():
            features_run_id = Path(latest_file).read_text().strip()
        else:
            features_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    features_run_dir = str(Path(features_dir) / "runs" / features_run_id)
    features_pipeline_model_dir = str(Path(features_run_dir) / "pipeline_models")

    models_dir = resolve_path(os.getenv("MODELS_DIR"), "./data/models")
    
    bronze_dir = resolve_path(os.getenv("BRONZE_DIR"), str(Path(processed_dir) / "bronze"))
    silver_dir = resolve_path(os.getenv("SILVER_DIR"), str(Path(processed_dir) / "silver"))
    gold_dir   = resolve_path(os.getenv("GOLD_DIR"),   str(Path(processed_dir) / "gold"))

    bronze_recipes_path = str(Path(bronze_dir) / "recipes_raw.parquet")
    bronze_interactions_path = str(Path(bronze_dir) / "interactions_raw.parquet")

    silver_recipes_path = str(Path(silver_dir) / "recipes_clean.parquet")
    silver_interactions_path = str(Path(silver_dir) / "interactions_clean.parquet")

    gold_recipe_path = str(Path(gold_dir) / "modeling_recipe.parquet")
    gold_reviews_path = str(Path(gold_dir) / "modeling_reviews.parquet")
    gold_ve_path = str(Path(gold_dir) / "modeling_ve.parquet")
    gold_cf_path = str(Path(gold_dir) / "modeling_cf.parquet")
    
    labeled_gold_reviews_path = str(Path(processed_dir) / "labeling" / "zero_shot" / "labeled_gold_reviews.parquet")
    
    features_dataset_path = str(Path(features_run_dir) / "dataset.parquet")
    features_splits_path = str(Path(features_run_dir) / "splits.parquet")
    features_manifest_path = str(Path(features_run_dir) / "manifest.json")
    features_metrics_path = str(Path(features_run_dir) / "metrics.json")
    features_tag_centroids_path = str(Path(features_run_dir) / "tag_centroids.parquet")

    spark_master = os.getenv("SPARK_MASTER", "local[*]").strip()
    spark_app_name = os.getenv("SPARK_APP_NAME", "recipe-review-tags").strip()
    spark_driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "6g").strip()

    pyspark_python = os.getenv("PYSPARK_PYTHON", "python").strip()
    pyspark_driver_python = os.getenv("PYSPARK_DRIVER_PYTHON", "python").strip()

    java_home = os.getenv("JAVA_HOME")
    if java_home:
        java_home = java_home.strip().rstrip("\\/")
    else:
        java_home = None

    zero_shot_model_id = os.getenv("ZERO_SHOT_MODEL_ID", "facebook/bart-large-mnli").strip()
    zero_shot_batch_size = int(os.getenv("ZERO_SHOT_BATCH_SIZE", "16").strip())
    zero_shot_max_length = int(os.getenv("ZERO_SHOT_MAX_LENGTH", "256").strip())
    zero_shot_label_threshold = float(os.getenv("ZERO_SHOT_LABEL_THRESHOLD", "0.35").strip())

    s = Settings(
        env=env,
        features_run_id=features_run_id,
        raw_recipes_path=raw_recipes_path,
        raw_interactions_path=raw_interactions_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        features_dir=features_dir,
        features_run_dir=features_run_dir,
        features_pipeline_model_dir=features_pipeline_model_dir,
        models_dir=models_dir,
        bronze_dir=bronze_dir,
        silver_dir=silver_dir,
        gold_dir=gold_dir,
        bronze_recipes_path=bronze_recipes_path,
        bronze_interactions_path=bronze_interactions_path,
        silver_recipes_path=silver_recipes_path,
        silver_interactions_path=silver_interactions_path,
        gold_recipe_path=gold_recipe_path,
        gold_reviews_path=gold_reviews_path,    
        gold_ve_path=gold_ve_path,
        gold_cf_path=gold_cf_path,
        labeled_gold_reviews_path=labeled_gold_reviews_path,  
        features_dataset_path=features_dataset_path,
        features_splits_path=features_splits_path,
        features_manifest_path=features_manifest_path,
        features_metrics_path=features_metrics_path,
        features_tag_centroids_path=features_tag_centroids_path,
        spark_master=spark_master,
        spark_app_name=spark_app_name,
        spark_driver_memory=spark_driver_memory,
        pyspark_python=pyspark_python,
        pyspark_driver_python=pyspark_driver_python,
        java_home=java_home,
        zero_shot_model_id=zero_shot_model_id,
        zero_shot_batch_size=zero_shot_batch_size,
        zero_shot_max_length=zero_shot_max_length,
        zero_shot_label_threshold=zero_shot_label_threshold,
    )

    validate_settings(s)
    return s
