from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


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

    p = Path(path)

    if p.is_absolute():
        return str(p)

    return str((root / p).resolve())


@dataclass(frozen=True)
class Settings:
    env: str

    raw_recipes_path: str
    raw_interactions_path: str

    raw_dir: str
    processed_dir: str
    models_dir: str

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
    if s.env == "local":
        for p in [s.raw_recipes_path, s.raw_interactions_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing required file: {p}")

    for d in [s.raw_dir, s.processed_dir, s.models_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if not (0.0 <= s.zero_shot_label_threshold <= 1.0):
        raise ValueError("ZERO_SHOT_LABEL_THRESHOLD must be between 0 and 1.")


def load_settings() -> Settings:
    load_dotenv(override=False)

    env = os.getenv("ENV", "local").strip().lower()

    raw_recipes_path = resolve_path(os.getenv("RAW_RECIPES_PATH"), "./data/raw/RAW_recipes.csv")
    raw_interactions_path = resolve_path(os.getenv("RAW_INTERACTIONS_PATH"), "./data/raw/RAW_interactions.csv")

    raw_dir = resolve_path(os.getenv("RAW_DIR"), "./data/raw")
    processed_dir = resolve_path(os.getenv("PROCESSED_DIR"), "./data/processed")
    models_dir = resolve_path(os.getenv("MODELS_DIR"), "./data/models")

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
        raw_recipes_path=raw_recipes_path,
        raw_interactions_path=raw_interactions_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        models_dir=models_dir,
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
