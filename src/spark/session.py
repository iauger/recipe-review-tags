# src/spark/session.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, cast, Optional

from pyspark.sql import SparkSession
from src.config import Settings, load_settings

logger = logging.getLogger(__name__)

# Environment Setup Helpers
def _resolve_python(s: Settings) -> tuple[str, str]:
    """Resolve python executables for driver + worker."""
    driver_py = (s.pyspark_driver_python or "").strip() or sys.executable or "python3"
    worker_py = (s.pyspark_python or "").strip() or driver_py
    return driver_py, worker_py

def _configure_java_home(s: Settings) -> None:
    """Ensure JAVA_HOME is valid for the current runtime."""
    if s.java_home:
        os.environ["JAVA_HOME"] = s.java_home
    elif not os.environ.get("JAVA_HOME"):
        candidate = "/usr/lib/jvm/java-17-openjdk-amd64"
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate

# Main Spark Session Helper
def get_spark(app_name: str | None = None, *, debug: bool = False) -> SparkSession:
    """Create a SparkSession using settings from .env/config.py."""
    s = load_settings()

    _configure_java_home(s)

    # WSL local IP binding
    if bool(os.environ.get("WSL_DISTRO_NAME")):
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        
    driver_py, worker_py = _resolve_python(s)

    os.environ["PYSPARK_DRIVER_PYTHON"] = driver_py
    os.environ["PYSPARK_PYTHON"] = worker_py

    if debug:
        logger.info("=== Spark Runtime ===")
        logger.info("master=%s app_name=%s", s.spark_master, s.spark_app_name)
        logger.info("driver_python=%s worker_python=%s", driver_py, worker_py)
        logger.info("JAVA_HOME=%s", os.environ.get("JAVA_HOME"))

    builder = cast(Any, SparkSession.builder)

    b = (
        builder
        .master(s.spark_master)
        .appName(app_name or s.spark_app_name)
        
        # Memory / Performance
        .config("spark.driver.memory", s.spark_driver_memory)
        .config("spark.driver.maxResultSize", getattr(s, 'spark_driver_max_result_size', '4g'))
        .config("spark.executor.memory", "4g")
        .config("spark.memory.fraction", "0.6")  
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # Python consistency
        .config("spark.pyspark.driver.python", driver_py)
        .config("spark.pyspark.python", worker_py)

        # UI & Networking
        .config("spark.ui.enabled", "true")
        .config("spark.ui.showConsoleProgress", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )

    return b.getOrCreate()

def spark_doctor(spark: SparkSession, *, out_dir: Optional[str] = None) -> None:
    """Quick WSL/Linux smoke test for JVM and Python interactions."""
    from pyspark.sql import Row

    print("spark.version:", spark.version)
    print("spark.pyspark.driver.python:", spark.conf.get("spark.pyspark.driver.python", "NOT SET"))
    print("JAVA_HOME:", os.environ.get("JAVA_HOME"))

    base = Path(out_dir) if out_dir else Path.cwd() / "spark_doctor_tmp"
    base.mkdir(parents=True, exist_ok=True)

    jvm_path = str(base / "_jvm_smoke.parquet")
    spark.range(2).write.mode("overwrite").parquet(jvm_path)
    print("JVM parquet write OK ->", jvm_path)

    py_path = str(base / "_py_smoke.parquet")
    spark.createDataFrame([Row(x=1), Row(x=2)]).write.mode("overwrite").parquet(py_path)
    print("Python parquet write OK ->", py_path)