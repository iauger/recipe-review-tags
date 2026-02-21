from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, cast, Optional

from pyspark.sql import SparkSession

from src.config import Settings, load_settings

logger = logging.getLogger(__name__)


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _resolve_python(s: Settings) -> tuple[str, str]:
    """
    Resolve python executables for driver + worker.

    Priority:
      1) Settings values if set to a real executable path
      2) sys.executable (notebook/venv python)
      3) fallback to "python"
    """
    def _valid_exe(p: str) -> bool:
        if not p:
            return False
        # allow "python" / "python3" names as valid fallbacks
        if Path(p).name.lower().startswith("python") and not Path(p).suffix:
            return True
        return Path(p).exists()

    driver_py = s.pyspark_driver_python.strip() if s.pyspark_driver_python else ""
    worker_py = s.pyspark_python.strip() if s.pyspark_python else ""

    if not _valid_exe(driver_py):
        driver_py = sys.executable or "python"
    if not _valid_exe(worker_py):
        worker_py = driver_py  # keep them consistent

    return driver_py, worker_py


def _log_runtime_diagnostics(s: Settings, driver_py: str, worker_py: str) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return

    logger.info("=== Spark runtime diagnostics ===")
    logger.info("OS: %s", platform.platform())
    logger.info("ENV=%s spark_master=%s app_name=%s", s.env, s.spark_master, s.spark_app_name)
    logger.info("python(sys.executable)=%s", sys.executable)
    logger.info("pyspark_driver_python=%s", driver_py)
    logger.info("pyspark_worker_python=%s", worker_py)

    logger.info("JAVA_HOME=%s", os.environ.get("JAVA_HOME"))
    logger.info("HADOOP_HOME=%s", os.environ.get("HADOOP_HOME"))
    logger.info("winutils=%s", shutil.which("winutils"))

    # Helpful for the exact issue you hit earlier
    if _is_windows():
        if not os.environ.get("HADOOP_HOME"):
            logger.warning("HADOOP_HOME not set; parquet writes may fail on Windows.")
        if not shutil.which("winutils"):
            logger.warning("winutils.exe not found on PATH; parquet writes may fail on Windows.")

    logger.info("===============================")


def get_spark(app_name: str | None = None) -> SparkSession:
    """
    Create a SparkSession using settings from .env/config.py, hardened for Windows local dev.
    """
    s = load_settings()
        
    # Avoid version mismatch if an external Spark install is present
    os.environ.pop("SPARK_HOME", None)
    os.environ.pop("PYSPARK_SUBMIT_ARGS", None)

    # Ensure JAVA_HOME is available (prefer settings)
    if s.java_home:
        os.environ["JAVA_HOME"] = s.java_home

    # Force Spark to use the same python interpreter as the notebook/venv
    driver_py = sys.executable
    worker_py = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = driver_py
    os.environ["PYSPARK_PYTHON"] = worker_py

    builder = cast(Any, SparkSession.builder)

    b = (
        builder
        .master(s.spark_master)
        .appName(app_name or s.spark_app_name)
        .config("spark.driver.memory", s.spark_driver_memory)

        # Force python at Spark conf level (prevents worker mismatch)
        .config("spark.pyspark.driver.python", driver_py)
        .config("spark.pyspark.python", worker_py)

        # Turn Arrow off until stable
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")

        # Reduce noisy UI
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")

        # Windows networking stability
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")

        # Less fragile output commit behavior
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
    )

    if _is_windows():
        b = (
            b.config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
            # Force RawLocalFileSystem to avoid NativeIO Windows access checks
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
            # Reduce checksum interactions that trigger listStatus / canRead paths
            .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        )

    return b.getOrCreate()


def spark_doctor(spark: SparkSession, *, out_dir: Optional[str] = None) -> None:
    """
    Quick self-test you can run in a notebook to isolate Windows issues:
      - verifies python config
      - tests JVM-only parquet write (no Python worker)
      - tests Python-created DF parquet write (uses Python workers)
    """
    from pyspark.sql import Row

    print("spark.version:", spark.version)
    print("hadoop.version:", spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion())  # type: ignore
    print("spark.pyspark.driver.python:", spark.conf.get("spark.pyspark.driver.python", "NOT SET"))
    print("spark.pyspark.python:", spark.conf.get("spark.pyspark.python", "NOT SET"))
    print("HADOOP_HOME:", os.environ.get("HADOOP_HOME"))
    print("winutils:", shutil.which("winutils"))

    base = Path(out_dir) if out_dir else Path.cwd() / "spark_doctor_tmp"
    base.mkdir(parents=True, exist_ok=True)

    # JVM-only write (no Python worker)
    jvm_path = str(base / "_jvm_smoke.parquet")
    spark.range(2).write.mode("overwrite").parquet(jvm_path)
    print("JVM parquet write OK ->", jvm_path)

    # Python worker write
    py_path = str(base / "_py_smoke.parquet")
    spark.createDataFrame([Row(x=1), Row(x=2)]).write.mode("overwrite").parquet(py_path)
    print("Python parquet write OK ->", py_path)