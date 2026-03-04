from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast, Optional

from pyspark.sql import SparkSession

from src.config import Settings, load_settings

logger = logging.getLogger(__name__)


# -------------------------
# OS / environment helpers
# -------------------------

def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _is_wsl() -> bool:
    """
    True when running inside WSL (WSL1/WSL2).
    """
    return bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()


def _is_windows_path(p: str) -> bool:
    """
    Detect paths like:
      C:\Program Files\Java\...
      \\server\share\...
    """
    return bool(re.match(r"^[A-Za-z]:\\", p)) or p.startswith("\\\\")


def _sanitize_env_for_pyspark() -> None:
    """
    Avoid accidental coupling to a Windows Spark install or submit args.
    Keep this conservative: remove vars that frequently cause mismatches.
    """
    os.environ.pop("SPARK_HOME", None)
    os.environ.pop("PYSPARK_SUBMIT_ARGS", None)
    os.environ.pop("HADOOP_HOME", None)  # only meaningful for native Windows Hadoop


def _resolve_python(s: Settings) -> tuple[str, str]:
    """
    Resolve python executables for driver + worker.

    Priority:
      1) Settings values if set to a real executable path or python/python3
      2) sys.executable (notebook/venv python)
      3) fallback to python3/python depending on OS
    """
    def _valid_exe(p: str) -> bool:
        if not p:
            return False
        p = p.strip()
        # allow "python" / "python3" names as valid
        if p in {"python", "python3"}:
            return True
        if _is_wsl() and _is_windows_path(p):
            return False
        return Path(p).exists()

    driver_py = (s.pyspark_driver_python or "").strip()
    worker_py = (s.pyspark_python or "").strip()

    if not _valid_exe(driver_py):
        driver_py = sys.executable or ("python" if _is_windows() else "python3")

    if not _valid_exe(worker_py):
        worker_py = driver_py  # keep consistent

    return driver_py, worker_py


def _configure_java_home(s: Settings) -> None:
    """
    Ensure JAVA_HOME is valid for the current runtime.

    Key rule:
      - In WSL/Linux: NEVER set JAVA_HOME to a Windows path (C:\...)
      - In Windows: use s.java_home if provided
      - Otherwise: leave whatever the OS already has (or user .bashrc)
    """
    if _is_windows():
        if s.java_home:
            os.environ["JAVA_HOME"] = s.java_home
        return

    # WSL/Linux
    if s.java_home and not _is_windows_path(s.java_home):
        os.environ["JAVA_HOME"] = s.java_home
        return

    # If settings has a Windows path, ignore it and rely on system env (.bashrc etc.)
    if s.java_home and _is_windows_path(s.java_home):
        logger.warning("Ignoring Windows-style JAVA_HOME from settings inside WSL/Linux: %s", s.java_home)

    # If JAVA_HOME already set correctly in shell, keep it. If not set, we can optionally set a sensible default.
    if not os.environ.get("JAVA_HOME"):
        candidate = "/usr/lib/jvm/java-17-openjdk-amd64"
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate


def _log_runtime_diagnostics(s: Settings, driver_py: str, worker_py: str) -> None:
    logger.info("=== Spark runtime diagnostics ===")
    logger.info("platform=%s", platform.platform())
    logger.info("is_windows=%s is_wsl=%s", _is_windows(), _is_wsl())
    logger.info("env=%s spark_master=%s app_name=%s", s.env, s.spark_master, s.spark_app_name)
    logger.info("sys.executable=%s", sys.executable)
    logger.info("PYSPARK_DRIVER_PYTHON=%s", os.environ.get("PYSPARK_DRIVER_PYTHON"))
    logger.info("PYSPARK_PYTHON=%s", os.environ.get("PYSPARK_PYTHON"))
    logger.info("resolved_driver_python=%s", driver_py)
    logger.info("resolved_worker_python=%s", worker_py)
    logger.info("JAVA_HOME=%s", os.environ.get("JAVA_HOME"))
    logger.info("SPARK_HOME=%s", os.environ.get("SPARK_HOME"))
    logger.info("HADOOP_HOME=%s", os.environ.get("HADOOP_HOME"))
    logger.info("winutils=%s", shutil.which("winutils"))
    logger.info("===============================")


# -------------------------
# Public API
# -------------------------

# src/spark/session.py
# ... (imports and helpers stay the same) ...

def get_spark(app_name: str | None = None, *, debug: bool = False) -> SparkSession:
    """
    Create a SparkSession using settings from .env/config.py.
    """
    s = load_settings()

    _sanitize_env_for_pyspark()
    _configure_java_home(s)

    if _is_wsl():
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        
    driver_py, worker_py = _resolve_python(s)

    os.environ["PYSPARK_DRIVER_PYTHON"] = driver_py
    os.environ["PYSPARK_PYTHON"] = worker_py

    if debug:
        _log_runtime_diagnostics(s, driver_py, worker_py)

    builder = cast(Any, SparkSession.builder)

    b = (
        builder
        .master(s.spark_master)
        .appName(app_name or s.spark_app_name)
        
        # Pulling from .env/Settings to handle the 1.6GB Word2Vec model
        .config("spark.driver.memory", s.spark_driver_memory)
        .config("spark.driver.maxResultSize", getattr(s, 'spark_driver_max_result_size', '4g'))
        .config("spark.executor.memory", "2g")
        .config("spark.memory.fraction", "0.6")  
        
        # Ensure python consistency
        .config("spark.pyspark.driver.python", driver_py)
        .config("spark.pyspark.python", worker_py)

        # Performance and Monitoring
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.showConsoleProgress", "true")
        
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )

    # Windows-specific logic remains unchanged
    if _is_windows():
        b = b.config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2") \
             .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")

    return b.getOrCreate()

    if _is_windows():
        b = (
            b
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
            .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
            .config("spark.hadoop.fs.file.impl.disable.cache", "true")
        )

    return b.getOrCreate()


def spark_doctor(spark: SparkSession, *, out_dir: Optional[str] = None) -> None:
    """
    Quick self-test to isolate environment issues:
      - prints core runtime details
      - JVM parquet write
      - Python-created DF parquet write
    """
    from pyspark.sql import Row

    print("spark.version:", spark.version)
    print("hadoop.version:", spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion())  # type: ignore
    print("spark.pyspark.driver.python:", spark.conf.get("spark.pyspark.driver.python", "NOT SET"))
    print("spark.pyspark.python:", spark.conf.get("spark.pyspark.python", "NOT SET"))
    print("JAVA_HOME:", os.environ.get("JAVA_HOME"))
    print("SPARK_HOME:", os.environ.get("SPARK_HOME"))
    print("HADOOP_HOME:", os.environ.get("HADOOP_HOME"))
    print("winutils:", shutil.which("winutils"))
    print("is_wsl:", _is_wsl(), "is_windows:", _is_windows())

    base = Path(out_dir) if out_dir else Path.cwd() / "spark_doctor_tmp"
    base.mkdir(parents=True, exist_ok=True)

    jvm_path = str(base / "_jvm_smoke.parquet")
    spark.range(2).write.mode("overwrite").parquet(jvm_path)
    print("JVM parquet write OK ->", jvm_path)

    py_path = str(base / "_py_smoke.parquet")
    spark.createDataFrame([Row(x=1), Row(x=2)]).write.mode("overwrite").parquet(py_path)
    print("Python parquet write OK ->", py_path)