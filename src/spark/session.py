from typing import Any, cast
import os
from pyspark.sql import SparkSession
from src.config import load_settings

def get_spark(app_name: str | None = None) -> SparkSession:
    """
    Create a SparkSession using settings from .env/config.py.
    """
    s = load_settings()

    # Avoid version mismatch if an external Spark install is present
    os.environ.pop("SPARK_HOME", None)
    os.environ.pop("PYSPARK_SUBMIT_ARGS", None)
    
    os.environ["PYSPARK_PYTHON"] = s.pyspark_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = s.pyspark_driver_python

    if s.java_home:
        os.environ["JAVA_HOME"] = s.java_home
    else:
        # If JAVA_HOME is not set in settings, try to find it from the system
        java_home = os.environ.get("JAVA_HOME")
        
        if java_home is not None:
            os.environ["JAVA_HOME"] = java_home

    builder = cast(Any, SparkSession.builder)

    spark = (
        builder
        .master(s.spark_master)
        .appName(app_name or s.spark_app_name)
        .config("spark.driver.memory", s.spark_driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    return spark