# src/spark/ingestion/spark/schemas_spark.py
from __future__ import annotations

from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType,
)

# RAW_recipes.csv (based on your observed dtypes)
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

# RAW_interactions.csv (based on your observed dtypes)
RAW_INTERACTIONS_SCHEMA = StructType([
    StructField("user_id", LongType(), True),
    StructField("recipe_id", LongType(), True),
    StructField("date", StringType(), True),
    StructField("rating", LongType(), True),
    StructField("review", StringType(), True),
])