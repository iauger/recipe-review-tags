from __future__ import annotations

import logging
from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F
from pyspark.sql import types as T
from src.utils.text_cleaning import normalize_text_spark

logger = logging.getLogger(__name__)


def python_list_string_to_array_str(colname: str) -> Column:
    """
    Convert a string that looks like a Python list of strings into array<string>.
    """
    s = F.col(colname)
    s_json = F.regexp_replace(s, r"'", '"')
    return F.from_json(s_json, T.ArrayType(T.StringType()))


def python_list_string_to_array_double(colname: str) -> Column:
    """
    Convert a string that looks like a Python list of numerics into array<double>.
    """
    s = F.col(colname)
    s_json = F.regexp_replace(s, r"'", '"')
    return F.from_json(s_json, T.ArrayType(T.DoubleType()))


def dtype_corrections(df: DataFrame) -> DataFrame:
    # Spark casts are "coerce-like": invalid parses become null (similar to errors="coerce")
    return (
        df.withColumn("id", F.col("id").cast("string"))
          .withColumn("minutes", F.col("minutes").cast("double"))
          .withColumn("contributor_id", F.col("contributor_id").cast("string"))
          # If submitted is yyyy-MM-dd, to_date is fine. If it includes time, to_timestamp is safer.
          .withColumn("submitted", F.to_timestamp(F.col("submitted")))
          .withColumn("n_steps", F.col("n_steps").cast("double"))
          .withColumn("n_ingredients", F.col("n_ingredients").cast("double"))
    )


def clean_ingredients(df: DataFrame) -> DataFrame:
    # Parse -> normalize each ingredient -> replace spaces with underscores -> join
    arr = python_list_string_to_array_str("ingredients")
    cleaned_arr = F.transform(
        arr,
        lambda x: F.regexp_replace(normalize_text_spark(x), r"\s+", "_")
    )
    return df.withColumn("ingredients_clean", F.concat_ws(" ", cleaned_arr))


def clean_steps(df: DataFrame) -> DataFrame:
    # Parse -> join steps -> normalize
    arr = python_list_string_to_array_str("steps")
    joined = F.concat_ws(" ", arr)
    return df.withColumn("steps_clean", normalize_text_spark(joined))


def clean_tags(df: DataFrame) -> DataFrame:
    # Parse -> normalize each tag -> join
    arr = python_list_string_to_array_str("tags")
    cleaned_arr = F.transform(arr, lambda x: normalize_text_spark(x))
    return df.withColumn("tags_clean", F.concat_ws(" ", cleaned_arr))


def extract_nutrition(df: DataFrame) -> DataFrame:
    # Parse nutrition list -> extract indices into columns
    # Order: calories, fat, sugar, sodium, protein, saturated_fat, carbs
    arr = python_list_string_to_array_double("nutrition")

    return (
        df.withColumn("calories",       F.element_at(arr, 1))  # 1-indexed
          .withColumn("fat",            F.element_at(arr, 2))
          .withColumn("sugar",          F.element_at(arr, 3))
          .withColumn("sodium",         F.element_at(arr, 4))
          .withColumn("protein",        F.element_at(arr, 5))
          .withColumn("saturated_fat",  F.element_at(arr, 6))
          .withColumn("carbs",          F.element_at(arr, 7))
    )


def clean_description(df: DataFrame) -> DataFrame:
    return df.withColumn("description_clean", normalize_text_spark(F.col("description")))


def clean_recipes(df: DataFrame) -> DataFrame:
    df = dtype_corrections(df)
    df = clean_ingredients(df)
    df = clean_steps(df)
    df = clean_tags(df)
    df = extract_nutrition(df)
    df = clean_description(df)
    return df