from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Column

from src.utils.text_cleaning import normalize_text_spark

def dtype_corrections(df: DataFrame) -> DataFrame:
    df = (
        df.withColumn("user_id", F.col("user_id").cast("string"))
          .withColumn("recipe_id", F.col("recipe_id").cast("string"))
          .withColumn("rating", F.col("rating").cast("double"))
          .withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
    )

    df = df.filter(
        F.col("user_id").isNotNull()
        & F.col("recipe_id").isNotNull()
        & F.col("rating").isNotNull()
    )

    df = df.filter(F.col("rating") != F.lit(0.0))

    return df

def prune_by_length(df: DataFrame, min_tokens: int = 15, max_tokens: int = 120) -> DataFrame:
    return df.withColumn("token_count", F.size(F.split(F.col("review"), " "))) \
             .filter((F.col("token_count") >= min_tokens) & 
                     (F.col("token_count") <= max_tokens))   

def clean_reviews(df: DataFrame) -> DataFrame:
    if "review" not in df.columns:
        return df.withColumn("review_clean", F.lit(""))

    # Always overwrite review_clean (fully lazy)
    return df.withColumn("review_clean", normalize_text_spark("review"))


def clean_interactions(df: DataFrame) -> DataFrame:
    df = prune_by_length(df)
    df = clean_reviews(df)
    df = dtype_corrections(df)
    return df