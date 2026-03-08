# src/spark/labeling/inference_pipeline.py
from cProfile import label
import json
import logging
from pathlib import Path
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.functions import vector_to_array

from src.config import load_settings
from src.spark.features import labels
from src.spark.features.embeddings import Word2VecSpec, add_word2vec_embeddings
from src.spark.features.labels import add_binary_label_cols
from src.spark.features.io import write_parquet
from src.spark.features.pipeline import add_token_union_column, TextFeatureSpec
from src.spark.features.similarity import _native_cosine_sim
from src.spark.labeling.taxonomy import get_tag_ids 

logger = logging.getLogger(__name__)

def run_full_corpus_inference(spark: SparkSession):
    # Context and Data Loading
    s = load_settings(prefer_latest_run=True) 
    raw_corpus = spark.read.parquet(s.silver_interactions_path)
    logger.info(f"🚀 Starting scale-out inference for {raw_corpus.count():,} reviews.")

    # Load Pipeline Artifacts
    prep_path = str(Path(s.features_pipeline_model_dir).joinpath("prep_model").resolve())
    w2v_path = str(Path(s.features_pipeline_model_dir).joinpath("w2v_model").resolve())
    
    prep_model = PipelineModel.load(prep_path)
    w2v_model = Word2VecModel.load(w2v_path)

    # Feature Engineering 
    spec = TextFeatureSpec(text_col="review_clean", output_col="features", token_union_col="tokens_all")
    w2v_spec = Word2VecSpec(input_col="tokens_all", output_col="review_embeddings")
    
    # Re-transform labeled data with new model
    df_prep = add_token_union_column(prep_model.transform(raw_corpus), spec)
    df_embed = add_word2vec_embeddings(df_prep, w2v_model, w2v_spec)
    df_embed = df_embed.withColumn("features", F.col("review_embeddings"))
    
    # Add Binary Label Columns (for evaluation and potential pruning)
    labels = get_tag_ids()
    df_embed = add_binary_label_cols(df_embed, labels)
    
    df_vectorized = w2v_model.transform(df_prep)

    # Calibration Context
    with open(s.features_metrics_path, "r") as f:
        metrics = json.load(f)
    thresholds = metrics["tag_thresholds"]
    centroids_df = spark.read.parquet(s.features_tag_centroids_path)
    
    local_centroids = {row['tag']: row['centroid'] for row in centroids_df.collect()}

    # Calculate all 17 similarities
    sim_exprs = [
        _native_cosine_sim(
            vector_to_array(F.col("features")), 
            F.lit(local_centroids[label])
        ).alias(f"sim_{label}")
        for label in labels if label in local_centroids
    ]

    scored_df = df_vectorized.select("*", *sim_exprs)

    # Apply Binary Thresholds
    final_cols = [F.col(c) for c in raw_corpus.columns] 
    final_cols += [F.col(f"sim_{label}") for label in labels] # Keep for negation pruning
    final_cols += [
        (F.col(f"sim_{label}") >= F.lit(thresholds[label])).cast("int").alias(f"pred_{label}")
        for label in labels
    ]

    gold_final_df = scored_df.select(*final_cols)

    # Materialize and Save
    output_path = f"{s.gold_dir}/gold_labeled_reviews_{s.features_run_id}.parquet"
    write_parquet(gold_final_df.repartition(40), output_path)
    
    logger.info(f"✅ Full corpus labeled and saved to {output_path}")

if __name__ == "__main__":
    from src.spark.session import get_spark
    spark = get_spark()
    run_full_corpus_inference(spark)