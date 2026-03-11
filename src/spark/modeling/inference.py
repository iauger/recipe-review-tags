# src/spark/labeling/inference.py
import json
import logging
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

from src.config import load_settings
from src.spark.features.io import write_parquet
from src.spark.features.nlp_pipeline import TextFeatureSpec, Word2VecSpec, add_token_union_column, add_word2vec_embeddings
from src.spark.features.semantic_space import _native_cosine_sim
from src.spark.modeling.postprocessing import resolve_negation_conflicts

logger = logging.getLogger(__name__)

def apply_feature_pipeline(df, prep_model, w2v_model, spec, w2v_spec):
    """Applies tokenization and Word2Vec to incoming raw reviews."""
    df_prep = add_token_union_column(prep_model.transform(df), spec)
    df_embed = add_word2vec_embeddings(df_prep, model=w2v_model, spec=w2v_spec)
    return df_embed.withColumn("features", F.col(w2v_spec.output_col))

def run_scale_out_inference(spark: SparkSession):
    s = load_settings(prefer_latest_run=True)
    logger.info(f"🚀 Starting scale-out inference for run {s.features_run_id}")

    # 1. Load Data & Models
    raw_corpus = spark.read.parquet(s.silver_interactions_path)
    
    prep_path = str(Path(s.features_pipeline_model_dir) / "prep_model")
    w2v_path = str(Path(s.features_pipeline_model_dir) / "w2v_model")
    prep_model = PipelineModel.load(prep_path)
    w2v_model = Word2VecModel.load(w2v_path)

    # Use the finalized 128-D / Window-5 specs 
    spec = TextFeatureSpec(text_col="review_clean", output_col="features", token_union_col="tokens_all")
    w2v_spec = Word2VecSpec(input_col="tokens_all", output_col="review_embeddings", vector_size=128, window_size=5)

    # 2. Vectorize Corpus
    full_featured_df = apply_feature_pipeline(raw_corpus, prep_model, w2v_model, spec, w2v_spec)

    # 3. Load Calibration Context (Centroids & Thresholds)
    with open(s.features_metrics_path, "r") as f:
        metrics = json.load(f)
    thresholds = metrics["tag_thresholds"]
    
    centroids_df = spark.read.parquet(s.features_tag_centroids_path)
    local_centroids = {row['tag']: row['centroid'] for row in centroids_df.collect()}

    # 4. Single-Pass Similarity Projection
    sim_exprs = [
        _native_cosine_sim(vector_to_array(F.col("features")), F.lit(local_centroids[t])).alias(f"sim_{t}")
        for t in thresholds.keys() if t in local_centroids
    ]
    scored_df = full_featured_df.select("*", *sim_exprs)

    # 5. Apply Automated Thresholds
    final_cols = [F.col(c) for c in raw_corpus.columns] + \
                 [F.col(f"sim_{t}") for t in thresholds.keys()] + \
                 [(F.col(f"sim_{t}") >= F.lit(thresholds[t])).cast("int").alias(f"pred_{t}") 
                  for t in thresholds.keys()]
    
    scored_df = scored_df.select(*final_cols)

    # 6. Post-processing (Prune contradictions)
    pruned_df = resolve_negation_conflicts(scored_df)

    # 7. Materialize and Save Gold Dataset
    output_path = f"{s.gold_dir}/gold_labeled_reviews_{s.features_run_id}.parquet"
    write_parquet(pruned_df.repartition(40), output_path)
    
    logger.info(f"✅ Full corpus labeled, pruned, and saved to {output_path}")

if __name__ == "__main__":
    from src.spark.session import get_spark
    spark = get_spark()
    run_scale_out_inference(spark)