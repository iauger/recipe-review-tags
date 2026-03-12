# src/spark/features/build_features.py
from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import load_settings
from src.spark.features.io import (
    ensure_dirs, 
    read_labeled_reviews, 
    write_manifest, 
    write_metrics, 
    write_parquet
)
from src.spark.labeling.taxonomy import get_tag_ids
from src.spark.features.dataset_prep import (
    assign_recipe_splits, 
    add_binary_label_cols, 
    get_label_cols
)
from src.spark.features.nlp_pipeline import (
    TextFeatureSpec, 
    Word2VecSpec, 
    build_prep_pipeline, 
    add_token_union_column, 
    fit_word2vec, 
    add_word2vec_embeddings, 
    drop_intermediate_columns
)
from src.spark.features.semantic_space import (
    PrototypeSpec, 
    build_tag_centroids, 
    calculate_diagonal_thresholds
)

logger = logging.getLogger(__name__)

# --- Metrics Helpers ---
def _split_counts(df: DataFrame, split_col: str = "split") -> dict[str, int]:
    rows = df.groupBy(split_col).count().collect()
    return {r[split_col]: int(r["count"]) for r in rows}

def _label_prevalence_by_split(df: DataFrame, label_cols: list[str], split_col: str = "split") -> dict[str, dict[str, float]]:
    agg_exprs = [F.avg(F.col(c).cast("double")).alias(c) for c in label_cols]
    rows = df.groupBy(split_col).agg(*agg_exprs).collect()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        d = r.asDict()
        split = d.pop(split_col, "unknown")
        out[split] = {k: float(v) if v is not None else 0.0 for k, v in d.items()}
    return out

# --- Main Builder ---
def build_features(
    spark: SparkSession,
    *,
    labels: list[str] | None = None,
    labeled_data_path: str | None = None,
) -> None:
    """
    Fits Word2Vec on the 1M+ review corpus for robust semantic context, 
    then calibrates thresholds using the Zero-Shot labeled data.
    """
    if labels is None:
        labels = get_tag_ids() 

    s = load_settings(prefer_latest_run=False)
    ensure_dirs(s)

    logger.info("=== Full Corpus Feature Build ===")
    
    # Load Data
    label_path = labeled_data_path if labeled_data_path else s.labeled_gold_reviews_path
    labeled_df = read_labeled_reviews(spark, label_path=label_path)
    full_corpus_df = spark.read.parquet(s.silver_interactions_path)
    
    # Prep Labeled Set (Splits & Labels)
    labeled_df = assign_recipe_splits(labeled_df, recipe_id_col="recipe_id")
    labeled_df = add_binary_label_cols(labeled_df, labels)
    label_cols = get_label_cols(labeled_df)

    # Fit Prep Pipeline on full corpus
    spec = TextFeatureSpec(
        text_col="review_clean", 
        output_col="features", 
        token_union_col="tokens_all",
        min_token_count=11,
        max_token_count=120,
        keep_negations=False,
        extra_stopwords=(
            'however', 'actually', 'although', 'btw', 'besides', 'anyway', 
            'anyhow', 'regardless', 'also', 'plus', 'since', 'guess', 'think', 
            'maybe', 'perhaps', 'suspect', 'wonder', 'wondering', 'suppose', 
            'lol', 'amp', 'basically', 'totally', 'rather', 'meant', 'case', 
            'favor', 'course', 'especially', 'probably', 'kind', 'sort', 'cause'
        )
    )
    
    logger.info("Fitting Text Prep Pipeline on full corpus...")
    prep_model = build_prep_pipeline(spec).fit(full_corpus_df)
    prep_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/prep_model") 
    
    # Fit Word2Vec on full corpus
    w2v_spec = Word2VecSpec(
        input_col=spec.token_union_col, 
        output_col="review_embeddings", 
        vector_size=128,
        window_size=5,
        min_count=10
    )
    
    logger.info("Transforming full corpus for Word2Vec training...")
    full_tokens = add_token_union_column(prep_model.transform(full_corpus_df), spec).select("tokens_all")
    
    full_tokens_filtered = full_tokens.withColumn("token_count", F.size(F.col("tokens_all"))) \
        .filter((F.col("token_count") >= spec.min_token_count) & (F.col("token_count") <= spec.max_token_count)) \
        .select("tokens_all").repartition(200) 
    
    logger.info("Fitting Word2Vec on 1M+ reviews...")
    w2v_model = fit_word2vec(full_tokens_filtered, spec=w2v_spec)
    w2v_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/w2v_model")
    
    # Transform the Labeled Set using the fitted pipeline & embeddings
    logger.info("Generating embeddings for the labeled subset...")
    df_prep = add_token_union_column(prep_model.transform(labeled_df), spec)
    df_embed = add_word2vec_embeddings(df_prep, model=w2v_model, spec=w2v_spec)
    df_embed = df_embed.withColumn(spec.output_col, F.col(w2v_spec.output_col))
    
    # Calibrate Centroids & Thresholds 
    p_spec = PrototypeSpec(features_col=spec.output_col, label_prefix="y_")
    centroids_df = build_tag_centroids(df_embed, spec=p_spec, labels=labels)
    threshold_map = calculate_diagonal_thresholds(labeled_df=df_embed, centroids_df=centroids_df, spec=p_spec)

    # Write Dataset
    df_out = drop_intermediate_columns(df_embed, spec)
    keep_cols = ["review_key", "user_id", "recipe_id", "date", "rating", "liked", "review_clean", "split", spec.output_col] + label_cols
    df_out = df_out.select(*[c for c in keep_cols if c in df_out.columns])

    write_parquet(df_out, s.features_dataset_path, partition_cols=None)

    # Metrics + manifest
    metrics = {
        "split_counts": _split_counts(df_out, "split"),
        "label_prevalence_by_split": _label_prevalence_by_split(df_out, label_cols, "split"),
        "w2v_vector_size": w2v_spec.vector_size,
        "w2v_window_size": w2v_spec.window_size,
        "w2v_min_count": w2v_spec.min_count,
        "tag_thresholds": threshold_map
    }
    write_metrics(s, metrics)
    write_manifest(s, {"feature_spec": spec.__dict__, "w2v_spec": w2v_spec.__dict__, "metrics_path": s.features_metrics_path})
    
    # LATEST_RUN pointers
    latest_path = Path(s.features_dir) / "LATEST_RUN"
    if latest_path.exists():
        current_stack = latest_path.read_text().strip()
        if not current_stack.endswith(s.features_run_id):
            latest_path.write_text(f"{current_stack}, {s.features_run_id}")
    else:
        latest_path.write_text(s.features_run_id)
    
    logger.info("=== Done ===")
    
if __name__ == "__main__":
    from src.spark.session import get_spark 
    spark = get_spark()
    build_features(spark)
    spark.stop()