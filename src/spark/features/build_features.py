# src/spark/features/build_features.py
from __future__ import annotations

import json
import logging
from typing import Any
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import load_settings
from src.spark.features.io import (
    ensure_dirs,
    read_labeled_reviews,
    write_manifest,
    write_metrics,
    write_parquet,
)
from src.spark.labeling.taxonomy import get_tag_ids
from src.spark.features.splits import assign_recipe_splits, build_splits_table
from src.spark.features.labels import add_binary_label_cols, get_label_cols
from src.spark.features.pipeline import TextFeatureSpec, build_prep_pipeline, build_tfidf_pipeline, add_token_union_column, drop_intermediate_columns
from src.spark.features.embeddings import Word2VecSpec, fit_word2vec, add_word2vec_embeddings
from src.spark.features.prototypes import PrototypeSpec, build_tag_centroids
from src.spark.features.thresholds import calculate_diagonal_thresholds

logger = logging.getLogger(__name__)


# -------------------------
# Helpers
# -------------------------

def _split_counts(df: DataFrame, split_col: str = "split") -> dict[str, int]:
    rows = df.groupBy(split_col).count().collect()
    return {r[split_col]: int(r["count"]) for r in rows}


def _label_prevalence_by_split(df: DataFrame, label_cols: list[str], split_col: str = "split") -> dict[str, dict[str, float]]:
    """
    Returns:
      {
        "train": {"y_a": 0.12, "y_b": 0.03, ...},
        "val":   {...},
        "test":  {...},
      }
    """
    agg_exprs = [F.avg(F.col(c).cast("double")).alias(c) for c in label_cols]
    rows = df.groupBy(split_col).agg(*agg_exprs).collect()

    out: dict[str, dict[str, float]] = {}
    for r in rows:
        split = r[split_col]
        d = r.asDict()
        d.pop(split_col, None)
        out[split] = {k: float(v) if v is not None else 0.0 for k, v in d.items()}
    return out


def _concat_tokens_and_ngrams(df: DataFrame, token_ns_col: str, ngram_col: str, out_col: str) -> DataFrame:
    return df.withColumn(out_col, F.concat(F.col(token_ns_col), F.col(ngram_col)))

# -------------------------
# Main builder
# -------------------------

def build_features(
    spark: SparkSession,
    *,
    labels: list[str] | None = None,
    labeled_data_path: str | None = None,
) -> None:
    """
    End-to-end feature build for labeled gold set:
      - read labeled reviews
      - assign recipe-level splits
      - add binary y_* labels
      - fit prep pipeline on train
      - add tokens_all (tokens_nostop + ngrams)
      - fit TF-IDF pipeline on train
      - write splits + dataset + metrics + manifest
    """
    if labels is None:
        labels = ["delicious_tasty", "ingredient_issue", "bland_lacks_flavor"]

    s = load_settings(prefer_latest_run=False)
    ensure_dirs(s)

    logger.info("=== Build Features Run ===")
    logger.info("features_run_id=%s", s.features_run_id)
    logger.info("run_dir=%s", s.features_run_dir)

    # -------------------------
    # Read + split + labels
    # -------------------------
    labeled_data_path = labeled_data_path if labeled_data_path else s.labeled_gold_reviews_path
    df = read_labeled_reviews(spark, label_path=labeled_data_path)

    df = assign_recipe_splits(
        df,
        recipe_id_col="recipe_id",
        split_col="split",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=42,
    )

    df = add_binary_label_cols(df, labels, source_col="zs_labels", prefix="y_",output_dtype="int")
    label_cols = get_label_cols(df, prefix="y_")

    # Save splits table early (useful even if feature build fails later)
    splits_df = build_splits_table(df, review_key_col="review_key", recipe_id_col="recipe_id", split_col="split")
    write_parquet(splits_df, s.features_splits_path)

    # -------------------------
    # Text pipeline: prep
    # -------------------------
    spec = TextFeatureSpec(
        text_col="review_clean",
        output_col="features",
        token_col="tokens",
        token_union_col="tokens_all",
        enable_ngrams=True,
        ngram_n=3,
        ngram_col="ngrams",
        num_features=262_144,
        binary_tf=False,
        min_doc_freq=1,
        keep_intermediate=False,
        tfidf_included=False,
    )

    train_df = df.filter(F.col("split") == F.lit("train"))

    prep_model = build_prep_pipeline(spec).fit(train_df)
    df_prep = prep_model.transform(df)
    df_prep = add_token_union_column(df_prep, spec)
    
    # -------------------------
    # Word2Vec embeddings (fit on train only)
    # -------------------------
    w2v_spec = Word2VecSpec(
        input_col=spec.token_union_col,  # "tokens_all"
        output_col="review_embeddings",
        vector_size=128,
        window_size=5,
        min_count=2,
        max_iter=10,
        seed=42,
    )

    train_tokens = (
        df_prep
        .filter(F.col("split") == "train")
        .select("tokens_all")   
    )
    w2v_model = fit_word2vec(train_tokens, spec=w2v_spec)

    # Save model
    w2v_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/w2v_model")
    df_embed = add_word2vec_embeddings(df_prep, model=w2v_model, spec=w2v_spec)

    # Canonical representation for downstream: embeddings => features
    df_embed = df_embed.withColumn(spec.output_col, F.col(w2v_spec.output_col))
    _ = df_embed.count()   # materialize cache

    # -------------------------
    # Tag Centroids
    # -------------------------
    logger.info("Building Tag Centroids (Semantic Anchors)...")
    p_spec = PrototypeSpec(
        features_col=spec.output_col, # "features"
        label_prefix="y_"
    )
    
    # Generate centroids from the training set silver labels
    centroids_df = build_tag_centroids(df_embed, spec=p_spec, labels=labels)
    
    # Save centroids for the downstream scale-out labeling
    write_parquet(centroids_df, s.features_tag_centroids_path)
    logger.info("Saved centroids to %s", s.features_tag_centroids_path)

    # -------------------------
    # Automated Thresholding
    # -------------------------
    logger.info("Calculating Data-Driven Similarity Thresholds...")
    
    # This implements the "diagonal" self-similarity thresholds from the paper
    threshold_map = calculate_diagonal_thresholds(
        labeled_df=df_embed,
        centroids_df=centroids_df,
        spec=p_spec
    )
    
    # -------------------------
    # TF-IDF pipeline
    # -------------------------
    if spec.tfidf_included:
        tfidf_model = build_tfidf_pipeline(spec).fit(df_embed.filter(F.col("split") == F.lit("train")))
        tfidf_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/tfidf_model")
        df_feat = tfidf_model.transform(df_embed)

        # Drop TF intermediate if present
        if "features_tfidf_tf" in df_feat.columns:
            df_feat = df_feat.drop("features_tfidf_tf")
    else:
        df_feat = df_embed

    # Now safe to drop tokens/ngrams/tf columns
    df_feat = drop_intermediate_columns(df_feat, spec)
    _ = df_feat.count()

    # -------------------------
    # Write dataset
    # -------------------------
    # Keep a lean set of columns + features vector + labels + split
    base_cols = [
        "review_key",
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "liked",
        "review_clean",
        "split",
    ]

    extra_cols = [spec.output_col]  # "features" = embeddings
    if w2v_spec.output_col in df_feat.columns:
        extra_cols.append(w2v_spec.output_col)  # keep raw name for debugging
    if "features_tfidf" in df_feat.columns:
        extra_cols.append("features_tfidf")

    keep_cols = [c for c in base_cols if c in df_feat.columns] + label_cols + extra_cols
    df_out = df_feat.select(*keep_cols)

    write_parquet(df_out, s.features_dataset_path, partition_cols=None)

    # Save models inside run directory
    prep_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/prep_model")        

    # -------------------------
    # Metrics + manifest
    # -------------------------
    metrics: dict[str, Any] = {}
    metrics["split_counts"] = _split_counts(df_out, "split")
    metrics["label_prevalence_by_split"] = _label_prevalence_by_split(df_out, label_cols, "split")
    metrics["num_features_hashing"] = spec.num_features
    metrics["ngram_n"] = spec.ngram_n if spec.enable_ngrams else None
    metrics["min_doc_freq"] = spec.min_doc_freq
    metrics["w2v_vector_size"] = w2v_spec.vector_size
    metrics["w2v_window_size"] = w2v_spec.window_size
    metrics["w2v_min_count"] = w2v_spec.min_count
    metrics["tag_thresholds"] = threshold_map
    metrics["split_counts"] = _split_counts(df_out, "split")

    write_metrics(s, metrics)

    manifest_payload = {
        "feature_spec": spec.__dict__,
        "w2v_spec": w2v_spec.__dict__,
        "metrics_path": s.features_metrics_path,
    }
    write_manifest(s, manifest_payload)
    
    latest_path = Path(s.features_dir) / "LATEST_RUN"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    # Spark parquet writes a directory. Check existence before updating pointer.
    if not Path(s.features_dataset_path).exists():
        raise FileNotFoundError(f"Feature dataset not found after write: {s.features_dataset_path}")

    latest_path.write_text(s.features_run_id + "\n")
    
    logger.info("Wrote LATEST_RUN pointer: %s -> %s", str(latest_path), s.features_run_id)
    logger.info("Wrote splits: %s", s.features_splits_path)
    logger.info("Wrote dataset: %s", s.features_dataset_path)
    logger.info("Wrote models dir: %s", s.features_pipeline_model_dir)
    logger.info("Wrote metrics: %s", s.features_metrics_path)
    logger.info("Wrote manifest: %s", s.features_manifest_path)
    logger.info("=== Done ===")


# src/spark/features/build_features.py

def build_features_with_full_corpus_w2v(
    spark: SparkSession,
    *,
    labels: list[str] | None = None,
    labeled_data_path: str | None = None,
) -> None:
    """
    Advanced Build: Fits Word2Vec on the 1M+ review corpus for robust 
    semantic context, then calibrates thresholds using labeled data.
    """
    if labels is None:
        labels = get_tag_ids() # Use all tags from taxonomy

    s = load_settings(prefer_latest_run=False)
    ensure_dirs(s)

    logger.info("=== Full Corpus Feature Build ===")
    
    # 1. Load Data
    label_path = labeled_data_path if labeled_data_path else s.labeled_gold_reviews_path
    labeled_df = read_labeled_reviews(spark, label_path=label_path)
    full_corpus_df = spark.read.parquet(s.silver_interactions_path)
    
    # 2. Prep Labeled Set (Splits & Labels)
    labeled_df = assign_recipe_splits(labeled_df, recipe_id_col="recipe_id")
    labeled_df = add_binary_label_cols(labeled_df, labels)
    label_cols = get_label_cols(labeled_df)

    # 3. Fit Prep Pipeline on FULL Corpus
    # This ensures "salty" and "acidic" are tokenized properly across all contexts
    spec = TextFeatureSpec(text_col="review_clean", output_col="features", token_union_col="tokens_all")
    
    logger.info("Fitting Text Prep Pipeline on full corpus...")
    prep_model = build_prep_pipeline(spec).fit(full_corpus_df)
    prep_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/prep_model") # Save early
    
    # 4. Fit Word2Vec on FULL Corpus
    w2v_spec = Word2VecSpec(input_col=spec.token_union_col, output_col="review_embeddings", vector_size=128)
    
    logger.info("Transforming full corpus for Word2Vec training...")
    full_tokens = add_token_union_column(prep_model.transform(full_corpus_df), spec).select("tokens_all")
    full_tokens = full_tokens.repartition(8) 
    
    logger.info("Fitting Word2Vec on 1M+ reviews...")
    w2v_model = fit_word2vec(full_tokens, spec=w2v_spec)
    w2v_model.write().overwrite().save(f"{s.features_pipeline_model_dir}/w2v_model")
    
    # 5. Transform the Labeled Set using the "Global" models
    logger.info("Generating embeddings for the labeled subset...")
    df_prep = add_token_union_column(prep_model.transform(labeled_df), spec)
    df_embed = add_word2vec_embeddings(df_prep, model=w2v_model, spec=w2v_spec)
    df_embed = df_embed.withColumn(spec.output_col, F.col(w2v_spec.output_col))
    
    # 6. Calibrate Centroids & Thresholds (Algorithms 4 & 5)
    p_spec = PrototypeSpec(features_col=spec.output_col, label_prefix="y_")
    centroids_df = build_tag_centroids(df_embed, spec=p_spec, labels=labels)
    
    threshold_map = calculate_diagonal_thresholds(
        labeled_df=df_embed, 
        centroids_df=centroids_df, 
        spec=p_spec
    )

    # -------------------------
    # Write dataset
    # -------------------------
    # Use existing helper to remove tokens, ngrams, and intermediate TF columns
    df_out = drop_intermediate_columns(df_embed, spec)
    
    # Define only the essential columns for the final Gold dataset
    keep_cols = [
        "review_key", "user_id", "recipe_id", "date", 
        "rating", "liked", "review_clean", "split",
        spec.output_col # This is your "features" (review_embeddings)
    ] + label_cols

    # Ensure we only select columns that actually exist
    final_cols = [c for c in keep_cols if c in df_out.columns]
    df_out = df_out.select(*final_cols)

    write_parquet(df_out, s.features_dataset_path, partition_cols=None)

    # -------------------------
    # Metrics + manifest
    # -------------------------
    metrics: dict[str, Any] = {}
    metrics["split_counts"] = _split_counts(df_out, "split")
    metrics["label_prevalence_by_split"] = _label_prevalence_by_split(df_out, label_cols, "split")
    metrics["num_features_hashing"] = spec.num_features
    metrics["ngram_n"] = spec.ngram_n if spec.enable_ngrams else None
    metrics["min_doc_freq"] = spec.min_doc_freq
    metrics["w2v_vector_size"] = w2v_spec.vector_size
    metrics["w2v_window_size"] = w2v_spec.window_size
    metrics["w2v_min_count"] = w2v_spec.min_count
    metrics["tag_thresholds"] = threshold_map
    metrics["split_counts"] = _split_counts(df_out, "split")

    write_metrics(s, metrics)

    manifest_payload = {
        "feature_spec": spec.__dict__,
        "w2v_spec": w2v_spec.__dict__,
        "metrics_path": s.features_metrics_path,
    }
    write_manifest(s, manifest_payload)
    
    latest_path = Path(s.features_dir) / "LATEST_RUN"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    # Spark parquet writes a directory. Check existence before updating pointer.
    if not Path(s.features_dataset_path).exists():
        raise FileNotFoundError(f"Feature dataset not found after write: {s.features_dataset_path}")

    if latest_path.exists():
        current_stack = latest_path.read_text().strip()
        # Only append if this ID isn't already the latest entry
        if not current_stack.endswith(s.features_run_id):
            new_stack = f"{current_stack}, {s.features_run_id}"
            latest_path.write_text(new_stack)
    else:
        latest_path.write_text(s.features_run_id)
    
    logger.info("Wrote LATEST_RUN pointer: %s -> %s", str(latest_path), s.features_run_id)
    logger.info("Wrote splits: %s", s.features_splits_path)
    logger.info("Wrote dataset: %s", s.features_dataset_path)
    logger.info("Wrote models dir: %s", s.features_pipeline_model_dir)
    logger.info("Wrote metrics: %s", s.features_metrics_path)
    logger.info("Wrote manifest: %s", s.features_manifest_path)
    logger.info("=== Done ===")
    
# Convenience CLI entrypoint
if __name__ == "__main__":
    from src.spark.session import get_spark  # adjust to your spark session helper if different

    spark = get_spark()
    build_features(spark)
    spark.stop()