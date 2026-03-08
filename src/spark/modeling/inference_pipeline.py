# src/spark/labeling/inference_pipeline.py
import logging
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.config import load_settings
from src.spark.features import labels
from src.spark.features.embeddings import Word2VecSpec, add_word2vec_embeddings
from src.spark.features.labels import add_binary_label_cols
from src.spark.features.io import write_metrics
from src.spark.features.pipeline import add_token_union_column, TextFeatureSpec
from src.spark.features.prototypes import PrototypeSpec, build_tag_centroids
from src.spark.features.similarity import _native_cosine_sim
from src.spark.features.splits import assign_recipe_splits
from src.spark.features.thresholds import calculate_diagonal_thresholds
from src.spark.labeling.taxonomy import get_tag_ids 

logger = logging.getLogger(__name__)

def apply_feature_pipeline(df, prep_model, w2v_model, spec, w2v_spec):
    """Ensures any DF matches the v7 Word2Vec semantic space."""
    # 1. Tokenize/N-gram -> 2. Vectorize -> 3. Alias 'features'
    df_prep = add_token_union_column(prep_model.transform(df), spec)
    df_embed = add_word2vec_embeddings(df_prep, model=w2v_model, spec=w2v_spec)
    return df_embed.withColumn("features", F.col("review_embeddings"))

from pyspark.sql import DataFrame


def resolve_negation_conflicts(df: DataFrame) -> DataFrame:
    """
    Resolves logical contradictions in multi-label assignments.
    If a review triggers both sides of a negation pair (e.g., 'dry' and 'moist'),
    the tag with the lower cosine similarity score is pruned.
    """
    negation_pairs = [
        ("would_make_again", "would_not_make_again"),
        ("delicious_tasty", "bland_lacks_flavor"),
        ("moist_tender", "dry"),
        ("crispy_crunchy", "mushy_soggy"),
        ("easy_quick", "time_consuming_complex")
    ]
    
    pruned_df = df
    for pos, neg in negation_pairs:
        s_pos, s_neg = f"sim_{pos}", f"sim_{neg}"
        p_pos, p_neg = f"pred_{pos}", f"pred_{neg}"
        
        # Resolve: Keep the label with the strongest semantic signal
        pruned_df = pruned_df.withColumn(
            p_pos,
            F.when((F.col(p_pos) == 1) & (F.col(p_neg) == 1) & (F.col(s_pos) <= F.col(s_neg)), 0)
             .otherwise(F.col(p_pos))
        )
        
        pruned_df = pruned_df.withColumn(
            p_neg,
            F.when((F.col(p_pos) == 1) & (F.col(p_neg) == 1) & (F.col(s_neg) < F.col(s_pos)), 0)
             .otherwise(F.col(p_neg))
        )
    
    # Recalculate consolidated tag lists for downstream analytics
    pred_cols = [c for c in pruned_df.columns if c.startswith("pred_")]
    tags_array_expr = F.array([
        F.when(F.col(c) == 1, F.lit(c.replace("pred_", ""))).otherwise(F.lit(None))
        for c in pred_cols
    ])
    
    return (
        pruned_df
        .withColumn("tag_list", F.array_remove(tags_array_expr, None))
        .withColumn("total_tags", F.size(F.col("tag_list")))
    )

def run_marathon_completion(spark: SparkSession):
    s = load_settings(prefer_latest_run=True)
    logger.info(f"Continuing Marathon Run: {s.features_run_id}")

    # --- CALIBRATION & VALIDATION ---
    # Load Ground Truth and Models
    labeled_df = spark.read.parquet(s.labeled_gold_reviews_path)
    prep_model = PipelineModel.load(f"{s.features_pipeline_model_dir}/prep_model")
    w2v_model = Word2VecModel.load(f"{s.features_pipeline_model_dir}/w2v_model")
    
    spec = TextFeatureSpec(text_col="review_clean", output_col="features", token_union_col="tokens_all")
    w2v_spec = Word2VecSpec(input_col="tokens_all", output_col="review_embeddings")

    # Mirror the labeled data to the new vector space
    df_embed = apply_feature_pipeline(labeled_df, prep_model, w2v_model, spec, w2v_spec)
    df_embed = assign_recipe_splits(add_binary_label_cols(df_embed, get_tag_ids()))

    # Build/Save Centroids
    p_spec = PrototypeSpec(features_col="features", label_prefix="y_", split_col="split")
    centroids_df = build_tag_centroids(df_embed, spec=p_spec, labels=get_tag_ids())
    centroids_df.write.mode("overwrite").parquet(s.features_tag_centroids_path)

    # Calculate Thresholds 
    thresholds = calculate_diagonal_thresholds(df_embed, centroids_df, spec=p_spec)
    write_metrics(s, {"tag_thresholds": thresholds})

    # --- SCALE-OUT INFERENCE ---
    raw_corpus = spark.read.parquet(s.gold_reviews_path) # Cleaned and filtered for length
    full_featured_df = apply_feature_pipeline(raw_corpus, prep_model, w2v_model, spec, w2v_spec)

    local_centroids = {row['tag']: row['centroid'] for row in centroids_df.collect()}
    
    # Single-Pass Similarity Projection
    sim_exprs = [
        _native_cosine_sim(vector_to_array(F.col("features")), F.lit(local_centroids[t])).alias(f"sim_{t}")
        for t in thresholds.keys() if t in local_centroids
    ]
    
    scored_df = full_featured_df.select("*", *sim_exprs)
    
    # Apply Thresholds
    final_cols = [F.col(c) for c in raw_corpus.columns] + \
                 [F.col(f"sim_{t}") for t in thresholds.keys()] + \
                 [(F.col(f"sim_{t}") >= F.lit(thresholds[t])).cast("int").alias(f"pred_{t}") 
                  for t in thresholds.keys()]
    
    scored_df = scored_df.select(*final_cols)
    pruned_df = resolve_negation_conflicts(scored_df)

    # Save final Gold Dataset
    output_path = f"{s.gold_dir}/gold_labeled_reviews_{s.features_run_id}.parquet"
    pruned_df.repartition(40).write.mode("overwrite").parquet(output_path)