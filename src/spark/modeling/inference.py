# src/spark/labeling/inference_pipeline.py
import logging
from pyspark.sql import SparkSession, functions as F
from src.config import load_settings
from src.spark.features.io import read_feature_dataset, write_parquet
from src.spark.features.similarity import _native_cosine_sim 

logger = logging.getLogger(__name__)

def run_full_corpus_inference(spark: SparkSession):
    """
    Applies calibrated semantic thresholds to the 1.4M review corpus.
    Produces the final 'Gold' dataset for the Deep Learning MLP.
    """
    # 1. Load the Marathon Run context
    s = load_settings(prefer_latest_run=True) 
    logger.info(f"🚀 Starting Inference on Run: {s.features_run_id}")

    # 2. Load the full featured dataset (1.4M rows with 128-dim vectors)
    df = read_feature_dataset(spark, s)
    
    # 3. Load Metrics (Centroids and Thresholds) from the marathon run
    import json
    with open(s.features_metrics_path, "r") as f:
        metrics = json.load(f)
    
    thresholds = metrics["tag_thresholds"]
    # We'll need the centroid vectors (stored in a separate Parquet usually)
    centroids_df = spark.read.parquet(s.features_tag_centroids_path)
    tag_list = centroids_df.select("tag").distinct().rdd.flatMap(lambda x: x).collect()

    logger.info(f"Applying thresholds for {len(tag_list)} tags...")

    # 4. Scale-Out Labeling (Algorithm 6 logic)
    # For each tag, calculate similarity to centroid and apply threshold
    scored_df = df
    for tag in tag_list:
        # Extract the specific centroid vector for this tag
        row = (
            centroids_df
            .filter(F.col("tag") == tag)
            .select("centroid")
            .first()
        )

        if row is None:
            raise ValueError(f"No centroid found for tag={tag!r}")

        centroid_vec = row["centroid"]  # or row[0]

        t_val = thresholds[tag]
        
        # Calculate Cosine Similarity (Review_Vector · Centroid_Vector)
        # We create a binary column: 1 if similarity >= threshold, else 0
        sim_col = f"sim_{tag}"
        label_col = f"pred_{tag}"
        
        scored_df = scored_df.withColumn(sim_col, _native_cosine_sim(F.col("features"), F.lit(centroid_vec)))
        scored_df = scored_df.withColumn(label_col, F.when(F.col(sim_col) >= t_val, 1).otherwise(0))

    # 5. Drop intermediate similarity scores to keep the dataset lean
    final_cols = [c for c in scored_df.columns if not c.startswith("sim_")]
    gold_df = scored_df.select(*final_cols)

    # 6. Save as the 'Gold' Feature Set for the MLP Phase
    output_path = f"{s.processed_dir}/gold_labeled_reviews_{s.features_run_id}.parquet"
    write_parquet(gold_df, output_path)
    
    logger.info(f"✅ Successfully labeled {gold_df.count()} reviews.")
    logger.info(f"📂 Gold Dataset: {output_path}")

if __name__ == "__main__":
    from src.spark.session import get_spark
    spark = get_spark()
    run_full_corpus_inference(spark)