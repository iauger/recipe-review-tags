# src/spark/labeling/zero_shot.py
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.spark.labeling.taxonomy import get_label_to_hypotheses, get_tag_ids, get_tag_polarity

logger = logging.getLogger(__name__)

# --- CONFIG ---
@dataclass(frozen=True)
class ZeroShotConfig:
    taxonomy_version: str = "v1"
    text_col: str = "review_clean"
    key_col: str = "review_key"

    sample_n: int = 65_000
    sample_seed: int = 42

    model_id: str = "valhalla/distilbart-mnli-12-1"  
    batch_size: int = 4
    max_length: int = 128

    threshold: float = 0.55
    threshold_pos: float = 0.85
    threshold_neg: float = 0.40
    threshold_neu: float = 0.55

    fallback_top_k: int = 1
    fallback_allow_positive: bool = False  

# --- SPARK PREP & IO ---
def select_informative_subset(df: DataFrame, cfg: ZeroShotConfig) -> DataFrame:
    """Sample a fixed-size subset from the pre-filtered Gold reviews."""
    if cfg.key_col not in df.columns:
        raise ValueError(f"Expected key column '{cfg.key_col}' not found. Run build_gold_reviews() first.")

    return (
        df.orderBy(F.rand(cfg.sample_seed))
          .limit(int(cfg.sample_n))
          .select(cfg.key_col, cfg.text_col)
    )

def attach_zero_shot_labels(df_base: DataFrame, df_labels: DataFrame, key_col: str = "review_key") -> DataFrame:
    """Join zero-shot outputs back onto a Spark dataframe."""
    return df_base.join(df_labels, on=key_col, how="inner")

# --- LOCAL INFERENCE ---
def run_zero_shot_local(pdf: pd.DataFrame, cfg: ZeroShotConfig) -> pd.DataFrame:
    try:
        from transformers import pipeline
        from tqdm.auto import tqdm
        import time
    except ImportError as e:
        raise ImportError("Install transformers and tqdm for zero-shot inference.") from e

    label_to_hyps = get_label_to_hypotheses(version=cfg.taxonomy_version)
    tag_ids = get_tag_ids(version=cfg.taxonomy_version)
    pol_map = get_tag_polarity(version=cfg.taxonomy_version)
    
    def _threshold_for(tag_id: str) -> float:
        pol = pol_map.get(tag_id, "neutral")
        if pol == "positive": return cfg.threshold_pos
        if pol == "negative": return cfg.threshold_neg
        if pol == "neutral": return cfg.threshold_neu
        return cfg.threshold

    all_hyps, hyp_map = [], {}
    for tid in tag_ids:
        for h in label_to_hyps.get(tid, []):
            all_hyps.append(h)
            hyp_map[h] = tid

    clf = pipeline("zero-shot-classification", model=cfg.model_id, device=-1, truncation=True, max_length=int(cfg.max_length))

    texts = pdf[cfg.text_col].fillna("").astype(str).tolist()
    keys = pdf[cfg.key_col].astype(str).tolist()
    bs = int(cfg.batch_size)
    n = len(texts)
    
    out_rows = []
    pbar = tqdm(range(0, n, bs), desc="Zero-shot batches", unit="batch")
    
    for start in pbar:
        batch_texts = texts[start : start + bs]
        batch_keys  = keys[start  : start + bs]

        results = clf(
            batch_texts, candidate_labels=all_hyps, multi_label=True,
            hypothesis_template="This review says the recipe was {}.",
            truncation=True, max_length=int(cfg.max_length), batch_size=bs,
        )
        if isinstance(results, dict): results = [results]

        for rk, res in zip(batch_keys, results):
            tag_scores = {tid: 0.0 for tid in tag_ids}
            for hyp_label, sc in zip(res["labels"], res["scores"]):
                tid = hyp_map.get(hyp_label)
                if tid and sc > tag_scores[tid]:
                    tag_scores[tid] = float(sc)
            
            assigned = [tid for tid, sc in tag_scores.items() if sc >= _threshold_for(tid)]
            out_rows.append({
                cfg.key_col: rk,
                "zs_labels": assigned,
                "zs_scores_json": json.dumps(tag_scores, ensure_ascii=False),
                "zs_num_labels": int(len(assigned)),
                "zs_max_score": float(max(tag_scores.values()) if tag_scores else 0.0),
            })

    return pd.DataFrame(out_rows)

def run_final_label_report(labeled_path: str) -> None:
    pdf = pd.read_parquet(labeled_path)
    print(f"--- Final Labeling Report: {Path(labeled_path).name} ---")
    print(f"Total Labeled Samples: {len(pdf):,}")
    print(f"Avg Labels/Review: {pdf['zs_num_labels'].mean():.2f}")
    
    all_labels = [label for sublist in pdf['zs_labels'].tolist() for label in sublist]
    tag_counts = Counter(all_labels)
    
    print("\nTop Tags (% of dataset):")
    for tag, count in tag_counts.most_common():
        print(f"{tag:.<30} {count:,} ({(count/len(pdf))*100:.2f}%)")

    unlabeled_count = len(pdf[pdf['zs_num_labels'] == 0])
    print(f"\nReviews with 0 labels: {unlabeled_count} ({(unlabeled_count/len(pdf))*100:.2f}%)")