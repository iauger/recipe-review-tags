from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.spark.labeling.taxonomy import get_label_to_hypotheses, get_tag_ids, get_tag_polarity

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class ZeroShotConfig:
    taxonomy_version: str = "v1"
    text_col: str = "review_clean"
    key_col: str = "review_key"

    min_tokens: int = 15
    sample_n: int = 10_000
    sample_seed: int = 42

    model_id: str = "valhalla/distilbart-mnli-12-1"  
    batch_size: int = 16
    max_length: int = 128

     # fallback global threshold if polarity unknown
    threshold: float = 0.55

    # polarity-specific thresholds
    threshold_pos: float = 0.85
    threshold_neg: float = 0.40
    threshold_neu: float = 0.55

    # fallback behavior
    fallback_top_k: int = 1
    fallback_allow_positive: bool = False  # prevent weak “delicious_tasty” from being forced in  


# -----------------------------
# Spark-side helpers
# -----------------------------

def add_review_key(
    df: DataFrame,
    user_col: str = "user_id",
    recipe_col: str = "recipe_id",
    date_col: str = "date",
    text_col: str = "review_clean",
    key_col: str = "review_key",
) -> DataFrame:
    """
    Create a deterministic row key for joining labels back to Spark later.

    We hash a stable concatenation of identifiers + cleaned text.
    """
    required = [user_col, recipe_col, date_col, text_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.withColumn(
        key_col,
        F.sha2(
            F.concat_ws(
                "||",
                F.coalesce(F.col(user_col).cast("string"), F.lit("")),
                F.coalesce(F.col(recipe_col).cast("string"), F.lit("")),
                F.coalesce(F.col(date_col).cast("string"), F.lit("")),
                F.coalesce(F.col(text_col).cast("string"), F.lit("")),
            ),
            256,
        ),
    )


def add_token_count(df: DataFrame, text_col: str = "review_clean", out_col: str = "token_count") -> DataFrame:
    return df.withColumn(out_col, F.size(F.split(F.coalesce(F.col(text_col), F.lit("")), r"\s+")))


def select_informative_subset(df: DataFrame, cfg: ZeroShotConfig) -> DataFrame:
    """
    Filter to informative reviews (min token threshold) and sample a fixed-size subset.
    Assumes cfg.key_col already exists in df (gold_reviews contract).
    """
    if cfg.text_col not in df.columns:
        raise ValueError(f"Expected text column '{cfg.text_col}' not found. Columns: {df.columns}")

    if cfg.key_col not in df.columns:
        raise ValueError(
            f"Expected key column '{cfg.key_col}' not found. "
            f"Run build_gold_reviews() to add it. Columns: {df.columns}"
        )

    df2 = add_token_count(df, text_col=cfg.text_col, out_col="token_count")
    df2 = df2.filter(F.col("token_count") >= F.lit(cfg.min_tokens))

    sampled = (
        df2.orderBy(F.rand(cfg.sample_seed))
           .limit(int(cfg.sample_n))
           .select(cfg.key_col, cfg.text_col, "token_count")
    )
    return sampled


def write_zero_shot_input(df_subset: DataFrame, out_path: str) -> None:
    """
    Write Spark subset to parquet for local (pandas) inference.
    """
    df_subset.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote zero-shot input to %s", out_path)


def read_zero_shot_output_spark(spark: SparkSession, out_path: str) -> DataFrame:
    """
    Read local inference output parquet into Spark.
    Expected columns: review_key, zs_labels, zs_scores_json, zs_num_labels, zs_max_score
    """
    return spark.read.parquet(out_path)


def attach_zero_shot_labels(
    df_base: DataFrame,
    df_labels: DataFrame,
    key_col: str = "review_key",
) -> DataFrame:
    """
    Join zero-shot outputs back onto a Spark dataframe using review_key.
    """
    if key_col not in df_base.columns:
        raise ValueError(f"Key column '{key_col}' missing from df_base. Did you run add_review_key()?")

    if key_col not in df_labels.columns:
        raise ValueError(f"Key column '{key_col}' missing from df_labels. Output schema issue.")

    return df_base.join(df_labels, on=key_col, how="inner")


# -----------------------------
# Local (pandas) inference
# -----------------------------

def _ensure_hf_available() -> None:
    try:
        import transformers  # noqa: F401
    except Exception as e:
        raise ImportError(
            "transformers is required for zero-shot inference. "
            "Install with: pip install transformers torch (and optionally accelerate)."
        ) from e


def _build_premise(text: str) -> str:
    # keep simple; you can  adjust template later
    # TODO: experiment with different templates or prompt formats 
    return text


def run_zero_shot_local(pdf: pd.DataFrame, cfg: ZeroShotConfig) -> pd.DataFrame:
    """
    Optimized + progress bar:
      - Single inference pass (removes duplicated loop)
      - Prebuild hyp->tag map once
      - tqdm progress bar with ETA
      - Optional: uses pipeline(batch_size=...) to help HF handle batching internally
    """
    _ensure_hf_available()
    from transformers import pipeline
    from tqdm.auto import tqdm
    import time
    import json

    required = {cfg.key_col, cfg.text_col}
    missing = required - set(pdf.columns)
    if missing:
        raise ValueError(f"Missing required columns for local inference: {sorted(missing)}")

    # Load taxonomy 
    label_to_hyps: Dict[str, List[str]] = get_label_to_hypotheses(version=cfg.taxonomy_version)
    tag_ids: List[str] = get_tag_ids(version=cfg.taxonomy_version)
    pol_map = get_tag_polarity(version=cfg.taxonomy_version)
    
    def _threshold_for(tag_id: str) -> float:
        pol = pol_map.get(tag_id, "neutral")
        if pol == "positive":
            return cfg.threshold_pos
        if pol == "negative":
            return cfg.threshold_neg
        if pol == "neutral":
            return cfg.threshold_neu
        return cfg.threshold

    # Flatten hypotheses and build hyp->tag lookup for quick mapping during inference
    all_hyps: List[str] = []
    hyp_map: Dict[str, str] = {}
    for tid in tag_ids:
        hyps = label_to_hyps.get(tid, [])
        if not hyps:
            raise ValueError(f"Taxonomy missing hypotheses for tag '{tid}'")
        for h in hyps:
            all_hyps.append(h)
            hyp_map[h] = tid

    logger.info("Zero-shot: %d tags, %d total hypotheses", len(tag_ids), len(all_hyps))

    # ---- HF pipeline ----
    clf = pipeline(
        "zero-shot-classification",
        model=cfg.model_id,
        device=-1,                  # CPU; set to 0 for GPU if available
        truncation=True,
        max_length=int(cfg.max_length),
    )

    # Convert columns once
    texts = pdf[cfg.text_col].fillna("").astype(str).tolist()
    keys = pdf[cfg.key_col].astype(str).tolist()

    bs = int(cfg.batch_size)
    n = len(texts)
    
    out_rows = []
    t0 = time.time()

    pbar = tqdm(range(0, n, bs), desc="Zero-shot batches", unit="batch")
    for start in pbar:
        batch_texts = texts[start : start + bs]
        batch_keys  = keys[start  : start + bs]

        b0 = time.time()
        results = clf(
            batch_texts,
            candidate_labels=all_hyps,
            multi_label=True,
            hypothesis_template="This review says the recipe was {}.",
            truncation=True,
            max_length=int(cfg.max_length),
            batch_size=bs,          # IMPORTANT
        )
        if isinstance(results, dict):
            results = [results]

        for rk, res in zip(batch_keys, results):
            labels = res["labels"]
            scores = res["scores"]

            tag_scores: Dict[str, float] = {tid: 0.0 for tid in tag_ids}
            for hyp_label, sc in zip(labels, scores):
                tid = hyp_map.get(hyp_label)
                if tid is None:
                    continue
                if sc > tag_scores[tid]:
                    tag_scores[tid] = float(sc)
            
            fallback_choice = None
            fallback_choice_score = None
            fallback_candidate_max = None
            
            assigned = [tid for tid, sc in tag_scores.items() if sc >= _threshold_for(tid)]
            candidates = list(tag_scores.items())
            if not cfg.fallback_allow_positive:
                candidates = [(t, s) for (t, s) in candidates if pol_map.get(t) != "positive"]

            fallback_candidate_max = max((s for _, s in candidates), default=0.0)

            used_fallback = False
            if not assigned and cfg.fallback_top_k and cfg.fallback_top_k > 0 and fallback_candidate_max >= cfg.threshold:
                used_fallback = True
                top = sorted(candidates, key=lambda x: x[1], reverse=True)[: cfg.fallback_top_k]
                assigned = [t for t, _ in top]
                fallback_choice, fallback_choice_score = top[0]

            out_rows.append({
                cfg.key_col: rk,
                "zs_labels": assigned,
                "zs_scores_json": json.dumps(tag_scores, ensure_ascii=False),
                "zs_num_labels": int(len(assigned)),
                "zs_max_score": float(max(tag_scores.values()) if tag_scores else 0.0),
                "zs_used_fallback": used_fallback,
                "zs_fallback_choice": fallback_choice,
                "zs_fallback_choice_score": float(fallback_choice_score) if fallback_choice_score is not None else None,
                "zs_fallback_candidate_max": float(fallback_candidate_max),
            })

        b1 = time.time()
        done = min(start + bs, n)
        elapsed = time.time() - t0
        sps = done / elapsed if elapsed > 0 else 0.0
        eta = (n - done) / sps if sps > 0 else float("inf")

        pbar.set_postfix({
            "done": f"{done}/{n}",
            "sec/batch": f"{(b1-b0):.2f}",
            "samp/sec": f"{sps:.3f}",
            "eta_min": f"{eta/60:.1f}" if eta != float("inf") else "inf",
        })

    return pd.DataFrame(out_rows)


def write_zero_shot_output(pdf_out: pd.DataFrame, out_path: str) -> None:
    """
    Write local inference outputs to parquet. Requires pyarrow installed.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.to_parquet(out_path, index=False)
    logger.info("Wrote zero-shot output to %s", out_path)


# -----------------------------
# Convenience end-to-end helpers
# -----------------------------

def build_zero_shot_io_paths(processed_dir: str) -> Tuple[str, str]:
    """
    Standardize where labeling IO lives under processed_dir.
    """
    base = Path(processed_dir) / "labeling" / "zero_shot"
    inp = str(base / "zs_input.parquet")
    out = str(base / "zs_output.parquet")
    return inp, out


def prepare_zero_shot_input_from_gold(
    spark: SparkSession,
    gold_reviews_path: str,
    cfg: ZeroShotConfig,
    processed_dir: str,
) -> str:
    inp_path, _ = build_zero_shot_io_paths(processed_dir)
    df = spark.read.parquet(gold_reviews_path)

    subset = select_informative_subset(df, cfg)
    write_zero_shot_input(subset, inp_path)
    return inp_path


def run_zero_shot_from_input_parquet(
    input_parquet_path: str,
    cfg: ZeroShotConfig,
    processed_dir: str,
) -> str:
    """
    Local function: reads zs_input.parquet -> runs inference -> writes zs_output.parquet.
    Returns output parquet path.
    """
    _, out_path = build_zero_shot_io_paths(processed_dir)
    pdf_in = pd.read_parquet(input_parquet_path)
    pdf_out = run_zero_shot_local(pdf_in, cfg)
    write_zero_shot_output(pdf_out, out_path)
    return out_path


def attach_zero_shot_labels_to_gold(
    spark: SparkSession,
    gold_reviews_path: str,
    processed_dir: str,
    cfg: ZeroShotConfig,
    out_path: Optional[str] = None,
) -> str:
    inp_path, zs_out_path = build_zero_shot_io_paths(processed_dir)

    df_base = spark.read.parquet(gold_reviews_path)

    if cfg.key_col not in df_base.columns:
        raise ValueError(
            f"Expected '{cfg.key_col}' in gold reviews. "
            "Rebuild gold_reviews with build_gold_reviews()"
        )

    df_labels = read_zero_shot_output_spark(spark, zs_out_path)

    df_labeled = attach_zero_shot_labels(df_base, df_labels, key_col=cfg.key_col)

    if out_path is None:
        out_path = str(Path(processed_dir) / "labeling" / "zero_shot" / "labeled_gold_reviews.parquet")

    df_labeled.write.mode("overwrite").parquet(out_path)
    logger.info("Wrote labeled gold reviews to %s", out_path)
    logger.info("Zero-shot input parquet was at %s (for reference)", inp_path)

    return out_path