# src/spark/models/train_binary.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import Settings, load_settings
from src.spark.features.io import ensure_dirs, read_feature_dataset, write_manifest
from src.spark.labeling.taxonomy import get_tag_ids

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LRSpec:
    max_iter: int = 50
    reg_param: float = 0.01
    elastic_net_param: float = 0.0
    standardization: bool = True
    threshold: Optional[float] = None 

# -------------------------
# Metrics & Weighting Helpers
# -------------------------

def add_inverse_frequency_weights(df: DataFrame, label_col: str, weight_col: str = "weight") -> DataFrame:
    """Balances expected total weight mass across classes."""
    rates = df.select(F.avg(F.col(label_col).cast("double"))).collect()[0][0]
    pos_rate = float(rates) if rates else 0.0
    neg_rate = 1.0 - pos_rate

    if pos_rate <= 0.0 or neg_rate <= 0.0:
        return df.withColumn(weight_col, F.lit(1.0))

    w_pos, w_neg = 0.5 / pos_rate, 0.5 / neg_rate
    return df.withColumn(weight_col, F.when(F.col(label_col) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)))

def _metrics_at_threshold(scored_df: DataFrame, label_col: str, threshold: float) -> dict[str, Any]:
    """Calculates confusion matrix and derived stats (F1, Youden's J)."""
    p1 = vector_to_array(F.col("probability"))[1]
    df = scored_df.select(
        F.col(label_col).cast("int").alias("y"),
        p1.cast("double").alias("p1")
    ).withColumn("yhat", (F.col("p1") >= F.lit(float(threshold))).cast("int"))

    agg = df.agg(
        F.sum(F.when((F.col("y")==1)&(F.col("yhat")==1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("y")==0)&(F.col("yhat")==1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("y")==0)&(F.col("yhat")==0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("y")==1)&(F.col("yhat")==0), 1).otherwise(0)).alias("fn")
    ).collect()[0]

    tp, fp, tn, fn = int(agg["tp"]), int(agg["fp"]), int(agg["tn"]), int(agg["fn"])
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fpr = fp/(tn+fp) if (tn+fp)>0 else 0.0
    
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "f1": (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0,
        "youden_j": rec - fpr
    }

def tune_threshold_on_val(val_pred: DataFrame, label_col: str, objective: str) -> dict[str, Any]:
    """Sweeps thresholds on validation to optimize the chosen objective."""
    val_scored = val_pred.select(label_col, "probability").cache()
    thresholds = [i / 100 for i in range(1, 100, 5)]
    
    best, best_score = {}, -1.0
    for t in thresholds:
        m = _metrics_at_threshold(val_scored, label_col, t)
        if m[objective] > best_score:
            best_score, best = m[objective], m
            
    val_scored.unpersist()
    return {"best_threshold": best["threshold"], "best": best}

# -------------------------
# Orchestration
# -------------------------

def train_all_w2v_only(spark: SparkSession, labels: list[str] = None) -> None:
    """Exclusively validates Word2Vec signal per tag."""
    s = load_settings()
    ensure_dirs(s)
    labels = labels or get_tag_ids()

    df = read_feature_dataset(spark, s)
    train_df = df.filter(F.col("split") == "train").cache()
    val_df   = df.filter(F.col("split") == "val").cache()
    test_df  = df.filter(F.col("split") == "test").cache()
    
    train_count = train_df.count()

    run_summary = {"results": {}}
    lr_spec = LRSpec()

    for lab in labels:
        label_col = f"y_{lab}"
        logger.info("Validating: %s", lab)

        # Standardize on F1 for all labels
        obj = "f1" 
        
        train_w = add_inverse_frequency_weights(train_df, label_col)
        lr = LogisticRegression(labelCol=label_col, featuresCol="features", maxIter=lr_spec.max_iter)
        model = lr.setWeightCol("weight").fit(train_w)

        # Tune based on the unified F1 objective
        tuning = tune_threshold_on_val(model.transform(val_df), label_col, obj)
        
        test_metrics = _metrics_at_threshold(model.transform(test_df), label_col, tuning["best_threshold"])
        run_summary["results"][lab] = {"threshold": tuning["best_threshold"], "test": test_metrics}
        
    summary_path = Path(s.features_run_dir) / "models" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(run_summary, indent=2))
    write_manifest(s, {"models_summary_path": str(summary_path)})