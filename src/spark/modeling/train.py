# src/spark/models/train_binary.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from collections.abc import Mapping

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import Settings, load_settings
from src.spark.features.io import ensure_dirs, read_feature_dataset, write_manifest, write_metrics  # adjust if needed

logger = logging.getLogger(__name__)


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class LRSpec:
    max_iter: int = 50
    reg_param: float = 0.01
    elastic_net_param: float = 0.0
    standardization: bool = True
    threshold: Optional[float] = None  # if None, Spark uses 0.5


# -------------------------
# Metrics helpers
# -------------------------

def _confusion_counts(df_pred: DataFrame, label_col: str, pred_col: str = "prediction") -> dict[str, int]:
    """
    Compute TP/FP/TN/FN from a prediction dataframe with 0/1 prediction.
    """
    # Ensure ints
    x = df_pred.select(
        F.col(label_col).cast("int").alias("y"),
        F.col(pred_col).cast("int").alias("yhat"),
    )
    agg = (
        x.select(
            F.sum(F.when((F.col("y") == 1) & (F.col("yhat") == 1), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("y") == 0) & (F.col("yhat") == 1), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("y") == 0) & (F.col("yhat") == 0), 1).otherwise(0)).alias("tn"),
            F.sum(F.when((F.col("y") == 1) & (F.col("yhat") == 0), 1).otherwise(0)).alias("fn"),
        )
        .collect()[0]
        .asDict()
    )
    return {k: int(v) for k, v in agg.items()}


def _precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _accuracy(df_pred: DataFrame, label_col: str, pred_col: str = "prediction") -> float:
    return float(
        df_pred.select(
            (F.col(label_col).cast("double") == F.col(pred_col).cast("double")).cast("double").alias("ok")
        ).agg(F.avg("ok")).collect()[0][0]
    )


def _roc_auc(df_pred: DataFrame, label_col: str, raw_pred_col: str = "rawPrediction") -> float:
    ev = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_pred_col, metricName="areaUnderROC")
    return float(ev.evaluate(df_pred))


def _pr_auc(df_pred: DataFrame, label_col: str, raw_pred_col: str = "rawPrediction") -> float:
    ev = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_pred_col, metricName="areaUnderPR")
    return float(ev.evaluate(df_pred))


def _positive_rate(df: DataFrame, label_col: str) -> float:
    return float(df.select(F.avg(F.col(label_col).cast("double"))).collect()[0][0])

def _validate_feature_artifact(df: DataFrame, s) -> dict:
    """
    Ensures the dataset contains a valid vector feature column
    and loads manifest metadata for reproducibility logging.
    """
    if "features" not in df.columns:
        raise ValueError("Feature dataset missing 'features' column.")

    # Validate vector type
    field = [f for f in df.schema.fields if f.name == "features"][0]
    if not isinstance(field.dataType, VectorUDT):
        raise ValueError("'features' column is not a Spark ML VectorUDT.")

    # Load manifest if present
    manifest_path = Path(s.features_manifest_path)
    if not manifest_path.exists():
        logger.warning("Manifest file not found at %s", manifest_path)
        return {}

    manifest = json.loads(manifest_path.read_text())
    return manifest

def _assert_vector_col(df: DataFrame, colname: str) -> None:
    """
    Ensure the given column exists and is a Spark ML VectorUDT.
    """
    if colname not in df.columns:
        raise ValueError(f"Missing required vector column: {colname}")

    field = [f for f in df.schema.fields if f.name == colname][0]
    if not isinstance(field.dataType, VectorUDT):
        raise ValueError(
            f"'{colname}' is not a Spark ML VectorUDT (got {field.dataType})."
        )

FEATURE_MODES = ("tfidf", "embed", "combined")

def _prepare_feature_views(
    df: DataFrame,
    *,
    split_col: str,
    tfidf_col: str = "features",
    embed_col: str = "embeddings",
) -> Mapping[str, Mapping[str, DataFrame | str]]:
    """
    Returns:
      {
        "tfidf":    {"train": ..., "val": ..., "test": ..., "features_col": "features"},
        "embed":    {"train": ..., "val": ..., "test": ..., "features_col": "embeddings"},
        "combined": {"train": ..., "val": ..., "test": ..., "features_col": "features_combined"},
      }
    """
    base = {
        "train": df.filter(F.col(split_col) == F.lit("train")),
        "val":   df.filter(F.col(split_col) == F.lit("val")),
        "test":  df.filter(F.col(split_col) == F.lit("test")),
    }

    out: dict[str, dict[str, DataFrame | str]] = {}

    # 1) TF-IDF only
    out["tfidf"] = {**base, "features_col": tfidf_col}

    # 2) embeddings only
    out["embed"] = {**base, "features_col": embed_col}

    # 3) combined: concatenate vectors
    assembled_col = "features_combined"
    assembler = VectorAssembler(inputCols=[tfidf_col, embed_col], outputCol=assembled_col)

    base_combined = {k: assembler.transform(v) for k, v in base.items()}
    out["combined"] = {**base_combined, "features_col": assembled_col}

    return out


# -------------------------
# Weighting
# -------------------------

def add_inverse_frequency_weights(df: DataFrame, label_col: str, weight_col: str = "weight") -> DataFrame:
    """
    Adds weightCol using inverse-frequency weights:
      w_pos = 0.5 / pos_rate
      w_neg = 0.5 / neg_rate
    so that expected total weight mass is balanced across classes.
    """
    pos_rate = _positive_rate(df, label_col)
    neg_rate = 1.0 - pos_rate

    # Avoid blow-ups if a class is missing
    if pos_rate <= 0.0 or neg_rate <= 0.0:
        logger.warning("Degenerate class rates for %s (pos_rate=%s). Skipping weights.", label_col, pos_rate)
        return df.withColumn(weight_col, F.lit(1.0))

    w_pos = 0.5 / pos_rate
    w_neg = 0.5 / neg_rate

    return df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1, F.lit(float(w_pos))).otherwise(F.lit(float(w_neg)))
    )


# -------------------------
# Train + eval
# -------------------------

def train_lr(
    train_df: DataFrame,
    label_col: str,
    features_col: str,
    *,
    spec: LRSpec,
    weight_col: Optional[str] = None,
) -> LogisticRegressionModel:
    lr = LogisticRegression(
        labelCol=label_col,
        featuresCol=features_col,
        maxIter=spec.max_iter,
        regParam=spec.reg_param,
        elasticNetParam=spec.elastic_net_param,
        standardization=spec.standardization,
    )
    if weight_col:
        lr = lr.setWeightCol(weight_col)
    if spec.threshold is not None:
        lr = lr.setThreshold(spec.threshold)

    return lr.fit(train_df)

def _metrics_at_threshold(
    scored_df: DataFrame,
    *,
    label_col: str,
    threshold: float,
    prob_col: str = "probability",
) -> dict[str, Any]:
    """
    Compute confusion counts + derived metrics using P(y=1) >= threshold.

    Expects Spark ML `probability` vector column.
    """
    p1 = vector_to_array(F.col(prob_col))[1]
    df = scored_df.select(
        F.col(label_col).cast("int").alias("y"),
        p1.cast("double").alias("p1"),
    ).withColumn("yhat", (F.col("p1") >= F.lit(float(threshold))).cast("int"))

    agg = df.agg(
        F.sum(F.when((F.col("y") == 1) & (F.col("yhat") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("y") == 0) & (F.col("yhat") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("y") == 0) & (F.col("yhat") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("y") == 1) & (F.col("yhat") == 0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]

    tp = int(agg["tp"])
    fp = int(agg["fp"])
    tn = int(agg["tn"])
    fn = int(agg["fn"])

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)         # TPR
    specificity = 0.0 if (tn + fp) == 0 else tn / (tn + fp)    # TNR
    fpr = 0.0 if (tn + fp) == 0 else fp / (tn + fp)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    acc = 0.0 if (tp + tn + fp + fn) == 0 else (tp + tn) / (tp + tn + fp + fn)
    bal_acc = 0.5 * (recall + specificity)
    youden_j = recall - fpr

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "balanced_accuracy": float(bal_acc),
        "youden_j": float(youden_j),
    }


def tune_threshold_on_val(
    val_pred: DataFrame,
    *,
    label_col: str,
    objective: str = "f1",
    thresholds: list[float] | None = None,
    prob_col: str = "probability",
) -> dict[str, Any]:
    """
    Sweep thresholds on validation predictions and choose best by objective.

    Returns:
      {
        "objective": "f1",
        "best_threshold": 0.17,
        "best": {...metrics...},
        "curve": [ {...}, {...}, ... ]   # optional but useful
      }
    """
    if thresholds is None:
        thresholds = [i / 100 for i in range(1, 100, 5)]

    if objective not in {"f1", "precision", "recall", "accuracy", "specificity", "balanced_accuracy", "youden_j"}:
        raise ValueError(f"Unsupported objective: {objective}")

    # Cache scored val once; reuse for each threshold aggregation
    val_scored = val_pred.select(label_col, prob_col).cache()
    _ = val_scored.count()  # materialize cache

    curve: list[dict[str, Any]] = []
    best = None
    best_score = -1.0

    for t in thresholds:
        m = _metrics_at_threshold(val_scored, label_col=label_col, threshold=t, prob_col=prob_col)
        score = float(m[objective])
        curve.append(m)
        if score > best_score:
            best_score = score
            best = m

    val_scored.unpersist()

    return {
        "objective": objective,
        "best_threshold": float(best["threshold"]) if best else 0.5,
        "best": best if best else {},
        "curve": curve,  # keep for analysis; remove if you want smaller JSON
    }

def _balanced_accuracy_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    tpr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    tnr = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
    return 0.5 * (tpr + tnr)

def evaluate_binary(df_pred: DataFrame, label_col: str) -> dict[str, Any]:
    counts = _confusion_counts(df_pred, label_col)
    prf = _precision_recall_f1_from_counts(counts["tp"], counts["fp"], counts["fn"])
    out: dict[str, Any] = {
        "accuracy": _accuracy(df_pred, label_col),
        "balanced_accuracy": _balanced_accuracy_from_counts(counts["tp"], counts["fp"], counts["tn"], counts["fn"]),
        "roc_auc": _roc_auc(df_pred, label_col),
        "pr_auc": _pr_auc(df_pred, label_col),
        **counts,
        **prf,
    }
    return out


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# -------------------------
# Orchestration
# -------------------------

def train_all(
    spark: SparkSession,
    *,
    labels: Optional[list[str]] = None,
    features_col: str = "features",
    split_col: str = "split",
    run_manifest: bool = True,
    settings: Optional[Settings] = None,
) -> None:
    """
    Trains baseline + weighted LR for each label in `labels`.

    Models + metrics are saved under:
      <features_run_dir>/models/<label>/<mode>/
    """
    if labels is None:
        labels = ["delicious_tasty", "ingredient_issue", "bland_lacks_flavor"]

    if settings is None:
        s = load_settings()
    else:
        s = settings
        
    ensure_dirs(s)

    df = read_feature_dataset(spark, s)
    manifest = _validate_feature_artifact(df, s)

    # guards
    required = {split_col}
    for lab in labels:
        required.add(f"y_{lab}")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature dataset missing required columns: {missing}")

    # Explicitly validate vector columns
    _assert_vector_col(df, "features")
    _assert_vector_col(df, "embeddings")

    views = _prepare_feature_views(df, split_col=split_col, tfidf_col="features", embed_col="embeddings")

    base_models_dir = Path(s.features_run_dir) / "models"
    base_models_dir.mkdir(parents=True, exist_ok=True)

    lr_spec = LRSpec()

    run_summary: dict[str, Any] = {
        "features_run_id": s.features_run_id,
        "features_run_dir": s.features_run_dir,
        "lr_spec": asdict(lr_spec),
        "labels": labels,
        "feature_manifest": manifest, 
        "results": {},
    }

    for lab in labels:
        label_col = f"y_{lab}"
        logger.info("=== Training label: %s (%s) ===", lab, label_col)

        run_summary["results"][lab] = {}
        objective = "youden_j" if lab == "delicious_tasty" else "f1"

        for mode in FEATURE_MODES:
            feat_train: DataFrame = views[mode]["train"]  # type: ignore
            feat_val: DataFrame = views[mode]["val"]  # type: ignore
            feat_test: DataFrame = views[mode]["test"]  # type: ignore
            feat_col: str = views[mode]["features_col"]  # type: ignore

            # -------------------------
            # Mode 1: baseline (no weights)
            # -------------------------
            baseline_dir = base_models_dir / lab / mode / "baseline"
            model_path = str(baseline_dir / "model")

            model = train_lr(feat_train, label_col, feat_col, spec=lr_spec, weight_col=None)
            model.write().overwrite().save(model_path)

            val_pred = model.transform(feat_val)
            test_pred = model.transform(feat_test)
            
            # --- threshold tuning on validation probabilities ---
            tuning = tune_threshold_on_val(
                val_pred,
                label_col=label_col,
                objective=objective,
                thresholds=[i / 100 for i in range(1, 100, 5)],
                prob_col="probability",
            )

            val_at_best = _metrics_at_threshold(
                val_pred.select(label_col, "probability"),
                label_col=label_col,
                threshold=float(tuning["best_threshold"]),
                prob_col="probability",
            )

            test_at_best = _metrics_at_threshold(
                test_pred.select(label_col, "probability"),
                label_col=label_col,
                threshold=float(tuning["best_threshold"]),
                prob_col="probability",
            )

            baseline_metrics = {
                "feature_mode": mode,
                "mode": "baseline",
                "label": lab,
                "label_col": label_col,
                "features_col": feat_col,
                "train_pos_rate": _positive_rate(feat_train, label_col),
                "val": evaluate_binary(val_pred, label_col),
                "val_at_best": val_at_best,
                "test": evaluate_binary(test_pred, label_col),
                "test_at_best": test_at_best,
            }
            _save_json(baseline_dir / "metrics.json", baseline_metrics)
            run_summary["results"][lab].setdefault(mode, {})["baseline"] = baseline_metrics

            # -------------------------
            # Mode 2: weighted (inverse frequency)
            # -------------------------
            weighted_dir = base_models_dir / lab / mode / "weighted_invfreq"
            model_path = str(weighted_dir / "model")

            train_w = add_inverse_frequency_weights(feat_train, label_col, weight_col="weight")
            model_w = train_lr(train_w, label_col, feat_col, spec=lr_spec, weight_col="weight")
            model_w.write().overwrite().save(model_path)

            val_pred_w = model_w.transform(feat_val)
            test_pred_w = model_w.transform(feat_test)

            weighted_metrics = {
                "feature_mode": mode,
                "mode": "weighted_invfreq",
                "label": lab,
                "label_col": label_col,
                "features_col": feat_col,
                "train_pos_rate": _positive_rate(feat_train, label_col),
                "val": evaluate_binary(val_pred_w, label_col),
                "val_at_best": _metrics_at_threshold(
                    val_pred_w.select(label_col, "probability"),
                    label_col=label_col,
                    threshold=float(tuning["best_threshold"]),
                    prob_col="probability",
                ),
                "test": evaluate_binary(test_pred_w, label_col),
                "test_at_best": _metrics_at_threshold(
                    test_pred_w.select(label_col, "probability"),
                    label_col=label_col,
                    threshold=float(tuning["best_threshold"]),
                    prob_col="probability",
                ),
            }
            _save_json(weighted_dir / "metrics.json", weighted_metrics)
            run_summary["results"][lab][mode]["weighted_invfreq"] = weighted_metrics

    # Write run-level summary to run dir
    _save_json(Path(s.features_run_dir) / "models" / "summary.json", run_summary)

    # Optional: also write into features metrics (append-style)
    if run_manifest:
        # keep it lightweight in manifest (point to summary file)
        write_manifest(s, {"models_summary_path": str(Path(s.features_run_dir) / "models" / "summary.json")})

    logger.info("Saved model artifacts under %s", str(base_models_dir))
