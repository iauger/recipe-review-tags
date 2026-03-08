# src/spark/modeling/postprocessing.py
import logging
from pyspark.sql import DataFrame, functions as F

logger = logging.getLogger(__name__)

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