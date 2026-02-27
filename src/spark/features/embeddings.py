# src/spark/features/embeddings.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word2VecSpec:
    input_col: str = "tokens_all"     
    output_col: str = "embeddings"    
    vector_size: int = 128
    window_size: int = 5
    min_count: int = 2
    max_iter: int = 10
    seed: int = 42


def fit_word2vec(train_df: DataFrame, *, spec: Word2VecSpec) -> Word2VecModel:
    """
    Fit a Spark Word2Vec model on the training split only.
    Expects input_col to be array<string>.
    """
    w2v = Word2Vec(
        inputCol=spec.input_col,
        outputCol=spec.output_col,
        vectorSize=spec.vector_size,
        windowSize=spec.window_size,
        minCount=spec.min_count,
        maxIter=spec.max_iter,
        seed=spec.seed,
    )

    # Basic guard: drop null/empty token rows for training
    train_tokens = train_df.where(
        F.col(spec.input_col).isNotNull() & (F.size(F.col(spec.input_col)) > 0)
    )

    logger.info(
        "Fitting Word2Vec: input=%s output=%s dim=%d window=%d minCount=%d iters=%d",
        spec.input_col, spec.output_col, spec.vector_size, spec.window_size, spec.min_count, spec.max_iter
    )
    return w2v.fit(train_tokens)


# src/spark/features/embeddings.py

def add_word2vec_embeddings(df: DataFrame, *, model: Word2VecModel, spec: Word2VecSpec) -> DataFrame:
    out = spec.output_col
    model_out = model.getOutputCol()
    if model_out != out:
        raise ValueError(f"Word2Vec model outputCol={model_out} does not match spec.output_col={out}")

    if out in df.columns:
        df = df.drop(out)

    return model.transform(df)