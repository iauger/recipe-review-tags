# src/spark/features/nlp_pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, Word2Vec, Word2VecModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# --- SPECS ---
@dataclass(frozen=True)
class TextFeatureSpec:
    text_col: str
    output_col: str
    token_col: str = "tokens"
    tokenizer_pattern: str = r"\w+"
    min_token_length: int = 3
    min_token_count: int = 12 
    max_token_count: int = 120 
    use_default_stopwords: bool = True
    extra_stopwords: tuple[str, ...] = tuple()
    keep_negations: bool = False  
    ngram_n: int = 3
    enable_ngrams: bool = True
    ngram_col: str = "ngrams"
    token_union_col: str = "tokens_all"
    keep_intermediate: bool = True

@dataclass(frozen=True)
class Word2VecSpec:
    input_col: str = "tokens_all"     
    output_col: str = "embeddings"    
    vector_size: int = 128
    window_size: int = 5
    min_count: int = 10
    max_iter: int = 15
    seed: int = 42

# --- PIPELINE BUILDERS ---
def build_stopwords(spec: TextFeatureSpec) -> list[str]:
    stopwords = []
    if spec.use_default_stopwords:
        stopwords.extend(StopWordsRemover.loadDefaultStopWords("english"))
    if spec.extra_stopwords:
        stopwords.extend(spec.extra_stopwords)
    if spec.keep_negations:
        negations = {"no", "nor", "not", "never", "none", "n't"}
        stopwords = [w for w in stopwords if w not in negations]
    return list(set(stopwords))

def build_prep_pipeline(spec: TextFeatureSpec) -> Pipeline:
    tok = RegexTokenizer(inputCol=spec.text_col, outputCol=spec.token_col, pattern=spec.tokenizer_pattern, minTokenLength=spec.min_token_length, toLowercase=True, gaps=False)
    sw = StopWordsRemover(inputCol=spec.token_col, outputCol=f"{spec.token_col}_nostop", stopWords=build_stopwords(spec), caseSensitive=False)
    stages = [tok, sw]
    if spec.enable_ngrams and spec.ngram_n >= 2:
        stages.append(NGram(n=spec.ngram_n, inputCol=f"{spec.token_col}_nostop", outputCol=spec.ngram_col))
    return Pipeline(stages=stages)

def add_token_union_column(df: DataFrame, spec: TextFeatureSpec) -> DataFrame:
    tokens_ns = F.col(f"{spec.token_col}_nostop")
    if spec.enable_ngrams and spec.ngram_n >= 2:
        return df.withColumn(spec.token_union_col, F.concat(tokens_ns, F.col(spec.ngram_col)))
    return df.withColumn(spec.token_union_col, tokens_ns)

def drop_intermediate_columns(df: DataFrame, spec: TextFeatureSpec) -> DataFrame:
    if spec.keep_intermediate: return df
    cols_to_drop = [spec.token_col, f"{spec.token_col}_nostop", spec.ngram_col if spec.enable_ngrams else None, spec.token_union_col]
    return df.drop(*[c for c in cols_to_drop if c and c in df.columns])

# --- EMBEDDING LOGIC ---
def fit_word2vec(train_df: DataFrame, *, spec: Word2VecSpec) -> Word2VecModel:
    w2v = Word2Vec(inputCol=spec.input_col, outputCol=spec.output_col, vectorSize=spec.vector_size, windowSize=spec.window_size, minCount=spec.min_count, maxIter=spec.max_iter, seed=datetime.now().microsecond)
    train_tokens = train_df.where(F.col(spec.input_col).isNotNull() & (F.size(F.col(spec.input_col)) > 0))
    logger.info("Fitting Word2Vec: dim=%d window=%d minCount=%d", spec.vector_size, spec.window_size, spec.min_count)
    return w2v.fit(train_tokens)

def add_word2vec_embeddings(df: DataFrame, *, model: Word2VecModel, spec: Word2VecSpec) -> DataFrame:
    if spec.output_col in df.columns: df = df.drop(spec.output_col)
    return model.transform(df)