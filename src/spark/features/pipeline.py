# src/features/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    NGram,
    HashingTF,
    IDF,
    SQLTransformer,
)

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

Representation = Literal["w2v", "tfidf", "hybrid"]

@dataclass(frozen=True)
class TextFeatureSpec:
    """
    Minimal config for a text feature extraction pipeline. Can be extended with more complex features and parameters as needed.
    """
    text_col: str
    output_col: str
    
    # Which representation to output as the canonical "features" column
    representation: Representation = "w2v"

    # Column names for optional side outputs (to avoid name collisions)
    tfidf_col: str = "features_tfidf"
    w2v_col: str = "review_embeddings"
    
    # Tokenization
    token_col: str = "tokens"
    tokenizer_pattern: str = r"\w+"
    min_token_length: int = 2
    
    # Pruning
    min_token_count: int = 15 # Minimum review length
    max_token_count: int = 180 # Maximum review length
    
     # Stopwords
    use_default_stopwords: bool = True
    extra_stopwords: tuple[str, ...] = ()
    keep_negations: bool = True  # common for complaint tags
    
    # N-grams
    ngram_n: int = 2
    enable_ngrams: bool = True
    ngram_col: str = "ngrams"
    
    # Token union col
    token_union_col: str = "tokens_all"
    
    # Vectorization
    num_features: int = 262_144  # 2^18, adjust as needed
    binary_tf: bool = False
    min_doc_freq: int = 2 # IDF param
    
    # Intermediate output for debugging/analysis
    keep_intermediate: bool = False
    
    # Whether to include TF-IDF or just vector embeddings
    tfidf_included: bool = True

def build_stopwords(spec: TextFeatureSpec) -> list[str]:
    """
     Build a stop words list for the StopWordsRemover stage. Starts with Spark's built-in English stop words, and can be extended with custom stop words as needed.
     """
    
    stopwords = []
    
    if spec.use_default_stopwords:
        stopwords.extend(StopWordsRemover.loadDefaultStopWords("english"))
    
    if spec.extra_stopwords:
        stopwords.extend(spec.extra_stopwords)
        
    if spec.keep_negations:
        negations = {"no", "nor", "not", "never", "none", "n't"}
        stopwords = [w for w in stopwords if w not in negations]
    
    # Deduplicate and return
    seen = set()
    out = []
    for w in stopwords:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def build_prep_pipeline(spec: TextFeatureSpec) -> Pipeline:
    tok = RegexTokenizer(
        inputCol=spec.text_col,
        outputCol=spec.token_col,
        pattern=spec.tokenizer_pattern,
        minTokenLength=spec.min_token_length,
        toLowercase=True,
        gaps=False,
    )
    
    sw = StopWordsRemover(
        inputCol=spec.token_col,
        outputCol=f"{spec.token_col}_nostop",
        stopWords=build_stopwords(spec),
        caseSensitive=False,
    )

    stages = [tok, sw]

    if spec.enable_ngrams and spec.ngram_n >= 2:
        ng = NGram(
            n=spec.ngram_n,
            inputCol=f"{spec.token_col}_nostop",
            outputCol=spec.ngram_col,
        )
        stages.append(ng)

    return Pipeline(stages=stages)


def build_tfidf_pipeline(spec: TextFeatureSpec) -> Pipeline:
    tf = HashingTF(
        inputCol=spec.token_union_col,
        outputCol=f"{spec.tfidf_col}_tf",
        numFeatures=spec.num_features,
        binary=spec.binary_tf,
    )

    idf = IDF(
        inputCol=f"{spec.tfidf_col}_tf",
        outputCol=spec.tfidf_col,
        minDocFreq=spec.min_doc_freq,
    )

    return Pipeline(stages=[tf, idf])

def add_token_union_column(
    df: DataFrame,
    spec: TextFeatureSpec,
) -> DataFrame:
    """
    Create the array column that feeds vectorization.
    """
    tokens_ns = F.col(f"{spec.token_col}_nostop")

    if spec.enable_ngrams and spec.ngram_n >= 2:
        return df.withColumn(spec.token_union_col, F.concat(tokens_ns, F.col(spec.ngram_col)))
    else:
        return df.withColumn(spec.token_union_col, tokens_ns)


def drop_intermediate_columns(df: DataFrame, spec: TextFeatureSpec) -> DataFrame:
    """
    Optionally drop intermediate token columns to keep final dataset lean.
    """
    if spec.keep_intermediate:
        return df

    cols_to_drop = [
        spec.token_col,
        f"{spec.token_col}_nostop",
        spec.ngram_col if spec.enable_ngrams and spec.ngram_n >= 2 else None,
        spec.token_union_col,
        f"{spec.output_col}_tf",
    ]
    cols_to_drop = [c for c in cols_to_drop if c and c in df.columns]
    return df.drop(*cols_to_drop)