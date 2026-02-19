import pandas as pd
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define TF-IDF parameters for each text column
TFIDF_COLUMNS = {
    "ingredients_clean":{"max_features": 0.9, "ngram_range": (1,2), "stop_words": None},
    "steps_clean":{"max_features": 0.33, "ngram_range": (1,2), "stop_words": None},
    "tags_clean":{"max_features": 1, "ngram_range": (1,1), "stop_words": None},
    "description_clean":{"max_features": 0.10, "ngram_range": (1,2), "stop_words": "english"} # Only use stopwords filtering for description 
}

# Define numeric columns to be scaled
NUMERIC_COLS = [
    "minutes",
    "n_steps",
    "n_ingredients",
    "calories",
    "fat",
    "sugar",
    "sodium",
    "protein",
    "saturated_fat",
    "carbs",
]

def vocab_size(series: pd.Series) -> int:
    """Return a vocabulary size for a given text series."""
    return (
        series
        .str.split()
        .explode()
        .nunique()
    )
 

def get_text_vectorizer(
    column: str,
    max_features: int,
    ngram_range=(1, 2),
    stop_words=None
):
    """Return a dict mapping column to a configured TfidfVectorizer."""
    return {
        column: TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words
        )
    }

def build_text_vectorizers(df: pd.DataFrame):
    """
    Build dictionary of TF-IDF vectorizers using dynamic max_features
    based on vocabulary percentages defined in TFIDF_COLUMNS.
    """
    vectorizers = {}

    for col, params in TFIDF_COLUMNS.items():

        # Compute vocab size for this column in the DF
        vocab = vocab_size(df[col])
        
        # Compute max_features based on % of vocab
        max_feat = max(1, int(vocab * params["max_features"]))

        # Build the vectorizer for this column
        vectorizers.update(
            get_text_vectorizer(
                column=col,
                max_features=max_feat,
                ngram_range=params["ngram_range"],
                stop_words=params["stop_words"]
            )
        )

    return vectorizers

numeric_transformer = Pipeline(
    steps=[
        ("scaler", MinMaxScaler())
    ]
)

def build_feature_transformer(df: pd.DataFrame):
    """
    Build a ColumnTransformer that applies TF-IDF vectorization to text columns and standard scaling to numeric columns.
    """
    text_vectorizers = build_text_vectorizers(df)
    
    transformers = []
    
    # Add text vectorizers
    for col, vectorizer in text_vectorizers.items():
        transformers.append((f"tfidf_{col}", vectorizer, col))
    
    # Add numeric transformer
    transformers.append(("numeric", numeric_transformer, NUMERIC_COLS))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
def prepare_features(
    df: pd.DataFrame,
    transformer: Optional[ColumnTransformer] = None
):
    """
    Fit and transform the DataFrame using the provided ColumnTransformer.
    Returns a DataFrame with transformed features and the fitted transformer.
    """
    
    if transformer is None:
        transformer = build_feature_transformer(df)
        X = transformer.fit_transform(df)
    else:
        X = transformer.transform(df)
    
    return X, transformer

        