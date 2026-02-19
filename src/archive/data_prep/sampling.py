# src/data_prep/sampling.py

import pandas as pd
from sklearn.utils import resample

def undersample_majority_class(
    df: pd.DataFrame,
    target_col: str,
    ratio: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:

    df = df.copy()
    
    minority_class = df[df[target_col] != df[target_col].value_counts().idxmax()]
    majority_class = df[df[target_col] == df[target_col].value_counts().idxmax()]
    
    target_majority_size = int(len(minority_class) * ratio)
    
    majority_downsampled = resample(
        majority_class,
        replace=False,
        n_samples=min(target_majority_size, len(majority_class)),
        random_state=random_state
    )
    
    dfs = [df for df in [minority_class, majority_downsampled] if df is not None]
    
    balanced_df: pd.DataFrame = (
        pd.concat(dfs, ignore_index=True)
        .sample(frac=1, random_state=random_state)
    )
    
    return balanced_df

def oversample_minority_class(
    df: pd.DataFrame,
    target_col: str,
    ratio: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:

    df = df.copy()
    
    majority_class = df[df[target_col] == df[target_col].value_counts().idxmax()]
    minority_class = df[df[target_col] != df[target_col].value_counts().idxmax()]
    
    target_minority_size = int(len(majority_class) * ratio)
    
    minority_upsampled = resample(
        minority_class,
        replace=True,
        n_samples=target_minority_size,
        random_state=random_state
    )
    
    dfs = [df for df in [majority_class, minority_upsampled] if df is not None]
    
    balanced_df: pd.DataFrame = (
        pd.concat(dfs, ignore_index=True)
        .sample(frac=1, random_state=random_state)
    )
    
    return balanced_df

def ordinal_resample(
    df: pd.DataFrame,
    target_col: str,
    method: str = "balanced",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Resample multi-class ordinal target data.
    """

    df = df.copy()
    classes = sorted(df[target_col].unique())
    counts = df[target_col].value_counts()

    # Choose target size for each class
    if method == "balanced":
        target_size = int(counts.median())
    elif method == "oversample":
        target_size = int(counts.max() * 0.75)
    elif method == "undersample":
        target_size = int(counts.min())
    else:
        raise ValueError(f"Unknown ordinal sampling method: {method}")

    resampled = []

    for c in classes:
        subset = df[df[target_col] == c]

        if len(subset) > target_size:
            # Undersample
            subset = subset.sample(target_size, random_state=random_state)
        elif len(subset) < target_size:
            # Oversample
            subset = subset.sample(target_size, replace=True, random_state=random_state)

        resampled.append(subset)

    df_resampled = pd.concat(resampled, ignore_index=True)
    df_resampled = df_resampled.sample(frac=1, random_state=random_state)  # shuffle

    return df_resampled
