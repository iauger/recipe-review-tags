import numpy as np
from src.data_prep.sampling import oversample_minority_class, undersample_majority_class, ordinal_resample
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from src.features.feature_engineering import prepare_features
from src.models.evaluate import classification_report, ordinal_classification_report, ordinal_classification_report, regression_report


# Abstracted training functions for classification and regression models.
# Classification training pipeline
def train_classifier(
    model,
    df,
    target_col="liked",
    test_size=0.2,
    transformer=None,
    random_state=42,
    sampling: str | None = None,
    sampling_ratio: float = 1.0,
    sampling_strategy: str | None = None,
    verbose: bool = True
):
    """
    Train a classifier using your TF-IDF + scaling pipeline.
    
    Automatically:
    - splits data into training and validation sets
    - fits the feature transformer ONLY on training data
    - trains the model
    - evaluates on validation set
    """
    
    df = df.copy()
    
    # Split X, y
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    X_train_df, X_val_df, y_train, y_val = sklearn_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=np.asarray(y) if sampling is None else None
    )
    
    train_df = X_train_df.copy()
    train_df[target_col] = y_train
    
    if sampling is not None:

        if sampling_strategy == "binary":

            if sampling == "undersample":
                train_df = undersample_majority_class(
                    train_df,
                    target_col=target_col,
                    ratio=sampling_ratio,
                    random_state=random_state
                )
                if verbose: 
                    print(f"[Sampling] Undersampled dataset: {df.shape}")

            elif sampling == "oversample":
                train_df = oversample_minority_class(
                    train_df,
                    target_col=target_col,
                    ratio=sampling_ratio,
                    random_state=random_state
                )
                if verbose:
                    print(f"[Sampling] Oversampled dataset: {df.shape}")

            else:
                raise ValueError(f"Unknown sampling option: {sampling}")


        elif sampling_strategy == "ordinal":

            train_df = ordinal_resample(
                df=train_df,
                target_col=target_col,
                method=sampling,
                random_state=random_state
            )
            if verbose:
                print(f"[Sampling - Ordinal] Resampled dataset: {df.shape}")

        else:
            raise ValueError(f"Sampling strategy '{sampling_strategy}' not implemented.")
    

    # Re-split X, y after sampling
    X_train_df = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int).values
    
    # Feature preprocessing
    # Fit transformer on training data only
    X_train, transformer = prepare_features(X_train_df, transformer=None)

    # Transform validation using same transformer
    X_val, _ = prepare_features(X_val_df, transformer=transformer)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_val)

    # Probability predictions if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_val)
        except:
            pass

    # Metrics
    if target_col == "rating":
        metrics = ordinal_classification_report(y_val, y_pred)
    else:
        metrics = classification_report(y_val, y_pred, y_prob)

    return model, transformer, metrics

# Regression training pipeline
def train_regressor(
    model,
    df,
    target_col="rating",
    test_size=0.2,
    transformer=None,
    random_state=42,
    sampling: str | None = None,
    sampling_ratio: float = 1.0, 
    verbose: bool = True
):
    """
    Train a regression model using your feature engineering pipeline.
    """
    df = df.copy()
    
    if sampling is not None:
        if sampling == "undersample":
            df = undersample_majority_class(
                df,
                target_col=target_col,
                ratio=sampling_ratio,
                random_state=random_state
            )
            if verbose:
                print(f"[Sampling] Undersampled dataset: {df.shape}")

        elif sampling == "oversample":
            df = oversample_minority_class(
                df,
                target_col=target_col,
                ratio=sampling_ratio,
                random_state=random_state
            )
            if verbose:
                print(f"[Sampling] Oversampled dataset: {df.shape}")

        else:
            raise ValueError(f"Unknown sampling option: {sampling}")
        
    # Split X, y
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    X_train_df, X_val_df, y_train, y_val = sklearn_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None
    )

    # Feature preprocessing
    if transformer is None:
        X_train, transformer = prepare_features(X_train_df)
    else:
        X_train = transformer.transform(X_train_df)

    X_val, _ = prepare_features(X_val_df, transformer)

    # Train regressor
    model.fit(X_train, y_train)

    # Predict + evaluate
    y_pred = model.predict(X_val)

    metrics = regression_report(y_val, y_pred)

    return model, transformer, metrics
