import tqdm
from src.models.sklearn_models import ordinal_logistic_regression
from src.models.train import train_classifier
import pandas as pd

def tune_ordinal_logistic_regression(
    df: pd.DataFrame, 
    alphas: list[float] = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0],
    sampling_modes: list[str | None] = [None, "undersample", "oversample", "balanced"],
    target_col: str = "rating",
    verbose: bool = True,
):
    """
    Perform hyperparameter tuning for Ordinal Logistic Regression using grid search.
    """
    
    results = []
    
    total_combinations = len(alphas) * len(sampling_modes)
    
    with tqdm.tqdm(total=total_combinations, desc="Tuning Progress") as pbar:
        for alpha in alphas:
            for sampling_mode in sampling_modes:
                if verbose:
                    print(f"[Tuning] Testing alpha={alpha}, sampling={sampling_mode}")
                
                model = ordinal_logistic_regression(alpha=alpha)
                
                _, _, metrics = train_classifier(
                    model=model,
                    df=df,
                    target_col=target_col,
                    sampling=sampling_mode,    
                    sampling_strategy="ordinal",   
                    sampling_ratio=1.0,
                    verbose=False         
                )
                
                results.append({
                    "params": {
                        "alpha": alpha,
                        "sampling": sampling_mode,
                    },
                    "metrics": metrics
                })
                
                pbar.update(1)

        
    return results