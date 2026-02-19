from src.models.train import train_classifier
import pandas as pd
import tqdm

def tune_model(
    model,
    param_grid: dict,
    df: pd.DataFrame,
    target_col="liked",
    sampling : str | None = None,
    sampling_ratio: float = 1.0,
    sampling_strategy: str = "binary",
    verbose: bool = True,
):
    """
    Perform hyperparameter tuning for a classification model using grid search.
    """
    results = []
    keys = list(param_grid.keys())
    
    def param_combinations(idx, current):
        """Generate all combinations of parameters from the param_grid."""
        if idx == len(keys):
            yield current.copy()
            return
        
        key = keys[idx]
        for value in param_grid[key]:
            current[key] = value
            yield from param_combinations(idx + 1, current)
            
    iterables = list(param_grid.values())
    total_combinations = pd.MultiIndex.from_product(iterables).size

    for params in tqdm.tqdm(param_combinations(0, {}), total=total_combinations):
        if verbose:
            print(f"[Tuning] Testing parameters: {params}")
        
        model_instance = model(**params)
            
        _, _, metrics = train_classifier(
            model=model_instance,
            df=df,
            target_col=target_col,
            sampling=sampling,
            sampling_ratio=sampling_ratio,
            sampling_strategy=sampling_strategy,
            verbose=verbose
        )
        
        results.append({
            "params": params,
            "metrics": metrics
        })
        
    return results

def results_to_df(results: list[dict], metric_name: str) -> pd.DataFrame:
    """
    Convert tuning results to a DataFrame for easier analysis.
    """
    records = []
    for result in results:
        row = {
            **result["params"],
            **{k: (str(v) if k == "confusion_matrix" else v) for k, v in result["metrics"].items()}
        }
        records.append(row)
    
    df = pd.DataFrame(records)
    
    if metric_name in df.columns:
        df = df.sort_values(by=metric_name, ascending=False).reset_index(drop=True)
    
    return df