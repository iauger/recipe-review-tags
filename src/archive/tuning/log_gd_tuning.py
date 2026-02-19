from src.models.logistic_regression import LogisticRegressionGD
from src.models.train import train_classifier
import pandas as pd
from tqdm import tqdm

def tune_logistic_gd(
    df: pd.DataFrame,
    target_col: str = "liked",
    lrs: list[float] = [0.001, 0.01, 0.05, 0.1],
    lambdas: list[float] = [0.0, 0.01, 0.1, 0.5],
    max_iters: list[int] = [100, 200, 300],
    tolerances: list[float] = [1e-4, 1e-5, 1e-6],
    sampling: str | None = "oversample",
    sampling_ratio: float = 1.0,
    sampling_strategy: str = "binary",
    verbose: bool = True,
):

    results = []
    
    total_combinations = len(lrs) * len(lambdas) * len(max_iters) * len(tolerances)
    
    with tqdm(total=total_combinations, desc="Tuning Progress") as pbar:
        for lr in lrs:
            for lmbda in lambdas:
                for max_iter in max_iters:
                    for tol in tolerances:
                        if verbose:
                            print(f"[Tuning] Testing lr={lr}, lmbda={lmbda}, max_iter={max_iter}, tol={tol}")
                        pbar.update(1)
                    
                    model = LogisticRegressionGD(
                        lr=lr,
                        lmbda=lmbda,
                        max_iter=max_iter,
                        tol=tol,
                        verbose=False
                    )
                    
                    _, _, metrics = train_classifier(
                        model=model,
                        df=df,
                        target_col=target_col,
                        sampling=sampling,
                        sampling_ratio=sampling_ratio,
                        sampling_strategy=sampling_strategy,
                        verbose=False
                    )
                    
                    results.append({
                        "params": {
                            "lr": lr,
                            "lmbda": lmbda,
                            "max_iter": max_iter,
                            "tol": tol
                        },
                        "metrics": metrics
                    })

    return results