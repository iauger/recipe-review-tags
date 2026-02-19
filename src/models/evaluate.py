import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, cohen_kappa_score
import contextlib
import io
from surprise import accuracy


def evaluate_surprise_model(algo, trainset, testset):
    """Fit a Surprise model and return RMSE/MAE scores without printing."""
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Suppress printed output from Surprise
    with contextlib.redirect_stdout(io.StringIO()):
        rmse_val = accuracy.rmse(predictions, verbose=False)
        mae_val = accuracy.mae(predictions, verbose=False)

    return {"rmse": rmse_val, "mae": mae_val}

def classification_report(y_true, y_pred, y_prob=None, zero_division=0):
    """
    Generate a classification report including accuracy, precision, recall, F1-score, and ROC-AUC (if probabilities provided).
    """
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=zero_division),
        'recall': recall_score(y_true, y_pred, zero_division=zero_division),
        'f1_score': f1_score(y_true, y_pred, zero_division=zero_division)
    }
    
    if y_prob is not None:
        try:
            # If 2D probability matrix, take positive class prob
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_1d = y_prob[:, 1]
            else:
                y_prob_1d = y_prob

            report["roc_auc"] = roc_auc_score(y_true, y_prob_1d)

        except Exception as e:
            report["roc_auc"] = None
    else:
        report["roc_auc"] = None
    
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return report

def ordinal_classification_report(y_true, y_pred, zero_division=0):
    """
    Generate an ordinal classification report including accuracy, precision, recall, F1-score, and Cohen's Kappa.
    """
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=zero_division),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=zero_division),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=zero_division),
        'quadratic_kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic')
    }
    
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return report

def regression_report(y_true, y_pred):
    """
    Generate a regression report including MSE, MAE, RMSE, and R^2 score.
    """
    report = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return report