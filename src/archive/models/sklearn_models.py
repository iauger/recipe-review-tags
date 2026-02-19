from typing import Literal
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from mord import LogisticIT

# Classification Models
def logistic_regression_balanced(
    C: float = 1.0,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    solver: Literal["liblinear", "newton-cg", "lbfgs", "sag", "saga"] = "liblinear",
    random_state: int = 42,
    max_iter: int = 1000,
    class_weight: str = 'balanced'
):
    """ Logistic Regression with class balancing """
    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight
    )

def linear_svm_balanced(
    C: float = 1.0,
    random_state: int = 42,
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge",
    max_iter: int = 1000,
    class_weight: str = 'balanced',
    calibration_cv: int = 3
):
    """ Linear SVM with class balancing and probability calibration """
    svc = LinearSVC(
        C=C,
        random_state=random_state,
        loss=loss,
        max_iter=max_iter,
        class_weight=class_weight
    )
    # Calibrate SVM to get probability estimates
    calibrated_svc = CalibratedClassifierCV(svc, cv=calibration_cv)
    return calibrated_svc

def naive_bayes_model(alpha: float = 1.0):
    """ Multinomial Naive Bayes classifier """
    return MultinomialNB(alpha=alpha)

def ordinal_logistic_regression(alpha: float = 1.0):
    """
    Ordinal Logistic Regression (Proportional Odds Model).
    """
    return LogisticIT(alpha=alpha)

def voting_ensemble(
    lr_params: dict | None = None,
    svm_params: dict | None = None,
    nb_params: dict | None = None,
    voting: Literal["hard", "soft"] = "soft"
):
    """ Ensemble of Logistic Regression, Linear SVM, Naive Bayes, and Ordinal Logistic Regression """
    # Create individual models with provided parameters
    lr_params = lr_params or {}
    svm_params = svm_params or {}
    nb_params = nb_params or {}
    
    lr_model = logistic_regression_balanced(**lr_params)
    svm_model = linear_svm_balanced(**svm_params)
    nb_model = naive_bayes_model(**nb_params)

    ensemble = VotingClassifier(
        estimators=[
            ('logistic_regression', lr_model),
            ('linear_svm', svm_model),
            ('naive_bayes', nb_model)
        ],
        voting=voting
    )

    return ensemble

# Regression Models
def linear_regression():
    """ Ordinary least squares regression """
    return LinearRegression()

def ridge_regression(alpha: float = 1.0, random_state: int = 42):
    """ Ridge regression for high-dimensional sparse features """
    return Ridge(alpha=alpha, random_state=random_state)

def lasso_regression(alpha: float = 0.001, max_iter: int = 5000):
    """ Lasso regression for feature selection """
    return Lasso(alpha=alpha, max_iter=max_iter)

def elasticnet_regression(alpha: float = 0.001, l1_ratio: float = 0.5, max_iter: int = 5000):
    """ Combined L1 and L2 penalty """
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

def linear_svr(epsilon: float = 0.1, C: float = 1.0):
    """ Support Vector Regression with a linear kernel """
    return LinearSVR(epsilon=epsilon, C=C, max_iter=5000)