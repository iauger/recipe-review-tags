import numpy as np
from src.models.gradient_descent import BaseOptimizer


class LogisticRegressionGD(BaseOptimizer):
    """
    Logistic Regression trained using Gradient Descent.
    """

    # Sigmoid function
    def sigmoid(self, z):
        # Numerically stable sigmoid
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

    # Gradient function
    def gradient(self, X, y, w):
        """
        Gradient of binary cross-entropy with L2 regularization.
        """
        n = X.shape[0]

        # Compute prediction probabilities
        y_pred = self.sigmoid(X @ w)

        # Error term
        error = y_pred - y

        # Gradient
        grad = (X.T @ error) / n

        return grad

    # Loss function (binary cross-entropy)
    def loss(self, X, y, w):
        """
        Binary Cross-Entropy loss.
        """
        p = self.sigmoid(X @ w)

        # Avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)

        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # Predict probabilities
    def predict_proba(self, X):
        Xb = self.add_bias(X)

        if self.w is None:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        return self.sigmoid(Xb @ self.w)

    # Predict binary labels (0/1)
    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    # Fit model
    def fit(self, X, y):
        """
        Fit logistic regression using gradient descent.
        Adds bias column automatically.
        """
        Xb = self.add_bias(X)
        self.gradient_descent(Xb, y)
        return self
