import numpy as np
from src.models.gradient_descent import BaseOptimizer

class LinearRegressionGD(BaseOptimizer):
    """
    Linear Regression trained using Gradient Descent.
    """

    # Gradient of MSE
    def gradient(self, X, y, w):
        """
        Compute gradient of MSE loss
        """
        n = X.shape[0]

        # Compute prediction
        y_pred = X @ w

        # Error term
        error = y_pred - y

        # Gradient
        grad = (X.T @ error) / n

        return grad

    # Prediction
    def predict(self, X):
        """
        Predictions = X @ w
        """
        # Add bias to X
        Xb = self.add_bias(X)
        
        if self.w is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        
        return Xb @ self.w


    # Loss function (MSE)
    def loss(self, X, y, w):
        """
        Mean Square Error Loss.
        """
        n = X.shape[0]
        y_pred = X @ w
        return np.sum((y - y_pred)**2) / (2 * n)

    # Fit model
    def fit(self, X, y):
        """
        Fit model using gradient descent.
        Adds bias column automatically.
        """
        Xb = self.add_bias(X)
        self.gradient_descent(Xb, y)
        return self
