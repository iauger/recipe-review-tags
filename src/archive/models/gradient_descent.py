import numpy as np
from scipy.sparse import issparse, hstack as sparse_hstack
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Abstract parent class for gradient-based optimizers.
    Child classes must implement:
        - gradient()
        - predict()
        - (optional) loss()
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6, lmbda=0.0, verbose=False):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.lmbda = lmbda
        self.verbose = verbose
        self.w = None
        self.loss_history = []


    # Abstract methods to be implemented by child classes
    @abstractmethod
    def gradient(self, X, y, w):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    # Optional — child can override
    def loss(self, X, y, w):
        return None

    # Helper methods
    # Bias-handling
    def add_bias(self, X):
        n = X.shape[0]
        if issparse(X):
            from scipy.sparse import csr_matrix
            bias = csr_matrix(np.ones((n, 1)))
            return sparse_hstack([bias, X])
        else:
            bias = np.ones((n, 1))
            return np.hstack([bias, X])

    # Weight initialization
    def init_weights(self, n_features):
        return np.zeros(n_features)


    # Gradient Descent Loop
    def gradient_descent(self, X, y):
        _, n_features = X.shape
        w = self.init_weights(n_features)

        for i in range(self.max_iter):

            grad = self.gradient(X, y, w)

            if grad is None:
                raise ValueError("gradient() returned None")

            w_new = w - self.lr * grad

            # Convergence check
            if np.linalg.norm(w_new - w) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}")
                w = w_new
                break

            w = w_new

            # Loss tracking (optional)
            if self.loss is not None:
                val = self.loss(X, y, w)
                if val is not None:
                    self.loss_history.append(val)

            if self.verbose and i % 100 == 0:
                print(f"Iter {i}, |grad|={np.linalg.norm(grad):.6f}")

        self.w = w
        return w
