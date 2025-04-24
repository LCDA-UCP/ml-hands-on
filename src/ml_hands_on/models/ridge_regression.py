from ml_hands_on.base import Model
from ml_hands_on.metrics.mse import mse
from ml_hands_on.data.dataset import Dataset

import numpy as np

# Should inherit from Model
class RidgeRegression(Model):
    """
    Ridge Regression model using Gradient Descent and L2 Regularization.

    Parameters
    ----------
    l2_penalty : float
        Regularization parameter (λ).
    alpha : float
        Learning rate.
    max_iter : int
        Maximum number of training iterations.
    patience : int
        Maximum iterations allowed without improvement in cost.
    scale : bool
        Whether to standardize features.
    """

    def __init__(self, l2_penalty=1.0, alpha=0.01, max_iter=1000, patience=10, scale=True):
        super().__init__()
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset):
        """
        Trains the model using Gradient Descent.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing features (X) and targets (y).
        """

        X = dataset.X
        y = dataset.y
        m, n = X.shape

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        self.theta = np.zeros(n)
        self.theta_zero = 0
        best_cost = float('inf')
        no_improve_counter = 0

        for iteration in range(self.max_iter):
            y_pred = X.dot(self.theta) + self.theta_zero
            error = y_pred - y

            grad_theta = (1/m) * X.T.dot(error)
            grad_theta_zero = (1/m) * np.sum(error)

            reg = (1 - self.alpha * self.l2_penalty / m)
            self.theta = self.theta * reg - self.alpha * grad_theta
            self.theta_zero -= self.alpha * grad_theta_zero

            cost = (1/(2*m)) * np.sum(error ** 2) + (self.l2_penalty/(2*m)) * np.sum(self.theta ** 2)
            self.cost_history[iteration] = cost

            if cost < best_cost:
                best_cost = cost
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= self.patience:
                break

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts target values using the learned parameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the feature matrix (X).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        X = dataset.X
        if self.scale and self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std
        return X.dot(self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the Mean Squared Error between true and predicted values.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the true values (y) and features (X).

        Returns
        -------
        float
            Mean Squared Error.
        """
        return mse(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        """
        Computes the total cost (MSE + L2 regularization).

        Parameters
        ----------
        dataset : Dataset
            Dataset containing features (X) and targets (y).

        Returns
        -------
        float
            Total cost.
        """
        X = dataset.X
        y = dataset.y
        m = len(y)

        if self.scale and self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std

        y_pred = X.dot(self.theta) + self.theta_zero
        mse_value = (1 / (2 * m)) * np.sum((y - y_pred) ** 2)
        l2_term = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)

        return mse_value + l2_term
