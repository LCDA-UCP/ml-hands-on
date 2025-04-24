import numpy as np
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.metrics.accuracy import accuracy
from ml_hands_on.base.model import Model


class Perceptron(Model):
    """
    Perceptron model for binary classification using the Perceptron learning rule.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of training iterations (epochs).
    learning_rate : float, default=0.01
        Learning rate for weight updates.
    """

    def __init__(self, max_iter: int = 1000, learning_rate: float = 0.01):
        """
        Initializes the Perceptron model.

        Parameters
        ----------
        max_iter : int, default=1000
            Maximum number of iterations (epochs) for training.
        learning_rate : float, default=0.01
            Step size used for updating weights during training.
        """
        super().__init__()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _fit(self, dataset: Dataset) -> None:
        """
        Trains the Perceptron model using the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the feature matrix (X) and binary labels (y).
        """
        X = dataset.X
        y = dataset.y.ravel()
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("The Dataset must contain at least one sample.")
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.max_iter):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0

                error = y[i] - y_pred
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts class labels for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the feature matrix (X).

        Returns
        -------
        np.ndarray
            Array of predicted class labels (0 or 1).
        """
        X = dataset.X
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the classification accuracy on a labeled dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset object containing the true labels (y).
        predictions : np.ndarray
            Predicted class labels (0 or 1).

        Returns
        -------
        float
            Accuracy of the model on the given dataset.
        """
        return accuracy(dataset.y, predictions)
