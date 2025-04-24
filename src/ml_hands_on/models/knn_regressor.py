import numpy as np

from ml_hands_on.base import Model
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.models.knn_classifier import KNNClassifier
from ml_hands_on.metrics.rmse import rmse


class KNNRegressor(Model):
    """
    K-Nearest Neighbors regressor.

    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    distance :  optional
        Function to compute distance between samples. Defaults to Euclidean.

    Attributes
    ----------
    dataset : Dataset
        Training dataset stored after fitting.
    """


    def __init__(self, k: int = 3, distance=None):
        """
        Initializes the K-Nearest Neighbors model.

        Parameters
        ----------
        k : int, default=3
            Number of nearest neighbors to consider.
        distance : callable, optional
            Distance function that takes two arrays and returns a float. Defaults to Euclidean distance.
        """
        super().__init__()
        self.k = k
        self.distance = distance or self.euclidean
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':

        """
           Fit the model using the training dataset.

           Parameters
           ----------
           dataset : Dataset
               The training dataset.

           Returns
           -------
           self : KNNClassifier
               Fitted estimator.
               """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        dataset : Dataset
            Dataset with features to predict.

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels.
        """
        predictions = []
        for x in dataset.X:
            distances = np.array([self.distance(x, x_train) for x_train in self.dataset.X])
            k_indices = distances.argsort()[:self.k]
            k_values = self.dataset.y[k_indices].flatten()
            predictions.append(np.mean(k_values))
        return np.array(predictions).reshape(-1, 1)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Return the RMSE on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            Test data with true labels.

        Returns
        -------
        score : float
            Root Mean Squared Error.
        """
        return rmse(dataset.y, predictions)

    @staticmethod
    def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the Euclidean distance between two points.

        Parameters
        ----------
        x1 : np.ndarray
        x2 : np.ndarray

        Returns
        -------
        distance : float
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))