import numpy as np
from ml_hands_on.data.dataset import Dataset
from collections import Counter
from ml_hands_on.metrics.accuracy import accuracy


class KNNClassifier:
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    distance : optional
        Function to compute distance between samples.

    Attributes
    ----------
    dataset : Dataset
        Training dataset stored after fitting.Defaults to Euclidean.
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


        self.k = k
        self.distance = distance or self.euclidean
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
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

    def predict(self, dataset: Dataset) -> np.ndarray:
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
            k_labels = self.dataset.y[k_indices].flatten()
            prediction = Counter(k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions).reshape(-1, 1)

    def score(self, dataset: Dataset) -> float:
        """
             Return the accuracy on the given test data and labels.

             Parameters
             ----------
             dataset : Dataset
                 Test data with true labels.

             Returns
             -------
             score : float
                 Classification accuracy.
             """

        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)


