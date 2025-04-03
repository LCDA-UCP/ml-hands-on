import numpy as np
from ml_hands_on.data import Dataset
from ml_hands_on.base.model import Model
from ml_hands_on.base.transformer import Transformer


class KMeans(Transformer, Model):
    """
    K-Means clustering algorithm following the Transformer and Model architecture.
    """

    def __init__(self, k, max_iter=100, distance_function=None):
        """
        Initialize the KMeans model.

        Parameters:
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        distance_function: Function to compute distances.
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.distance_function = distance_function
        self.centroids = None
        self.labels = None

    def _fit(self, dataset: Dataset):
        """
        Fit the KMeans model to the dataset.

        Parameters:
        dataset (Dataset): Input dataset.
        """
        X = dataset.X
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iter):

            distances = np.array([self.distance_function(c, X) for c in self.centroids])
            self.labels = np.argmin(distances, axis=0)


            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])


            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.is_fitted = True
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Compute the distances from each sample to all centroids and transform dataset.

        Parameters:
        dataset (Dataset): Input dataset.

        Returns:
        Dataset: Transformed dataset with distances to centroids.
        """
        X = dataset.X
        transformed_data = np.array([self.distance_function(c, X) for c in self.centroids]).T
        return Dataset(X=transformed_data)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Assign clusters to new samples.

        Parameters:
        dataset (Dataset): Input dataset.

        Returns:
        np.ndarray: Cluster assignments.
        """
        distances = self._transform(dataset).features
        return np.argmin(distances, axis=1)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        raise NotImplementedError