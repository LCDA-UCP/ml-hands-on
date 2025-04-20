import unittest
import numpy as np

from ml_hands_on.clustering.kmeans import KMeans
from ml_hands_on.statistics.euclidean_distance import euclidean_distance
from ml_hands_on.data import Dataset


class test_kmeans(unittest.TestCase):

    def test_fit(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        dataset = Dataset(X=X, y=None, features=["f1", "f2"])
        kmeans = KMeans(k=2, distance_function=euclidean_distance)
        kmeans._fit(dataset)

        assert len(kmeans.labels) == X.shape[0]
        assert kmeans.centroids.shape == (2, X.shape[1])

    def test_transform(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        dataset = Dataset(X=X, y=None, features=["f1", "f2"])
        kmeans = KMeans(k=2, distance_function=euclidean_distance)
        kmeans._fit(dataset)
        transformed_dataset = kmeans._transform(dataset)

        assert transformed_dataset.X.shape == (X.shape[0], 2)

    def test_predict(self):
        X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

        dataset = Dataset(X=X, y=None, features=["f1", "f2"])
        kmeans = KMeans(k=2, distance_function=euclidean_distance)
        kmeans._fit(dataset)
        predictions = kmeans._predict(dataset)

        assert len(predictions) == X.shape[0]
        assert set(predictions).issubset({0, 1})
