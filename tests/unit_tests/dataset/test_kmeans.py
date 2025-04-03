import unittest
import numpy as np
from ml_hands_on.clustering.kmeans import KMeans
from ml_hands_on.statistics.euclidean_distance import euclidean_distance
from ml_hands_on.data import Dataset

class test_kmeans(unittest.TestCase):

    def test_fit(self):
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [10, 2], [10, 4], [10, 0]])
        dataset = Dataset(X=X, y=None)

        model = KMeans(k=2, max_iter=100, distance_function=euclidean_distance)
        model._fit(dataset)

        assert model.centroids.shape == (2, 2)
        assert len(model.labels) == 6
        assert set(model.labels).issubset({0, 1})

    def test_transform(self):
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [10, 2], [10, 4], [10, 0]])
        dataset = Dataset(X=X, y=None)

        model = KMeans(k=2, max_iter=100, distance_function=euclidean_distance)
        model._fit(dataset)

        transformed = model._transform(dataset)
        assert transformed.X.shape == (6, 2)

    def test_predict(self):
        X_train = np.array([[1, 2], [1, 4], [1, 0],
                            [10, 2], [10, 4], [10, 0]])
        train_set = Dataset(X=X_train, y=None)

        model = KMeans(k=2, max_iter=100, distance_function=euclidean_distance)
        model._fit(train_set)

        X_test = np.array([[0, 0], [12, 3]])
        test_set = Dataset(X=X_test, y=None)

        preds = model._predict(test_set)
        assert preds.shape == (2,)
        assert all(p in [0, 1] for p in preds)

