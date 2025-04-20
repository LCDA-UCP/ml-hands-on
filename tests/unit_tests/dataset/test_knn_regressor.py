import unittest
import numpy as np
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.models.knn_regressor import KNNRegressor
from ml_hands_on.metrics.rmse import rmse


class TestKNNRegressor(unittest.TestCase):
    def test_fit(self):
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])
        dataset = Dataset(X_train, y_train)

        knn = KNNRegressor(k=2)
        knn.fit(dataset)
        self.assertEqual(knn.dataset, dataset)

    def test_predict(self):
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])
        dataset = Dataset(X_train, y_train)

        knn = KNNRegressor(k=2)
        knn.fit(dataset)

        X_test = np.array([[2]])
        test_dataset = Dataset(X_test)

        y_pred = knn.predict(test_dataset)
        expected = np.mean([4, 2])  # k=2 neighbors: 2 and 4
        self.assertAlmostEqual(y_pred[0][0], expected)

    def test_score(self):
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([2, 4, 6])
        dataset = Dataset(X_train, y_train)

        X_test = np.array([[2], [3]])
        y_test = np.array([4, 6])
        test_dataset = Dataset(X_test, y_test)

        knn = KNNRegressor(k=1)
        knn.fit(dataset)

        score = knn.score(test_dataset)
        self.assertAlmostEqual(score, 0.0)  # Com k=1, RMSE deve ser 0 se prever perfeitamente

    def test_predict_without_fit(self):
        X_test = np.array([[1]])
        test_dataset = Dataset(X_test)

        knn = KNNRegressor(k=3)
        with self.assertRaises(AttributeError):
            knn.predict(test_dataset)

    def test_rmse_with_lists(self):
        y_true = [2, 4, 6]
        y_pred = [2, 5, 5]
        result = rmse(y_true, y_pred)
        expected = np.sqrt(((0**2 + 1**2 + 1**2) / 3))
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
