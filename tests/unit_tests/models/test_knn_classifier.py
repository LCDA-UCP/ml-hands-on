import unittest
import numpy as np
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.models.knn_classifier import KNNClassifier
from ml_hands_on.metrics.accuracy import accuracy
from ml_hands_on.model_selection.split import train_test_split


class TestKNNClassifier(unittest.TestCase):
    def test_fit(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        dataset = Dataset(X_train, y_train)

        knn = KNNClassifier(k=3)
        knn.fit(dataset)
        self.assertEqual(knn.dataset, dataset)

    def test_predict(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        dataset = Dataset(X_train, y_train)

        knn = KNNClassifier(k=3)
        knn.fit(dataset)

        X_test = np.array([[3, 4]])
        test_dataset = Dataset(X_test)

        y_pred = knn.predict(test_dataset)
        self.assertEqual(y_pred[0][0], 0)

    def test_score(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        dataset = Dataset(X_train, y_train)

        X_test = np.array([[3, 4], [5, 6]])
        y_test = np.array([1, 0])
        test_dataset = Dataset(X_test, y_test)

        knn = KNNClassifier(k=1)
        knn.fit(dataset)

        score = knn.score(test_dataset)
        self.assertAlmostEqual(score, 1.0)

    def test_accuracy_with_lists(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 1, 1, 0]
        acc = accuracy(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.5)

    def test_predict_without_fit(self):
        X_test = np.array([[1, 2], [3, 4]])
        test_dataset = Dataset(X_test)

        knn = KNNClassifier(k=3)
        with self.assertRaises(ValueError):
            knn.predict(test_dataset)

    def test_invalid_test_size(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        dataset = Dataset(X, y)

        with self.assertRaises(ValueError):
            train_test_split(dataset, test_size=1.5)

if __name__ == "__main__":
    unittest.main()
