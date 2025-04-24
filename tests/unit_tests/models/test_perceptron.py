import unittest
import numpy as np
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.models.perceptron import Perceptron
from ml_hands_on.metrics.accuracy import accuracy


class TestPerceptron(unittest.TestCase):

    def setUp(self):
        X = np.array([[2, 3], [4, 1], [1, 2], [3, 3], [2, 1], [4, 3]])
        y = np.array([0, 1, 0, 1, 0, 1])
        self.dataset = Dataset(X, y)
        self.model = Perceptron(max_iter=1000, learning_rate=0.01)

    def test_fit(self):
        self.model._fit(self.dataset)
        self.assertTrue(self.model.weights is not None)
        self.assertTrue(self.model.bias is not None)
        self.assertTrue(self.model._is_fitted)

    def test_predict(self):
        self.model._fit(self.dataset)
        predictions = self.model._predict(self.dataset)
        self.assertEqual(predictions.shape, (len(self.dataset.X),))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_score(self):
        self.model._fit(self.dataset)
        predictions = self.model._predict(self.dataset)
        accuracy_score = self.model._score(self.dataset, predictions)
        self.assertGreaterEqual(accuracy_score, 0)
        self.assertLessEqual(accuracy_score, 1)

    def test_accuracy_function(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        acc = accuracy(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.8333, places=4)

    def test_multiple_epochs(self):
        self.model = Perceptron(max_iter=5000, learning_rate=0.01)
        self.model._fit(self.dataset)
        predictions = self.model._predict(self.dataset)
        accuracy_score = self.model._score(self.dataset, predictions)
        self.assertGreater(accuracy_score, 0.7)

    def test_edge_case_empty_dataset(self):
        empty_X = np.array([[]])
        empty_y = np.array([])
        empty_dataset = Dataset(empty_X, empty_y)
        with self.assertRaises(ValueError):
            self.model._fit(empty_dataset)


if __name__ == "__main__":
    unittest.main()
