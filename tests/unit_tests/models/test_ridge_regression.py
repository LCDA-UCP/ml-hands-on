import unittest
import numpy as np
from ml_hands_on.models.ridge_regression import RidgeRegression
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.metrics.mse import mse


class TestRidgeRegression(unittest.TestCase):

    def setUp(self):

        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        self.dataset = Dataset(X, y)

    def test_fit_and_predict(self):
        model = RidgeRegression(l2_penalty=0.1, alpha=0.1, max_iter=1000, scale=False)
        model.fit(self.dataset)
        predictions = model.predict(self.dataset)
        self.assertEqual(predictions.shape, self.dataset.y.shape)
        self.assertLess(mse(self.dataset.y, predictions), 1e-1)

    def test_score_returns_mse(self):
        model = RidgeRegression(l2_penalty=0.0, alpha=0.1, max_iter=1000, scale=False)
        model.fit(self.dataset)
        score = model.score(self.dataset)
        expected = mse(self.dataset.y, model.predict(self.dataset))
        self.assertAlmostEqual(score, expected, places=5)

    def test_cost_decreases(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.05, max_iter=200, scale=False)
        model.fit(self.dataset)
        costs = list(model.cost_history.values())
        self.assertGreater(costs[0], costs[-1])

    def test_model_with_scaling(self):
        model = RidgeRegression(scale=True, alpha=0.1, max_iter=500)
        model.fit(self.dataset)
        predictions = model.predict(self.dataset)
        self.assertEqual(predictions.shape, self.dataset.y.shape)
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)
        self.assertLess(mse(self.dataset.y, predictions), 0.25)

    def test_no_regularization(self):
        model = RidgeRegression(l2_penalty=0.0, alpha=0.1, max_iter=1000)
        model.fit(self.dataset)
        predictions = model.predict(self.dataset)
        self.assertLess(mse(self.dataset.y, predictions), 1e-1)



if __name__ == '__main__':
    unittest.main()
