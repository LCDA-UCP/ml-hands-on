from ml_hands_on.search.randomized_search import RandomizedSearch
from ml_hands_on.models.perceptron import Perceptron
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.metrics.accuracy import accuracy

import unittest
import numpy as np
from scipy.stats import uniform


class TestRandomizedSearch(unittest.TestCase):

    def setUp(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 1])
        X_val = np.array([[4, 5], [5, 6]])
        y_val = np.array([1, 0])

        self.dataset = Dataset(X=X_train, y=y_train, features=["f1", "f2"])
        self.dataset.train = Dataset(X=X_train, y=y_train, features=["f1", "f2"])
        self.dataset.val = Dataset(X=X_val, y=y_val, features=["f1", "f2"])

    def test_init_randomized_search(self):
        rs = RandomizedSearch(
            param_distributions={"learning_rate": [0.01, 0.1]},
            n_iter=3,
            random_state=42,
            scoring=lambda m, d: m.score(d)
        )
        assert rs.n_iter == 3
        assert rs.param_distributions is not None

    def test_search_returns_best(self):
        model = Perceptron()
        rs = RandomizedSearch(
            {"learning_rate": [0.001, 0.01], "max_iter": [10, 20]},
            n_iter=5,
            random_state=123,
            scoring=lambda m, d: m.score(d)
        )
        rs.search(model, self.dataset)
        assert rs.best_params_
        assert rs.best_score_ >= 0

    def test_search_is_reproducible(self):
        params = {"learning_rate": [0.01, 0.1], "max_iter": [5, 10]}
        model1 = Perceptron()
        model2 = Perceptron()

        rs1 = RandomizedSearch(params, n_iter=5, random_state=42, scoring=lambda m, d: m.score(d))
        rs2 = RandomizedSearch(params, n_iter=5, random_state=42, scoring=lambda m, d: m.score(d))

        rs1.search(model1, self.dataset)
        rs2.search(model2, self.dataset)

        assert rs1.best_params_ == rs2.best_params_
        assert rs1.best_score_ == rs2.best_score_

    def test_with_distribution(self):
        model = Perceptron()
        dist = {
            "learning_rate": uniform(0.001, 0.009),
            "max_iter": [10, 20]
        }

        rs = RandomizedSearch(dist, n_iter=5, random_state=1, scoring=lambda m, d: m.score(d))
        rs.search(model, self.dataset)

        lr = rs.best_params_["learning_rate"]
        it = rs.best_params_["max_iter"]

        assert 0.001 <= lr <= 0.01
        assert it in [10, 20]
