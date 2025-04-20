import unittest
import numpy as np
from scipy.stats import uniform

from ml_hands_on.search.randomized_search import RandomizedSearch
from ml_hands_on.models.perceptron import Perceptron
from ml_hands_on.data import Dataset
from ml_hands_on.metrics import accuracy  # assumindo que já existe


class TestRandomizedSearch(unittest.TestCase):

    def setUp(self):
        # Dados para treino e validação (X com 2 features, y binário)
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 1])
        X_val = np.array([[4, 5], [5, 6]])
        y_val = np.array([1, 0])

        self.dataset = Dataset(X=X_train, y=y_train, features=["f1", "f2"])
        self.dataset.train = Dataset(X=X_train, y=y_train, features=["f1", "f2"])
        self.dataset.val = Dataset(X=X_val, y=y_val, features=["f1", "f2"])

        self.scoring = lambda model, val: model.score(val)

    def test_initialization(self):
        rs = RandomizedSearch(param_distributions={"learning_rate": [0.01, 0.1]}, n_iter=3, random_state=42, scoring=self.scoring)
        assert rs.n_iter == 3
        assert "learning_rate" in rs.param_distributions

    def test_search_runs_and_returns_best(self):
        model = Perceptron()
        rs = RandomizedSearch({"learning_rate": [0.001, 0.01], "max_iter": [10, 20]}, n_iter=5, random_state=123, scoring=self.scoring)
        rs.search(model, self.dataset)
        assert rs.best_params_ is not None
        assert rs.best_score_ is not None
        assert rs.best_score_ >= 0

    def test_reproducibility(self):
        model1 = Perceptron()
        model2 = Perceptron()

        rs1 = RandomizedSearch({"learning_rate": [0.01, 0.1]}, n_iter=5, random_state=99, scoring=self.scoring)
        rs2 = RandomizedSearch({"learning_rate": [0.01, 0.1]}, n_iter=5, random_state=99, scoring=self.scoring)

        rs1.search(model1, self.dataset)
        rs2.search(model2, self.dataset)

        assert rs1.best_params_ == rs2.best_params_
        assert rs1.best_score_ == rs2.best_score_

    def test_distribution_object_sampling(self):
        model = Perceptron()
        dist = {"learning_rate": uniform(loc=0.001, scale=0.009), "max_iter": [10, 20]}
        rs = RandomizedSearch(dist, n_iter=5, random_state=42, scoring=self.scoring)
        rs.search(model, self.dataset)
        assert 0.001 <= rs.best_params_["learning_rate"] <= 0.01
