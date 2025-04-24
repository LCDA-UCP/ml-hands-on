import unittest
import numpy as np

from ml_hands_on.models.random_forest_classifier import RandomForestClassifier
from ml_hands_on.data.dataset import Dataset


class TestRandomForestClassifier(unittest.TestCase):

    def setUp(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 1, 0, 1])
        self.dataset = Dataset(X=X, y=y, features=["f1", "f2"])

    def test_fit(self):
        model = RandomForestClassifier(n_estimators=5, seed=42)
        model.fit(self.dataset)
        assert len(model.trees) == 5
        assert len(model.feature_indices) == 5

    def test_predict_shape(self):
        model = RandomForestClassifier(n_estimators=3, seed=42)
        model.fit(self.dataset)
        predictions = model.predict(self.dataset)
        assert predictions.shape == (self.dataset.shape()[0],)

    def test_score_range(self):
        model = RandomForestClassifier(n_estimators=3, seed=42)
        model.fit(self.dataset)
        score = model.score(self.dataset)
        assert 0.0 <= score <= 1.0

    def test_reproducibility(self):
        model1 = RandomForestClassifier(n_estimators=3, seed=42)
        model2 = RandomForestClassifier(n_estimators=3, seed=42)

        model1.fit(self.dataset)
        model2.fit(self.dataset)

        preds1 = model1.predict(self.dataset)
        preds2 = model2.predict(self.dataset)

        assert np.array_equal(preds1, preds2)

    def test_predict_classes_are_valid(self):
        model = RandomForestClassifier(n_estimators=3, seed=42)
        model.fit(self.dataset)
        predictions = model.predict(self.dataset)
        unique_classes = np.unique(self.dataset.y)

        for p in predictions:
            assert p in unique_classes
