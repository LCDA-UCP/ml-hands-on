import unittest
import numpy as np
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.base.model import Model
from ml_hands_on.models.stacking import StackingClassifier



class DummyClassifier(Model):

    def __init__(self, constant=None):
        self.constant = constant
        self.is_fitted = False

    def _fit(self, dataset: Dataset):
        if self.constant is None:
            self.constant = round(np.mean(dataset.y))
        self.is_fitted = True

    def _predict(self, dataset: Dataset) -> np.ndarray:
        return np.full(shape=(dataset.X.shape[0],), fill_value=self.constant)

    def _score(self, dataset: Dataset) -> float:
        y_pred = self._predict(dataset)
        return np.mean(y_pred == dataset.y)




class TestStackingClassifier(unittest.TestCase):

    def setUp(self):

        self.X = np.array([[0], [1], [2], [3], [4]])
        self.y = np.array([0, 1, 1, 0, 1])
        self.dataset = Dataset(X=self.X, y=self.y)

        self.base_models = [DummyClassifier(constant=0), DummyClassifier(constant=1)]
        self.meta_model = DummyClassifier()

        self.model = StackingClassifier(base_models=self.base_models, meta_model=self.meta_model)

    def test_fit(self):

        self.model.fit(self.dataset)
        self.assertTrue(all(model.is_fitted for model in self.base_models))
        self.assertTrue(self.meta_model.is_fitted)

    def test_predict_shape(self):

        self.model.fit(self.dataset)
        y_pred = self.model.predict(self.dataset)
        self.assertEqual(y_pred.shape, (self.X.shape[0],))

    def test_score(self):

        self.model.fit(self.dataset)
        score = self.model.score(self.dataset)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
