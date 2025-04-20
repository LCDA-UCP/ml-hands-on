import unittest
import numpy as np
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.model_selection.split import train_test_split

class TestTrainTestSplit(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(30).reshape(10, 3)
        self.y = np.arange(10)
        self.dataset = Dataset(X=self.X, y=self.y)

    def test_split_shapes(self):
        train_ds, test_ds = train_test_split(self.dataset, test_size=0.3, random_state=42)
        self.assertEqual(train_ds.X.shape[0], 7)
        self.assertEqual(test_ds.X.shape[0], 3)