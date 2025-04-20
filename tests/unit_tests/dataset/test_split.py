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

    def test_split_stratified_labels(self):

        _, test_ds = train_test_split(self.dataset, test_size=0.3, random_state=42)
        for label in test_ds.y:
            self.assertIn(label, self.y)

    def test_random_state_reproducibility(self):

        split1 = train_test_split(self.dataset, test_size=0.3, random_state=123)
        split2 = train_test_split(self.dataset, test_size=0.3, random_state=123)

        np.testing.assert_array_equal(split1[0].X, split2[0].X)
        np.testing.assert_array_equal(split1[1].X, split2[1].X)

    def test_invalid_test_size(self):
        with self.assertRaises(ValueError):
            train_test_split(self.dataset, test_size=1.5)

    def test_split_without_labels(self):
        dataset_no_labels = Dataset(X=self.X)
        train_ds, test_ds = train_test_split(dataset_no_labels, test_size=0.2, random_state=1)

        self.assertIsNone(train_ds.y)
        self.assertIsNone(test_ds.y)
        self.assertEqual(train_ds.X.shape[0], 8)
        self.assertEqual(test_ds.X.shape[0], 2)

if __name__ == '__main__':
    unittest.main()