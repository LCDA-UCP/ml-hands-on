import unittest

import numpy as np

from ml_hands_on.data import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_shape(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')
        self.assertEqual((2, 3), dataset.shape())

    def test_dataset_label(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        dataset = Dataset(X)
        self.assertFalse(dataset.has_label())

        dataset = Dataset(X,y)
        self.assertTrue(dataset.has_label())

    def test_get_class(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[1], [2], [2]])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        self.assertTrue(len(dataset.get_class())==2)

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])
        dataset = Dataset(X)

        self.assertIsNone(dataset.get_class())

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([[1], [2]])
        dataset = Dataset(X, y, features=['a', 'b', 'c'], label='y')

        self.assertTrue(len(dataset.get_class()) == 2)