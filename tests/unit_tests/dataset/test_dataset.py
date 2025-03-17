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

    def test_get_mean(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(X, features=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(dataset.get_mean(), np.array([4, 5, 6]))

    def test_get_variance(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(X, features=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(dataset.get_variance(), np.array([6, 6, 6]))

    def test_get_median(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(X, features=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(dataset.get_median(), np.array([4, 5, 6]))

    def test_get_min(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(X, features=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(dataset.get_min(), np.array([1, 2, 3]))

    def test_get_max(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataset = Dataset(X, features=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(dataset.get_max(), np.array([7, 8, 9]))


    def test_dropna(self):
        X = np.array([[1, np.nan, 3], [4, 5, 6], [np.nan, 8, 9]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)
        dataset.dropna()
        self.assertEqual(dataset.shape(), (1, 3))

    def test_fillna(self):
        X = np.array([[1, np.nan, 3], [4, 5, 6], [np.nan, 8, 9]])
        dataset = Dataset(X)
        dataset.fillna(strategy='mean')
        self.assertFalse(np.isnan(dataset.X).any())

        dataset = Dataset(X)
        dataset.fillna(strategy='value', value=0)
        self.assertFalse(np.isnan(dataset.X).any())

