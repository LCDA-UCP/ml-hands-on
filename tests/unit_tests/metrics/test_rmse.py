import unittest
import numpy as np
from ml_hands_on.metrics.rmse import rmse

class TestRMSE(unittest.TestCase):

    def test_rmse_basic(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        result = rmse(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_rmse_with_different_values(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([3, 2, 1])
        expected = np.sqrt(((1 - 3)**2 + (2 - 2)**2 + (3 - 1)**2) / 3)
        result = rmse(y_true, y_pred)
        self.assertAlmostEqual(result, expected)

    def test_rmse_with_lists(self):
        y_true = [1, 2, 3]
        y_pred = [3, 2, 1]
        expected = np.sqrt(((1 - 3)**2 + (2 - 2)**2 + (3 - 1)**2) / 3)
        result = rmse(y_true, y_pred)
        self.assertAlmostEqual(result, expected)

    def test_rmse_with_zeros(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        result = rmse(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_rmse_with_negative_values(self):
        y_true = np.array([-1, -2, -3])
        y_pred = np.array([-3, -2, -1])
        expected = np.sqrt(((-1 + 3)**2 + (-2 + 2)**2 + (-3 + 1)**2) / 3)
        result = rmse(y_true, y_pred)
        self.assertAlmostEqual(result, expected)

    def test_rmse_with_large_values(self):
        y_true = np.array([1000000, 2000000, 3000000])
        y_pred = np.array([1000000, 2000000, 3000000])
        result = rmse(y_true, y_pred)
        self.assertEqual(result, 0.0)

if __name__ == "__main__":
    unittest.main()

