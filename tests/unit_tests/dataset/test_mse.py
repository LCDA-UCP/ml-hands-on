import unittest
import numpy as np
from ml_hands_on.metrics.mse import mse


class TestMSEFunction(unittest.TestCase):
    def test_mse_perfect_prediction(self):
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        self.assertEqual(mse(y_true, y_pred), 0.0)

    def test_mse_simple_case(self):
        y_true = [1, 2, 3]
        y_pred = [2, 3, 4]
        expected = np.mean([1**2, 1**2, 1**2])
        self.assertAlmostEqual(mse(y_true, y_pred), expected, places=7)

    def test_mse_numpy_inputs(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 0.0])
        expected = ((1.0 - 2.0) ** 2 + (2.0 - 0.0) ** 2) / 2
        self.assertAlmostEqual(mse(y_true, y_pred), expected, places=7)

    def test_mse_negatives(self):
        y_true = [-1, -2, -3]
        y_pred = [-1, -1, -1]
        expected = np.mean([(0)**2, (1)**2, (2)**2])
        self.assertAlmostEqual(mse(y_true, y_pred), expected, places=7)



if __name__ == '__main__':
    unittest.main()