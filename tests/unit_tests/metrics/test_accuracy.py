import unittest
import numpy as np
from ml_hands_on.metrics.accuracy import accuracy

class TestAccuracy(unittest.TestCase):

    def test_perfect_accuracy(self):
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 0])
        acc = accuracy(y_true, y_pred)
        self.assertEqual(acc,1.0)

    def test_zero_accuracy(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        acc = accuracy(y_true, y_pred)
        self.assertEqual(acc,0.0)

    def test_partial_accuracy(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])
        acc = accuracy(y_true, y_pred)
        self.assertEqual(acc,0.5)

    def test_accuracy_with_lists(self):
        y_true = [1, 0, 1, 1]
        y_pred = [1, 1, 1, 0]
        acc = accuracy(y_true, y_pred)
        self.assertAlmostEqual(acc, 0.5)

    def test_accuracy_empty(self):
        y_true = []
        y_pred = []
        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)