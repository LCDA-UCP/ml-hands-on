import unittest
import numpy as np
from ml_hands_on.statistics.euclidean_distance import euclidean_distance
from ml_hands_on.statistics.manhattan_distance import manhattan_distance

class test_distances(unittest.TestCase):

    def test_euclidean_distance(self):
        x = [0, 0]
        y = [[3, 4], [6, 8]]
        result = euclidean_distance(x, y)
        expected = np.array([5.0, 10.0])
        assert np.allclose(result, expected)

    def test_manhattan_distance(self):
        x = [1, 1]
        y = [[2, 2], [3, 3]]
        result = manhattan_distance(x, y)
        expected = np.array([2, 4])
        assert np.allclose(result, expected)
