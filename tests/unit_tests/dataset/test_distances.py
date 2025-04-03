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

    def test_euclidean_distance_zero(self):
        x = [2, 2]
        y = [[2, 2]]
        result = euclidean_distance(x, y)
        expected = np.array([0.0])
        assert np.allclose(result, expected)

    def test_manhattan_distance_zero(self):
        x = [5, 5]
        y = [[5, 5]]
        result = manhattan_distance(x, y)
        expected = np.array([0])
        assert np.allclose(result, expected)

    def test_shape_euclidean(self):
        x = [1, 2, 3]
        y = [[4, 5, 6], [7, 8, 9]]
        result = euclidean_distance(x, y)
        assert result.shape == (2,)

    def test_shape_manhattan(self):
        x = [1, 2, 3, 4]
        y = [[1, 2, 3, 4], [4, 3, 2, 1]]
        result = manhattan_distance(x, y)
        assert result.shape == (2,)


