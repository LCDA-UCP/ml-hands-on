import numpy as np


def euclidean_distance(x, y):
    """
    Compute the Euclidean distance between a single sample x and multiple samples y.

    Parameters:
    x: A single sample of shape (n,).
    y: Multiple samples of shape (m, n).

    Returns:
    np.ndarray: An array containing the distances between x and each sample in y.
    """
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((y - x) ** 2, axis=1))