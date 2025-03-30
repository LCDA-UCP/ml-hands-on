import numpy as np
from ml_hands_on.data import Dataset
from ml_hands_on.model import Model
from ml_hands_on.base.transformer import Transformer


class KMeans(Transformer, Model):
    """
    K-Means clustering algorithm following the Transformer and Model architecture.
    """

    def __init__(self, k, max_iter=100, distance_function=None):
        """
        Initialize the KMeans model.

        Parameters:
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        distance_function: Function to compute distances.
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.distance_function = distance_function
        self.centroids = None
        self.labels = None

