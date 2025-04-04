from ml_hands_on.data.dataset import Dataset
from ml_hands_on.base.transformer import Transformer

import numpy as np

class VarianceThreshold(Transformer):
    """
        Class to remove features with low variance.

        Parameters
        ----------
        threshold: float
            Minimum variance required to retain a feature.
    """

    def __init__(self, threshold: float = 0.0):
        """
            Constructor of the VarianceThreshold class.

            Parameters
            ----------
            threshold: float
                Minimum variance value to retain a feature.
        """

        self.threshold = threshold
        self.variance = None
        super().__init__()

    def _fit(self, dataset:Dataset):
        """
            Calculates and stores the variance of the dataset features.

            Parameters
            ----------
                dataset : Dataset
                Original dataset containing all features.

            Returns
            -------
                self : object
                Returns the instance itself with variances already computed.
        """

        self.variance = dataset.get_variance()
        return self

    def _transform(self, dataset:Dataset)-> Dataset:
        """
            Removes features with variance less than or equal to the threshold.

            Parameters
            ----------
                dataset : Dataset
                Original dataset containing all features.

            Returns
            -------
            Dataset
                New dataset containing only features whose variance exceeds the threshold.
        """

        mask = self.variance > self.threshold
        X_selected = dataset.X[:,mask]

        features_selected = None
        if dataset.features is not None:
            features_selected = list(np.array(dataset.features)[mask])

        return Dataset(X=X_selected, y=dataset.y, features=features_selected, label=dataset.label)




