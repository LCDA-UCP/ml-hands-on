from typing import Tuple, Sequence

import numpy as np
import pandas as pd


class Dataset:

    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label (self)->bool:

        if self.y is None:
            return False
        else:
            return True

#get_classes: Returns the classes of the dataset (possible values of y).

    def get_class (self):

        if self.has_label():

            return np.unique(self.y)
        else:
            return None


    def get_mean(self)->np.ndarray:
        """
        Computes the mean of each feature.

        returns
        -------
        np.ndarray
        array containing the mean of each feature.
        """
        return np.nanmean(self.X)

    def get_variance(self)->np.ndarray:
        """
        Computes the variance of each feature.

        returns
        -------
        np.ndarray
        array containing the variance of each feature.
        """
        return np.nanvar(self.X)

    def get_median(self)->np.ndarray:
        """
        Computes the median of each feature.

        returns
        -------
        np.ndarray
        array containing the median of each feature.
        """
        return np.nanmedian(self.X)

    def get_min(self)->np.ndarray:
        """
        Computes the minimum value of each feature.

        returns
        -------
        np.ndarray
        array containing the  minimum value of each feature.
        """
        return np.nanmin(self.X)

    def get_max(self) -> np.ndarray:

        """
       Computes the max value of each feature.

       returns
       -------
       np.ndarray
       array containing the  max value of each feature.
       """

        return np.nanmax(self.X)


    def get_summary(self)-> pd.DataFrame:

        data={
            'Mean': self.get_mean(),
            'Variance': self.get_variance(),
            'Median': self.get_median(),
            'Max': self.get_max(),
            'Min': self.get_min()
        }
        summary = pd.DataFrame(data, index=self.features)

        return summary