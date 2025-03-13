from typing import Tuple, Sequence, Optional

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

    def dropna(self)->None:

        """Removes all samples containing ate least one null value from the dataset"""

        mask = ~np.isnan(self.X).any(axis=1)
        self.X=self.X[mask]
        if self.has_label():
            self.y=self.y[mask]


    def fillna(self,strategy:str = "mean", value: Optional[float] = None)->None:

        """
        Fills missing values in the dataset using a specified strategy.

        Parameters
        ----------
        strategy : str, default="mean"
            Strategy to use for filling missing values. Options are "mean", "median", or "value".
        value : Optional[float], default=None
            Specific value to use if strategy is "value".

        Raises
        ------
        ValueError
            If an invalid strategy is provided.
              """
        if strategy == "mean":   # I set strategy = mean by default
            fill_values = np.nanmean(self.X)
        elif strategy == "median":
            fill_values = np.nanmedian(self.X)
        elif strategy == "value" and value is not None:
            fill_values = np.full(self.X.shape[1],value)
        elif strategy== "value" and value is None:
            raise ValueError("must specify the value when the strategy is value")
        else:
            raise ValueError("Invalid strategy. Choose mean , media, or provide a especific value ")

        for i in range (self.X.shape[1]) :

            self.X[:,i] = np.where(np.isnan(self.X[:,i]), fill_values[i],self.X[:,i])


    def remove_index(self, index:int)->None:
        """
          Removes a sample from the dataset by its index.

           Parameters
           ----------
           index : int
               Index of the sample to remove.
        """
        self.X =np.delete(self.X,index)
        if self.has_label():
            self.y = np.delete(self.y,index)