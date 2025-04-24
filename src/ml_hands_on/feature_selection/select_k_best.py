from typing import Callable

import numpy as np

from ml_hands_on.base import Transformer
from ml_hands_on.data.dataset import Dataset
from ml_hands_on.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        """
        super().__init__()
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        valid_idxs = ~np.isnan(self.F)  # Exclude NaN features
        F_valid = self.F[valid_idxs]

        if len(F_valid) < self.k:  # Adjust k if fewer valid features exist
            self.k = len(F_valid)

        idxs = np.argsort(F_valid)[-self.k:]  # Select top-k valid features
        selected_features = np.array(dataset.features)[valid_idxs][idxs]

        return Dataset(X=dataset.X[:, valid_idxs][:, idxs], y=dataset.y, features=list(selected_features),
                       label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3, 8],
                                  [0, 1, 4, 3, 1],
                                  [0, 1, 1, 3, 8]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4", "f5"],
                      label="y")

    selector = SelectKBest(k=3)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
