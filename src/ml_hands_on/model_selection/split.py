from typing import Tuple
import numpy as np
from ml_hands_on.data.dataset import Dataset



def train_test_split(dataset: Dataset, test_size: float=0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
        """
            Split arrays or matrices into random train and test subsets.

            Parameters
            ----------
            dataset : Dataset
                The dataset to split.
            test_size : float, default=0.2
                Proportion of the dataset to include in the test split.
            random_state : int, optional
                Random seed used to shuffle the data.

            Returns
            -------
            train_dataset : Dataset
                The training dataset.
            test_dataset : Dataset
                The testing dataset.
        """

        np.random.seed(random_state)
        n_samples = dataset.X.shape[0]
        indices = np.random.permutation(n_samples)

        test_count = int(n_samples * test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        X_train, y_train = dataset.X[train_indices], dataset.y[train_indices] if dataset.has_label() else None
        X_test, y_test = dataset.X[test_indices], dataset.y[test_indices] if dataset.has_label() else None

        train_dataset = Dataset(X_train, y_train, features=dataset.features, label=dataset.label)
        test_dataset = Dataset(X_test, y_test, features=dataset.features, label=dataset.label)

        return train_dataset, test_dataset
