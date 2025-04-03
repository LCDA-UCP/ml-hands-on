import numpy as np

from ml_hands_on.data.dataset import Dataset


class PCA:
    """
    Principal Component Analysis
    Linear algebra technique to reduce the dimensionality of a dataset.
    It finds correlations between features and projects them onto a lower dimensional space.
    Each PC is a linear combination of the original features that explains the most variance in the data.

    Parameters
    ----------
    n_components: int
        Number of components to keep

    Attributes
    ----------
    mean: np.ndarray
        The mean of the dataset
    components: np.ndarray
        The principal components aka the unitary matrix of eigenvectors
    explained_variance: np.ndarray
        The variance explained by each principal component aka the diagonal matrix of eigenvalues
    """
    def __init__(self, n_components: int = 2):
        """
        Principal Component Analysis algorithm

        Parameters
        ----------
        n_components: int
            Number of components to keep
        """
        # parameters
        self.n_components = n_components

        # attributes
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> 'PCA':
        """
        It fits PCA to compute the eigenvectors and eigenvalues.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        # center the data
        self.mean = np.mean(dataset.X, axis=0)
        X = dataset.X - self.mean

        # u is the unitary matrix of eigenvectors
        # s is the diagonal matrix of eigenvalues
        # v_t is the unitary matrix of right singular vectors
        u, s, v_t = np.linalg.svd(X, full_matrices=False)

        # keep the first n_components
        self.components = v_t[:self.n_components]

        # explained variance
        explained_variance = (s ** 2) / (X.shape[0] - 1)
        self.explained_variance = explained_variance[:self.n_components]
        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It projects the data onto the principal components.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        X_reduced: np.ndarray
            The projected data
        """
        X = dataset.X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits PCA and projects the data onto the principal components.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        X_reduced: np.ndarray
            The projected data
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from ml_hands_on.data.dataset import Dataset
    dataset_ = Dataset.from_random(100, 10)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(dataset_)
    print(X_reduced.shape)
    print(pca.components.shape)
    print(pca.explained_variance)
