from ml_hands_on.data.dataset import Dataset
from ml_hands_on.base.model import Model
from ml_hands_on.metrics.accuracy import accuracy

import numpy as np

class StackingClassifier(Model):
    """
    Stacking Classifier that combines multiple base classifiers with a final meta-model.

    Parameters
    ----------
    base_models : list
        List of base model instances, following the portfolio's Model interface.
    meta_model : Model
        A classifier that learns from the predictions of the base models.
    """

    def __init__(self, base_models: list, meta_model: Model):
        self.base_models = base_models
        self.meta_model = meta_model

    def _fit(self, dataset: Dataset):
        """
        Trains each base model on the original dataset and the meta-model on their predictions.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the input features (X) and labels (y).
        """
        X = dataset.X
        y = dataset.y
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        for model in self.base_models:
            model.fit(dataset)

        Z = np.zeros((n_samples, n_models))
        for i, model in enumerate(self.base_models):
            Z[:, i] = model.predict(dataset)

        meta_dataset = Dataset(X=Z, y=y)
        self.meta_model.fit(meta_dataset)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts labels using the base models' outputs as input to the meta-model.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the input features (X).

        Returns
        -------
        np.ndarray
            Final predictions made by the meta-model.
        """
        X = dataset.X
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        Z = np.zeros((n_samples, n_models))
        for i, model in enumerate(self.base_models):
            Z[:, i] = model.predict(dataset)

        meta_dataset = Dataset(X=Z)
        return self.meta_model.predict(meta_dataset)

    def _score(self, dataset: Dataset,predictions=None) -> float:
        """
        Computes the accuracy of the stacking classifier on the given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing the input features (X) and true labels (y).
            predictions : np.ndarray, optional
        Precomputed predictions to use for scoring. If None, predictions will be
        generated using the model's _predict method.

        Returns
        -------
        float
            Accuracy score (ratio of correct predictions).
        """
        y_true = dataset.y
        if predictions is None:
            y_pred = self._predict(dataset)
        else:
            y_pred = predictions
        return accuracy(y_true, y_pred)


