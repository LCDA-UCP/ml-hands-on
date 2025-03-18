from ml_hands_on.data.dataset import Dataset
from ml_hands_on.base.transformer import Transformer

import numpy as np

class VarianceThreshold(Transformer):
    """
       Classe para remover features com baixa variância.

       Parâmetros
       ----------
       threshold: float
           Valor mínimo de variância para manter uma feature.
       """

    def __init__(self, threshold: float = 0.0):
        """
            Construtor da classe VarianceThreshold.

            Parâmetros
            ----------
            threshold: float
                Valor mínimo de variância para manter uma feature.
            """

        self.threshold = threshold
        self.variance = None
        super().__init__()

    def _fit(self, dataset:Dataset):
        """
                Calcula e guarda a variância das features do dataset.

                Parameters
                ----------
                dataset : Dataset
                    Dataset original com todas as features.

                Returns
                -------
                self : object
                    Retorna a própria instância já com as variâncias calculadas.
                """
        self.variance = dataset.get_variance()
        return self

    def _transform(self, dataset:Dataset)-> Dataset:
        """
            Remove as features com variância inferior ou igual ao threshold.

            Parameters
            ----------
            dataset : Dataset
                Dataset original com todas as features.

            Returns
            -------
            Dataset
                Novo dataset apenas com features cuja variância é superior ao threshold.
            """

        mask = self.variance > self.threshold
        X_selected = dataset.X[:,mask]

        features_selected = None
        if dataset.features is not None:
            features_selected = []
            for i in range(len(mask)):
                if mask[i]:
                    features_selected.append(dataset.features[i])

        return Dataset(X=X_selected, y=dataset.y, features=features_selected, label=dataset.label)




