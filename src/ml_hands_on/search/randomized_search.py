import numpy as np
import random
from copy import deepcopy


class RandomizedSearch:
    def __init__(self, param_distributions: dict, n_iter: int, random_state: int = None, scoring=None):
        """
        RandomizedSearch performs randomized hyperparameter search.

        Parameters
        ----------
        param_distributions : dict
            Keys are parameter names, values are lists or distributions to sample from.
        n_iter : int
            Number of parameter settings to sample.
        random_state : int, optional
            Random seed for reproducibility.
        scoring : callable
            Function that takes (model, dataset) and returns a score.
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def search(self, model, dataset):
        """
        Perform randomized search over hyperparameters.

        Parameters
        ----------
        model : object
            Model with fit() and score() methods.
        dataset : Dataset
            Dataset object with training and validation sets (e.g., dataset.train, dataset.val).
        """
        best_score = -float("inf")
        best_params = None

        for _ in range(self.n_iter):
            # Sample a random parameter set
            sampled_params = {
                key: np.random.choice(values) if isinstance(values, list) else values.rvs()
                for key, values in self.param_distributions.items()
            }

            # Clone the model and set the sampled parameters
            model_clone = deepcopy(model)
            for param, value in sampled_params.items():
                setattr(model_clone, param, value)

            # Train and evaluate
            model_clone.fit(dataset.train)
            score = self.scoring(model_clone, dataset.val)

            if score > best_score:
                best_score = score
                best_params = sampled_params

        self.best_params_ = best_params
        self.best_score_ = best_score