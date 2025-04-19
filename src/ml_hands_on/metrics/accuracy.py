
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
       Compute the classification accuracy.

       Parameters
       ----------
       y_true : np.ndarray
           Ground truth (correct) target values.
       y_pred : np.ndarray
           Estimated target values.

       Returns
       -------
       accuracy : float
           The proportion of correctly predicted samples.
       """

    return np.mean(y_true.flatten() == y_pred.flatten())
