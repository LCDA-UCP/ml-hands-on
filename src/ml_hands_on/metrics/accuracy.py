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
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")

    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")

    return np.mean(y_true == y_pred)
