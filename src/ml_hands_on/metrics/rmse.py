import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
       Compute Root Mean Squared Error.

       Parameters
       ----------
       y_true : np.ndarray
           Ground truth (correct) target values.
       y_pred : np.ndarray
           Estimated target values.

       Returns
       -------
       rmse : float
           Root Mean Squared Error.
       """

    return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))
