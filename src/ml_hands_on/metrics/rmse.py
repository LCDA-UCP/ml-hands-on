import numpy as np

def rmse(y_true, y_pred):
    """
       Compute Root Mean Squared Error.

       Parameters
       ----------
       y_true : array-like
           Ground truth (correct) target values.
       y_pred : array-like
           Estimated target values.

       Returns
       -------
       rmse : float
           Root Mean Squared Error.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

