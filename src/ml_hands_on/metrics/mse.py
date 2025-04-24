import numpy as np

def mse(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    - y_true: array-like of shape (n_samples,) – actual target values
    - y_pred: array-like of shape (n_samples,) – predicted target values

    Returns:
    - float: MSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)