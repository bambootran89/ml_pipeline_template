import numpy as np


def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) between
    true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)


def smape(y_true, y_pred):
    """
    Compute Symmetric Mean Absolute Percentage Error
    (sMAPE) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: sMAPE in percentage.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return (
        np.mean(np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)) * 100
    )
