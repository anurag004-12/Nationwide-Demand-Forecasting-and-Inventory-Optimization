import numpy as np


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error = total absolute error / total actual sales."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def score(y_true, y_pred):
    return round(wape(y_true, y_pred), 4), round(rmse(y_true, y_pred), 4)


def coverage(y_true, lower, upper):
    """Share of actual values that fall inside the interval (should be ~0.90 for a 90% interval)."""
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    return ((y_true >= lower) & (y_true <= upper)).mean()