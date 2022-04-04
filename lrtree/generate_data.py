"""
generate_data module for the Lrtree class: generating some data to test the algorithm on.
"""
import numpy as np


def generate_data(n: int, d: int, theta: np.array = None):
    """
    Generates some toy continuous data that gets discretized, and a label
    is drawn from a logistic regression given the discretized features.

    :param int n: Number of observations to draw.
    :param int d: Number of features to draw.
    :param numpy.ndarray theta: Logistic regression coefficient to use (if None, use the one provided).
    :return: generated data x and y, coefficient theta and bic
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, float
    """
    if theta is not None and not isinstance(theta, np.ndarray):
        raise ValueError("theta must be an np.array (or None).")
    elif theta is None:
        theta = np.random.normal(0, 5, (4, 2))  # TODO: generalize to more dimensions and leaves
        # theta = np.array([[3, -9], [-2, 4], [6, 7], [8, -3]])

    x = np.array(np.random.normal(0, 1, (n, d)))
    leaf = np.zeros(n)
    leaf[np.logical_and(x[:, 0] < 0, x[:, 1] >= 0)] = 1
    leaf[np.logical_and(x[:, 0] >= 0, x[:, 1] < 0)] = 2
    leaf[np.logical_and(x[:, 0] >= 0, x[:, 1] >= 0)] = 3
    leaf = leaf.astype(int)

    log_odd = theta[leaf, 0] + theta[leaf, 1] * x[:, 2]
    p = 1 / (1 + np.exp(- log_odd))
    y = np.random.binomial(1, p)

    bic = -d * np.log(n) + 2 * np.log(y * p + (1 - y) * (1 - p)).sum()

    return x, y, theta, bic
