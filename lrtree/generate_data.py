"""
generate_data module for the Lrtree class: generating some data to test the algorithm on.
"""
import numpy as np


@staticmethod
def generate_data(n: int, d: int, seed=None, theta: np.ndarray = None):
    """
    Generates some toy continuous data that gets discretized, and a label
    is drawn from a logistic regression given the discretized features.

    :param int n: Number of observations to draw.
    :param int d: Number of features to draw.
    :param numpy.ndarray theta: Logistic regression coefficient to use (if None, use the one provided).
    :param int seed: numpy random seed
    :return: generated data x and y, coefficient theta and bic
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, float
    """
    np.random.seed(seed)
    if theta is not None and not isinstance(theta, np.ndarray):
        raise ValueError("theta must be an np.array (or None).")
    elif theta is not None and seed is not None:
        raise ValueError("theta and seed provided, aborting.")
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

    # penalization term: x[:, 1] not used but there's an intercept
    bic = (d - 1) * sum([np.log((leaf == i).sum()) for i in np.unique(leaf)])
    # log loss term
    bic -= 2 * np.log(y * p + (1 - y) * (1 - p)).sum()

    return x, y, theta, bic
