import numpy as np


def generate_data2(n, d, theta=None):
    """
    Generates some toy continuous data that gets discretized, and a label
    is drawn from a logistic regression given the discretized features.

    :param int n: Number of observations to draw.
    :param int d: Number of features to draw.
    :param numpy.ndarray theta: Logistic regression coefficient to use (if None, drawn from N(0,2)).
    """

    theta = [[3, -9], [-2, 4], [6, 7], [8, -3]]

    x = np.array(np.random.normal(0, 1, (n, d)))
    leaf = np.zeros(n)
    for i in range(n):
        if x[i][0]<0 and x[i][1]<0 :
            leaf[i]=0
        elif x[i][0] < 0 and x[i][1] >= 0:
            leaf[i] = 1
        elif x[i][0] >= 0 and x[i][1] < 0:
            leaf[i] = 2
        else :
            leaf[i]=3

    log_odd = np.array([0] * n)
    for i in range(n):
        log_odd[i] += theta[int(leaf[i])][0] + theta[int(leaf[i])][1]*x[i][2]

    p = 1 / (1 + np.exp(- log_odd))
    y = np.random.binomial(1, p)

    BIC = -d * np.log(n)
    for i in range(n):
        BIC += y[i]*np.log(p[i]) + (1-y[i])*np.log(1-p[i])

    return x, y, theta, BIC