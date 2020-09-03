import math
import numpy as np
import warnings
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pytest

import glmtree

def test_args_fit():
    n = 1000
    d = 4
    X, y, _ = glmtree.Glmtree._generate_test_data(n, d)

    model = glmtree.Glmtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)
    model = glmtree.Glmtree(test=False, validation=True, criterion="aic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)
    model = glmtree.Glmtree(test=False, validation=True, criterion="gini", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)
    model = glmtree.Glmtree(test=False, validation=True, criterion="bic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)
    model = glmtree.Glmtree(test=False, validation=False, criterion="bic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)
    model = glmtree.Glmtree(test=False, validation=False, criterion="gini", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)


def test_dataset_length():

    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.zeros(shape=(1001, 1))
        model = glmtree.Glmtree()
        model.fit(X, y)

def test_data_type():
    n = 1000
    d = 4
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), dtype=str)

    with pytest.raises(ValueError):
        X = np.random.choice(alphabet, [n, d])
        y = np.zeros(shape=(1000, 1))
        model = glmtree.Glmtree()
        model.fit(X, y)

    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.random.choice(alphabet, [n, d])
        model = glmtree.Glmtree()
        model.fit(X, y)

def test_split():
    n = 1000
    d = 4
    X, y, _ = glmtree.Glmtree._generate_test_data(n, d)

    model = glmtree.Glmtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y)


def test_not_fit():
    pass



