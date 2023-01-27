"""
implements segment-specific, possibly single-class, logistic regression

.. autoclass:: PossiblyOneClassReg
   :members:

.. autoclass:: LogRegSegment
   :members:
"""
from __future__ import annotations
import inspect

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder

from lrtree.discretization import _categorie_data_bin_test
from lrtree.discretization import _categorie_data_bin_train


class PossiblyOneClassReg(sk.linear_model.LogisticRegression):
    """
    One class logistic regression (e.g. when a leaf is pure)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._single_class_label = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray, weights: np.ndarray = None):
        """
        Fit the one class regression: put the label of the single class and the number of features
        """
        if y.nunique() == 1:
            self._single_class_label = y.iloc[0]
            self.n_features_in_ = X.shape[1]
            return self
        else:
            return super().fit(X, y, weights)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict the single class if there's only one class, or predicts

        :param numpy.ndarray or pandas.DataFrame X:
        :returns: column of class prediction
        :rtype: numpy.ndarray
        """
        if self._single_class_label is not None:
            return np.full(X.shape[0], self._single_class_label)
        else:
            return super().predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame):
        """
        Predict the single class with a 1 probability

        :param numpy.ndarray or pandas.DataFrame X:
        :returns: two columns, one of which full of ones
        :rtype: numpy.ndarray
        """
        if self._single_class_label is not None:
            if self._single_class_label == 1:
                return np.full((X.shape[0], 2), [0, 1])
            else:
                return np.full((X.shape[0], 2), [1, 0])
        else:
            return super().predict_proba(X)


class LogRegSegment(PossiblyOneClassReg):
    """
    Logistic Regression for a given segment; fine-tuned PossiblyOneClassReg, itself from sklearn.LogisticRegression,
    allowing (1) to have only one class and (2) to embed data processing (merging, discretization).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.categories = None
        self.data_treatment = False
        self.discretization = False
        self.column_names = None
        self.categories = {}
        if 'data_treatment' in kwargs:
            self.data_treatment = kwargs['data_treatment']
            self.discretization = kwargs['discretization']
            self.column_names = kwargs['column_names']
            self.categories = {"enc": OneHotEncoder(), "merged_cat": {}, "discret_cat": {}}
            self.scaler = None
            self.categorical = None

    def fit(self, **kwargs):
        """
        Fits the LogRegSegment by preprocessing the data (if need be / asked for) and calling fit on sklearn
        """
        train_data = kwargs['X']
        if self.data_treatment:
            train_data, labels, enc, merged_cat, discret_cat, scaler, len_col_num = _categorie_data_bin_train(
                data=train_data,
                var_cible="y",
                categorical=kwargs['categorical'],
                discretize=self.discretization
            )
            self.categorical = kwargs['categorical']
            self.scaler = scaler
            self.categories["enc"] = enc
            self.categories["merged_cat"] = merged_cat
            self.categories["discret_cat"] = discret_cat
        else:
            try:
                train_data.drop(columns="y", inplace=True)
            except KeyError:
                pass
        kwargs.pop("X")
        super_fit_args = list(inspect.signature(super().fit).parameters)
        kwargs_fit = {k: kwargs.pop(k) for k in dict(kwargs) if k in super_fit_args}
        return super().fit(X=train_data, **kwargs_fit)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the LogRegSegment by preprocessing the data (if need be / asked for) and calling predict on sklearn
        """
        if self.data_treatment:
            # Applies the data treatment for this leaf
            # X = X.rename(columns=self.column_names)
            X = _categorie_data_bin_test(data_val=X,
                                         enc=self.categories["enc"],
                                         scaler=self.scaler,
                                         merged_cat=self.categories["merged_cat"],
                                         discret_cat=self.categories["discret_cat"],
                                         categorical=self.categorical,
                                         discretize=self.discretization)
        return super().predict(X.to_numpy())

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts the LogRegSegment by preprocessing the data (if need be / asked for) and calling predict_proba
        on sklearn
        """
        if self.data_treatment:
            # Applies the data treatment for this leaf
            X = X.rename(columns=self.column_names)
            X = _categorie_data_bin_test(data_val=X,
                                         enc=self.categories["enc"],
                                         scaler=self.scaler,
                                         merged_cat=self.categories["merged_cat"],
                                         discret_cat=self.categories["discret_cat"],
                                         categorical=self.categorical,
                                         discretize=self.discretization)
        return super().predict_proba(X.to_numpy())
