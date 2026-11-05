"""
Logistic regressions used inside the segments of an Lrtree
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lrtree.discretization import Processing, as_categories

# Parameters of the parent class, computed once: scikit-learn re-inspects the signature of ``__init__`` on every
# ``get_params`` call (hence on every ``repr`` / ``clone``)
_LOGREG_PARAMS = tuple(sorted(set(inspect.signature(LogisticRegression.__init__).parameters) - {"self"}))


class PossiblyOneClassReg(LogisticRegression):
    """
    Logistic regression that also accepts a single class in the training labels (e.g. when a leaf is pure): the
    prediction is then that class, with probability 1.
    """
    _single_class_label = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the regression, or remember the label of the single class and the number of features
        """
        y = np.asarray(y)
        if np.unique(y).size == 1:
            self._single_class_label = y[0]
            self.n_features_in_ = np.shape(X)[1]
            return self
        # A segment that was pure at the previous fit is not anymore
        self._single_class_label = None
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict the single class if there is one
        """
        if self._single_class_label is not None:
            return np.full(np.shape(X)[0], self._single_class_label)
        return super().predict(X)

    def predict_proba(self, X):
        """
        Predict the single class with a 1 probability if there is one
        """
        if self._single_class_label is not None:
            proba = np.zeros((np.shape(X)[0], 2))
            proba[:, int(self._single_class_label == 1)] = 1.0
            return proba
        return super().predict_proba(X)


class LogRegSegment(PossiblyOneClassReg):
    """
    Logistic regression of one segment. All keyword arguments but the first four are those of
    :class:`sklearn.linear_model.LogisticRegression`.

    When the features have to be treated inside the segment (categorical features, discretisation or merging of
    levels), the regression takes the features as a 2-d array whose columns are in the order of ``column_names`` and
    treats them with a :class:`lrtree.discretization.Processing` fitted on the segment.

    :param bool discretize: whether to discretise the numerical features inside the segment
    :param bool group: whether to merge the levels of the categorical features inside the segment
    :param list categorical: names of the categorical features
    :param list column_names: names of the columns of the features (needed to find the categorical ones)
    """

    def __init__(self, discretize: bool = False, group: bool = False, categorical=None, column_names=None,
                 **kwargs):
        unknown = set(kwargs) - set(_LOGREG_PARAMS)
        if unknown:
            raise TypeError(f"Unexpected keyword argument(s) {sorted(unknown)}")
        super().__init__(**kwargs)
        self.discretize = discretize
        self.group = group
        self.categorical = categorical
        self.column_names = column_names
        self.processing = None

    @classmethod
    def _get_param_names(cls):
        return [*_LOGREG_PARAMS, "discretize", "group", "categorical", "column_names"]

    @property
    def _treats_features(self) -> bool:
        return bool(self.discretize or self.categorical)

    def _as_frame(self, X) -> pd.DataFrame:
        X = np.asarray(X, dtype=object)
        columns = list(self.column_names) if self.column_names is not None else [f"x{j}" for j in range(X.shape[1])]
        frame = pd.DataFrame(X, columns=columns)
        categorical = [] if self.categorical is None else list(self.categorical)
        for column in columns:
            frame[column] = as_categories(frame[column]) if column in categorical else frame[column].astype(float)
        return frame

    def _transform(self, X):
        if self._treats_features and self._single_class_label is None:
            return self.processing.transform(self._as_frame(X)).to_numpy(dtype=float)
        return np.asarray(X)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if self._treats_features and np.unique(y).size > 1:
            frame = self._as_frame(X)
            frame["y"] = y
            self.processing = Processing(target="y", discretize=self.discretize, group=self.group)
            X = self.processing.fit_transform(frame, self.categorical).to_numpy(dtype=float)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        return super().predict(self._transform(X))

    def predict_proba(self, X):
        return super().predict_proba(self._transform(X))
