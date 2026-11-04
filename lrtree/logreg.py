"""
Logistic regressions used inside the segments of an Lrtree
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lrtree.discretization import Processing

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
    Logistic regression of one segment. All keyword arguments but ``discretization`` are those of
    :class:`sklearn.linear_model.LogisticRegression`.

    :param bool discretization: whether to discretise the features inside the segment before fitting
        (see :class:`lrtree.discretization.Processing`)
    """

    def __init__(self, discretization: bool = False, **kwargs):
        unknown = set(kwargs) - set(_LOGREG_PARAMS)
        if unknown:
            raise TypeError(f"Unexpected keyword argument(s) {sorted(unknown)}")
        super().__init__(**kwargs)
        self.discretization = discretization
        self.processing = None

    @classmethod
    def _get_param_names(cls):
        return [*_LOGREG_PARAMS, "discretization"]

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])

    def _transform(self, X):
        if self.discretization and self._single_class_label is None:
            return self.processing.transform(self._as_frame(X)).to_numpy()
        return np.asarray(X)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.discretization and np.unique(y).size > 1:
            frame = self._as_frame(X)
            frame["y"] = y
            self.processing = Processing(target="y", discretize=True)
            X = self.processing.fit_transform(frame).to_numpy()
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        return super().predict(self._transform(X))

    def predict_proba(self, X):
        return super().predict_proba(self._transform(X))
