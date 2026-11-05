"""
Predict, predict_proba and precision methods for the Lrtree class
"""
import numpy as np

from lrtree.fit import _model_features, _segments, _tree_features


def _predict(self, X: np.ndarray, fun: str) -> np.ndarray:
    self._check_is_fitted()
    # Features are used positionally, as when fitting
    X_tree = _tree_features(self, X)
    X = _model_features(self, X)
    # Predicted segment (c_map) of each sample
    classes = _segments(self.best_link, X_tree, self.leaves_as_segment)

    prediction = np.empty(X.shape[0])
    for label in np.unique(classes):
        in_segment = classes == label
        # A segment with no training observation has no regression of its own
        model = self.best_logreg.get(label, self.global_logreg)
        if fun == "predict":
            prediction[in_segment] = model.predict(X[in_segment])
        else:
            prediction[in_segment] = model.predict_proba(X[in_segment])[:, 1]
    if fun == "predict":
        # Back to the labels given when fitting
        return self.classes_[prediction.astype(int)]
    return prediction


def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predicts the labels for new values using previously fitted lrtree object

    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    """
    return _predict(self, X, fun="predict")


def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """
    Predicts the probability of the labels for new values using previously fitted lrtree object

    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    """
    return _predict(self, X, fun="predict_proba")


def precision(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Scores the precision of the prediction on the test set

    :param numpy.ndarray X_test:
        array_like of shape (n_samples, n_features)
        Vector used to predict values of y
    :param numpy.ndarray y_test:
        array_like of shape (n_samples, 1)
        Vector of the value, aimed to be predicted, in the data
    :return: precision
    :rtype: float
    """

    # X_train and y_train same size
    if len(X_test) != len(y_test):
        msg = "X_test and y_test need to have the same size"
        raise ValueError(msg)

    prediction = self.predict(X_test)
    return float(np.mean(prediction == np.asarray(y_test)))
