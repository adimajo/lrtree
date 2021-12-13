"""
Predict, predict_proba and precision methods for the Glmtree class
"""
import pandas as pd
import numpy as np
from copy import deepcopy


def predict(self, X):
    """
    Predicts the labels for new values using previously fitted glmtree object

    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    """
    # List of models for each class
    link = self.best_link
    logreg = self.best_logreg
    # Predicted class for each sample
    classes = link.predict(X)
    liste_cla = np.unique(classes)

    X_df = pd.DataFrame(X)
    X_df["class"] = classes
    X_df["pred"] = 0

    for i in range(len(liste_cla)):
        filtre = X_df["class"] == liste_cla[i]
        bloc = deepcopy(X_df[filtre].drop(["class", "pred"], axis=1))
        bloc_pred = logreg[i].predict(bloc.add_prefix("par_"))
        k = 0
        for j in range(len(X_df)):
            if filtre[j]:
                X_df.loc[j, "pred"] = bloc_pred[k]
                k = k + 1

    return X_df["pred"].to_numpy()


def predict_proba(self, X):
    """
    Predicts the probability of the labels for new values using previously fitted glmtree object

    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    """
    # List of models for each class
    link = self.best_link
    logreg = self.best_logreg
    # Predicted class for each sample
    classes = link.predict(X)
    liste_cla = np.unique(classes)

    X_df = pd.DataFrame(X)
    X_df["class"] = classes
    X_df["pred"] = 0

    for i in range(len(liste_cla)):
        filtre = X_df["class"] == liste_cla[i]
        bloc = deepcopy(X_df[filtre].drop(["class", "pred"], axis=1))
        bloc_pred = logreg[i].predict_proba(bloc.add_prefix("par_"))
        k = 0
        for j in range(len(X_df)):
            if filtre[j]:
                X_df.loc[j, "pred"] = bloc_pred[k][1]
                k = k + 1

    return X_df["pred"].to_numpy()


def precision(self, X_test, y_test):
    """Scores the precision of the prediction on the test set
    :param numpy.ndarray X_test:
        array_like of shape (n_samples, n_features)
        Vector used to predict values of y
    :param numpy.ndarray y_test:
        array_like of shape (n_samples, 1)
        Vector of the value, aimed to be predicted, in the data
    """

    # X_train and y_train same size
    if len(X_test) != len(y_test):
        msg = "X_test and y_test need to have the same size"
        raise ValueError(msg)

    prediction = self.predict(X_test)
    diff = np.count_nonzero(prediction - y_test)
    precision = 1 - diff / len(X_test)
    return precision
