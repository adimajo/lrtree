"""
Predict, predict_proba and precision methods for the Lrtree class
"""
import pandas as pd
import numpy as np
from copy import deepcopy
from lrtree.discretization import bin_data_cate_test


def _predict(self, X: np.ndarray, fun: str) -> np.ndarray:
    self._check_is_fitted()
    # List of models for each class
    link = self.best_link
    logreg = self.best_logreg
    # Predicted class for each sample
    if not link:
        liste_cla = [0]
        classes = np.zeros(len(X))
    else:
        if self.data_treatment:
            enc_global = self.best_treatment["global"]
            classes = link.predict(bin_data_cate_test(X, enc_global, self.categorical))
        else:
            classes = link.predict(X)
        # Classes that were predicted : c_map
        liste_cla = np.unique(classes)

    X_df = pd.DataFrame(X).copy()
    if not isinstance(X, pd.DataFrame):
        X_df = X_df.add_prefix("par_")

    X_df["class"] = classes
    X_df["pred"] = 0
    for i in range(len(liste_cla)):
        filtre = X_df["class"] == liste_cla[i]
        bloc = deepcopy(X_df[filtre].drop(["class", "pred"], axis=1))
        # if self.data_treatment:
        # treatment = self.best_treatment
        # bloc = _categorie_data_bin_test(bloc.rename(columns=self.column_names),
        #                                 treatment[liste_cla[i]]["enc"],
        #                                 treatment[liste_cla[i]]["scaler"],
        #                                 treatment[liste_cla[i]]["merged_cat"],
        #                                 treatment[liste_cla[i]]["discret_cat"])
        if fun == "predict":
            bloc_pred = logreg[i].predict(bloc)
        else:
            bloc_pred = logreg[i].predict_proba(bloc)
        k = 0
        for j in range(len(X_df)):
            if filtre[j]:
                if fun == "predict":
                    X_df.loc[j, "pred"] = bloc_pred[k]
                else:
                    X_df.loc[j, "pred"] = bloc_pred[k][1]
                k = k + 1

    return X_df["pred"].to_numpy()


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
    diff = np.count_nonzero(prediction - y_test)
    return 1 - diff / len(X_test)
