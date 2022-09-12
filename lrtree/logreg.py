import numpy as np
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from lrtree.discretization import categorie_data_bin_train
from lrtree.discretization import categorie_data_bin_test


class PossiblyOneClassReg(sk.linear_model.LogisticRegression):
    """
    One class logistic regression (e.g. when a leaf is pure)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._single_class_label = None
        self.n_features_in_ = None

    def fit(self, X, y, weights=None):
        """
        Fit the one class regression: put the label of the single class and the number of features
        """
        if y.nunique() == 1:
            self._single_class_label = y.iloc[0]
            self.n_features_in_ = X.shape[1]
        else:
            return super().fit(X, y, weights)
        return self

    def predict(self, X):
        """
        Predict the single class
        """
        if self._single_class_label is not None:
            np.full(X.shape[0], self._single_class_label)
        else:
            return super().predict(X)

    def predict_proba(self, X):
        """
        Predict the single class with a 1 probability
        """
        if self._single_class_label is not None:
            if self._single_class_label == 1:
                return np.full((X.shape[0], 2), [0, 1])
            else:
                return np.full((X.shape[0], 2), [1, 0])
        else:
            return super().predict_proba(X)


class LogRegSegment(PossiblyOneClassReg):
    def __init__(self, **kwargs):
        super().__init__()
        self.discretization = False
        self.categories = None
        if 'discretization' in kwargs:
            self.discretization = kwargs['discretization']
            self.categories = {"enc": OneHotEncoder(), "merged_cat": {}, "discret_cat": {}}

    def fit(self, **kwargs):
        if self.discretization:
            train_data = kwargs['X'].rename(columns=self.column_names)
            train_data, labels, enc, merged_cat, discret_cat = categorie_data_bin_train(
                train_data, var_cible="y")
            self.categories["enc"] = enc
            self.categories["merged_cat"] = merged_cat
            self.categories["discret_cat"] = discret_cat

        return super().fit(**kwargs)

    def predict(self, X):
        if self.discretization:
            # Applies the data treatment for this leaf
            X = X.rename(columns=self.column_names)
            X = categorie_data_bin_test(X,
                                        self.categories["enc"],
                                        self.categories["merged_cat"],
                                        self.categories["discret_cat"])
        return super().predict(X.to_numpy())

    def predict_proba(self, X):
        if self.discretization:
            # Applies the data treatment for this leaf
            X = X.rename(columns=self.column_names)
            X = categorie_data_bin_test(X,
                                        self.categories["enc"],
                                        self.categories["merged_cat"],
                                        self.categories["discret_cat"])

        return super().predict_proba(X.to_numpy())
