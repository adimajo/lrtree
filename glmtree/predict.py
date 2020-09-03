

def predict(self, X):
    """
    Predicts the labels and c-hat values
    for new values using previously fitted glmtree object

    :param numpy.array X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    """
    return self.best_logreg.predict(X)

def predict_proba(self, X):
    """
    Predicts the labels and c-hat values
    for new values using previously fitted glmtree object

    :param numpy.array X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    """
    return self.best_logreg.predict_proba(X)
