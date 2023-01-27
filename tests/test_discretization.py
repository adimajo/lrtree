import numpy as np

import lrtree
from scripts.realdataopen import get_adult_data


def test_discretization():
    original_train, original_val, original_test, labels_train, labels_val, labels_test, categorical = get_adult_data(
        target="Target", seed=0)
    for discretize in [True, False]:
        processing = lrtree.discretization.Processing(target="Target", discretize=discretize)
        X_train = processing.fit_transform(X=original_train,
                                           categorical=categorical)
        X_test = processing.transform(original_test)
        assert X_train.shape[1] == X_test.shape[1]
        if discretize:
            assert np.logical_or(X_train.values == 0, X_train.values == 1).all()
            assert np.logical_or(X_test.values == 0, X_test.values == 1).all()
