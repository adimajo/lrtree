import lrtree
from scripts.realdataopen import get_adult_data


def test_discretization():
    processing = lrtree.discretization.Processing(target="Target", discretize=True)
    original_train, original_test, labels_train, labels_test, categorical = get_adult_data()
    X_train = processing.fit_transform(X=original_train,
                                       categorical=categorical)
    X_test = processing.transform(original_test)
