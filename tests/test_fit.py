import numpy as np
import pandas as pd
import pytest
import lrtree


def test_args_fit():
    n = 1000
    d = 4
    X, y, _, _ = lrtree.Lrtree.generate_data(n, d)

    with pytest.raises(ValueError):
        lrtree.Lrtree.generate_data(n, d, seed=1, theta=np.ones(5))

    model = lrtree.Lrtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10, max_iter=50)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=False, leaves_as_segment=True, validation=False, criterion="aic", ratios=(0.7,),
                          class_num=10, max_iter=50)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=False, validation=True, criterion="aic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=False, validation=True, criterion="gini", ratios=(0.7,), class_num=10, max_iter=200)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(algo="EM", test=False, validation=True, criterion="gini", ratios=(0.7,), class_num=10,
                          max_iter=40)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=False, validation=True, criterion="bic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=False, validation=False, criterion="bic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=False, validation=False, criterion="gini", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=True, validation=True, criterion="gini", ratios=(0.4, 0.3), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model = lrtree.Lrtree(test=True, validation=False, criterion="bic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    with pytest.raises(ValueError):
        model = lrtree.Lrtree(test=False, validation=False, criterion="gini", ratios=(0.7,), class_num=10,
                              max_iter=1, data_treatment=True)
        model.fit(X, y, nb_init=1, tree_depth=2)
    # model = lrtree.Lrtree(test=False, validation=True, criterion="gini", ratios=(0.7,), class_num=10,
    #                       max_iter=1, data_treatment=True)
    # model.fit(pd.DataFrame(X), y, nb_init=1, tree_depth=2)
    # model = lrtree.Lrtree(test=False, validation=False, criterion="gini", ratios=(0.7,), class_num=10,
    #                       max_iter=1, data_treatment=True)
    # model.fit(pd.DataFrame(X), y, nb_init=1, tree_depth=2)
    lrtree.fit._fit_func(fit_kwargs={'X': X, 'y': y})
    lrtree.fit._fit_func(class_kwargs={'test': False}, fit_kwargs={'X': X, 'y': y})
    lrtree.fit._fit_parallelized(fit_kwargs={'X': X, 'y': y})


def test_dataset_length():
    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.zeros(shape=(1001, 1))
        model = lrtree.Lrtree()
        model.fit(X, y, nb_init=1, tree_depth=2)
    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.zeros(shape=(1000, 2))
        model = lrtree.Lrtree()
        model.fit(X, y, nb_init=1, tree_depth=2)


def test_data_type():
    n = 1000
    d = 4
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), dtype=str)

    with pytest.raises(ValueError):
        X = np.random.choice(alphabet, [n, d])
        y = np.zeros(shape=(1000, 1))
        model = lrtree.Lrtree()
        model.fit(X, y, nb_init=1, tree_depth=2)

    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.random.choice(alphabet, [n, d])
        model = lrtree.Lrtree()
        model.fit(X, y, nb_init=1, tree_depth=2)


def test_split():
    n = 1000
    d = 4
    X, y, _, _ = lrtree.Lrtree.generate_data(n, d)

    model = lrtree.Lrtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10, max_iter=1)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model.fit(pd.DataFrame(X), pd.Series(y), nb_init=1, tree_depth=2)

    with pytest.raises(ValueError):
        lrtree.Lrtree.generate_data(n, d, theta="toto")


def test_not_fit():
    model = lrtree.Lrtree(test=False, validation=False, criterion="aic", ratios=(0.7,), class_num=10, max_iter=1)
    with pytest.raises(lrtree.NotFittedError):
        model.predict(X=None)
    with pytest.raises(ValueError):
        lrtree.Lrtree(algo=4)
    with pytest.raises(ValueError):
        lrtree.Lrtree(algo="toto")
    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=(1.1,))
    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=(-0.1,))
    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=(0.8, 0.8))
    with pytest.raises(ValueError):
        lrtree.Lrtree(validation=True, test=False, ratios=(0.3, 0.3))


def test_oneclass():
    n = 1000
    d = 4
    X, _, _, _ = lrtree.Lrtree.generate_data(n, d)
    y = np.ones(n)
    one_class = lrtree.logreg.LogRegSegment()
    one_class.fit(X=pd.DataFrame(X), y=pd.Series(y))
    one_class.predict(pd.DataFrame(X))
    one_class.predict_proba(pd.DataFrame(X))
