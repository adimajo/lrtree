import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

import lrtree
from lrtree import fit as fit_module
from lrtree.logreg import LogRegSegment, PossiblyOneClassReg


@pytest.fixture(scope="module")
def data():
    return lrtree.Lrtree.generate_data(1000, 4, seed=0)[:2]


def test_args_fit(data):
    X, y = data
    kwargs = dict(class_num=10, random_state=0)

    for algo, validation, test, criterion, ratios, max_iter in [
        ("SEM", False, False, "aic", (0.7,), 50),
        ("SEM", True, False, "aic", (0.7,), 1),
        ("SEM", True, False, "gini", (0.7,), 40),
        ("EM", True, False, "gini", (0.7,), 10),
        ("EM", False, False, "bic", (0.7,), 10),
        ("SEM", True, False, "bic", (0.7,), 1),
        ("SEM", False, False, "bic", (0.7,), 1),
        ("SEM", False, False, "gini", (0.7,), 1),
        ("SEM", True, True, "gini", (0.4, 0.3), 1),
        ("SEM", False, True, "bic", (0.7,), 1),
    ]:
        model = lrtree.Lrtree(algo=algo, test=test, validation=validation, criterion=criterion, ratios=ratios,
                              max_iter=max_iter, **kwargs)
        model.fit(X, y, nb_init=1, tree_depth=2)
        if max_iter > 1:
            assert model.best_link is not None
            assert model.best_logreg
            assert model.predict(X).shape == (1000,)
        else:
            # no tree has been fitted yet when the criterion of the first iteration is computed
            with pytest.raises(lrtree.NotFittedError):
                model.predict(X)

    with pytest.raises(ValueError):
        model = lrtree.Lrtree(criterion="gini", max_iter=1, data_treatment=True)
        model.fit(X, y, nb_init=1, tree_depth=2)
    for validation in (True, False):
        model = lrtree.Lrtree(validation=validation, criterion="gini", max_iter=1, data_treatment=True,
                              random_state=0)
        model.fit(pd.DataFrame(X), y, nb_init=1, tree_depth=2)


def test_fit_func_and_parallelized(data):
    X, y = data
    class_kwargs = {"class_num": 4, "max_iter": 3, "random_state": 0}
    fit_kwargs = {"X": X, "y": y, "tree_depth": 2}
    model = lrtree.fit._fit_func(class_kwargs, fit_kwargs)
    assert isinstance(model, lrtree.Lrtree)
    best = lrtree.fit._fit_parallelized(class_kwargs, fit_kwargs, nb_init=2, nb_jobs=1)
    assert isinstance(best, lrtree.Lrtree)
    assert best.best_link is not None


def test_dataset_length():
    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.zeros(shape=(1001, 1))
        lrtree.Lrtree().fit(X, y, nb_init=1, tree_depth=2)
    with pytest.raises(ValueError):
        X = np.zeros(shape=(1000, 4))
        y = np.zeros(shape=(1000, 2))
        lrtree.Lrtree().fit(X, y, nb_init=1, tree_depth=2)


def test_data_type():
    n = 1000
    d = 4
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), dtype=str)
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        lrtree.Lrtree().fit(rng.choice(alphabet, [n, d]), np.zeros(shape=(1000, 1)))

    with pytest.raises(ValueError):
        lrtree.Lrtree().fit(np.zeros(shape=(1000, 4)), rng.choice(alphabet, [n, d]))

    with pytest.raises(ValueError):
        lrtree.Lrtree().fit(pd.DataFrame({"a": rng.choice(alphabet, n), "b": rng.normal(size=n)}),
                            rng.integers(0, 2, n))

    with pytest.raises(ValueError):
        lrtree.Lrtree().fit(np.zeros((n, d)), np.arange(n))


def test_check_args_encodes_labels():
    X = np.random.default_rng(0).normal(size=(10, 2))
    y, classes = fit_module._check_args(X, np.array([1, 2] * 5))
    assert y.tolist() == [0, 1] * 5
    assert classes.tolist() == [1, 2]
    # a Series with an index that is not 0..n-1 does not matter: only the values are used
    y, _ = fit_module._check_args(X, pd.Series([1, 2] * 5, index=range(100, 110)))
    assert y.tolist() == [0, 1] * 5


def test_split(data):
    X, y = data

    model = lrtree.Lrtree(criterion="aic", class_num=10, max_iter=1, random_state=0)
    model.fit(X, y, nb_init=1, tree_depth=2)
    model.fit(pd.DataFrame(X), pd.Series(y, index=range(500, 1500)), nb_init=1, tree_depth=2)

    with pytest.raises(ValueError):
        lrtree.Lrtree.generate_data(1000, 4, theta="toto")


def test_split_sizes(data):
    X, y = data
    model = lrtree.Lrtree(test=True, validation=True, ratios=(0.5, 0.3), max_iter=1, random_state=0)
    model.fit(X, y, tree_depth=2)
    assert (len(model.train_rows), len(model.validate_rows), len(model.test_rows)) == (500, 300, 200)
    assert len(np.unique(np.concatenate([model.train_rows, model.validate_rows, model.test_rows]))) == 1000


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
    X = lrtree.Lrtree.generate_data(100, 4, seed=0)[0]
    one_class = PossiblyOneClassReg()
    one_class.fit(pd.DataFrame(X), pd.Series(np.ones(100)))
    assert one_class.predict(X).tolist() == [1.] * 100
    assert one_class.predict_proba(X)[:, 1].tolist() == [1.] * 100


def test_reproducible(data):
    X, y = data
    first = lrtree.Lrtree(class_num=5, max_iter=15, random_state=3)
    second = lrtree.Lrtree(class_num=5, max_iter=15, random_state=3)
    first.fit(X, y, tree_depth=2)
    second.fit(X, y, tree_depth=2)
    assert first.criterion_iter == second.criterion_iter
    assert first.best_criterion == second.best_criterion
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))


def test_refit_resets_state(data):
    X, y = data
    model = lrtree.Lrtree(class_num=5, max_iter=12, random_state=0)
    model.fit(X, y, tree_depth=2)
    first_criterion = model.best_criterion
    # A second fit on pure noise must not keep the (much better) model of the first one
    model.fit(X, np.random.default_rng(1).integers(0, 2, len(y)), tree_depth=2)
    assert model.best_criterion < first_criterion
    assert len(model.criterion_iter) <= 12


def test_import_does_not_touch_global_state():
    before = list(warnings.filters)
    importlib.reload(fit_module)
    assert warnings.filters == before


def test_fit_does_not_leak_warning_filters(data):
    X, y = data
    before = list(warnings.filters)
    lrtree.Lrtree(class_num=3, max_iter=2, random_state=0).fit(X, y, tree_depth=2)
    assert warnings.filters == before


def test_segments_are_labels_not_positions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    labels = np.where(X[:, 0] > 0, 5, 2)
    link = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, labels)
    assert set(np.unique(fit_module._segments(link, X, leaves_as_segment=False))) == {2, 5}
    leaves = fit_module._segments(link, X, leaves_as_segment=True)
    assert set(leaves) == set(np.flatnonzero(link.tree_.children_left == -1))


def test_draw_c_hat_uses_tree_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    # a noisy segmentation: the leaves of a stump are impure, so the likelihood of y has a say in the draw
    c_hat = np.where((X[:, 0] > 0) ^ (rng.random(500) < 0.4), 6, 1)
    y = rng.integers(0, 2, 500)
    link = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, c_hat)
    # P(y = 1 | x, c) of the 8 possible segments
    predictions = rng.uniform(0.01, 0.99, size=(500, 8))
    tree_pred, drawn = fit_module._draw_c_hat(link, X, y, predictions, np.random.default_rng(1))
    assert tree_pred.shape == (500, 2)
    assert set(np.unique(drawn)) <= {1, 6}
    # a deterministic likelihood must be honoured: segment 1 explains y = 1 only, segment 6 y = 0 only
    predictions = np.zeros((500, 8))
    predictions[:, 1] = 1
    predictions[:, 6] = 0
    _, drawn = fit_module._draw_c_hat(link, X, y, predictions, np.random.default_rng(1))
    assert np.all(drawn[y == 1] == 1)
    assert np.all(drawn[y == 0] == 6)


def test_vectorized_multinouilli():
    rng = np.random.default_rng(0)
    probabilities = np.tile([0.2, 0.3, 0.5], (20000, 1))
    drawn = fit_module._vectorized_multinouilli(probabilities, np.array([10, 20, 30]), rng)
    np.testing.assert_allclose([(drawn == i).mean() for i in (10, 20, 30)], [0.2, 0.3, 0.5], atol=0.02)
    # rows summing to a bit less than one (rounding) must not index past the last level
    assert set(fit_module._vectorized_multinouilli(np.full((1000, 2), 0.4999), np.array([0, 1]), rng)) <= {0, 1}


def _fitted_segments(rows_labels, y):
    """Models fitted on the given segments, keyed by label"""
    model = lrtree.Lrtree(class_num=2, max_iter=1, random_state=0)
    model.n = len(y)
    model.train_rows = np.arange(len(y))
    model.validate_rows = None
    X = np.random.default_rng(0).normal(size=(len(y), 2))
    return model, X, fit_module._fit_segments(model, X, y, rows_labels)


def test_calc_criterion_bic_per_segment_and_missing_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    segments = np.repeat([0, 1, 2, 3], 100)
    model, X, models = _fitted_segments(segments, y)
    model.criterion = "bic"
    fallback = models[0]
    full = fit_module._calc_criterion(model, X, y, segments, models, fallback)

    expected = 0.
    for label in range(4):
        rows = segments == label
        p = models[label].predict_proba(X[rows])[:, 1]
        ll = np.sum(y[rows] * np.log(p) + (1 - y[rows]) * np.log(1 - p))
        expected -= -2 * ll + np.log(rows.sum()) * (np.count_nonzero(models[label].coef_) + 1)
    assert full == pytest.approx(expected)

    # Segment 3 has no model: its rows are scored with the fallback, the other segments are unaffected
    without = {label: m for label, m in models.items() if label != 3}
    value = fit_module._calc_criterion(model, X, y, segments, without, fallback)
    assert np.isfinite(value)
    assert value != pytest.approx(full)


def test_calc_criterion_validation_is_log_likelihood():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    segments = np.repeat([0, 1], 200)
    model, X, models = _fitted_segments(segments, y)
    model.validation, model.validate_rows, model.criterion = True, np.arange(100, 300), "aic"
    value = fit_module._calc_criterion(model, X, y, segments, models, models[0])
    p = np.concatenate([models[0].predict_proba(X[100:200])[:, 1], models[1].predict_proba(X[200:300])[:, 1]])
    yv = y[100:300]
    assert value == pytest.approx(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    model.criterion = "gini"
    assert -1 <= fit_module._calc_criterion(model, X, y, segments, models, models[0]) <= 1


def test_fit_tree_keeps_depth_and_picks_best_pruning():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3))
    c_hat = (X[:, 0] > 0).astype(int) + 2 * (rng.random(600) < 0.1)
    model = lrtree.Lrtree(validation=True, criterion="gini", random_state=0)
    model.n, model.tree_depth, model.min_impurity_decrease = 600, 4, 0.0
    model.min_samples_leaf, model.optimal_size = 1, True
    model.train_rows, model.validate_rows = np.arange(400), np.arange(400, 600)
    model._rng = np.random.default_rng(0)
    link = fit_module._fit_tree(model, X, c_hat)
    assert link.get_depth() <= 4
    full = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[:400], c_hat[:400])
    assert link.score(X[400:], c_hat[400:]) >= full.score(X[400:], c_hat[400:])


@pytest.mark.parametrize("rule", [lrtree.LOW_IMPROVEMENT, lrtree.LOW_VARIATION, lrtree.CHANGED_SEGMENTS])
def test_early_stopping_rules_stop(data, rule):
    X, y = data
    model = lrtree.Lrtree(class_num=4, max_iter=60, burn_in=3, early_stopping=rule, random_state=0)
    model.fit(X, y, tree_depth=2, tol=0.5)
    assert len(model.criterion_iter) < 60


def test_no_early_stopping_runs_all_iterations(data):
    X, y = data
    model = lrtree.Lrtree(class_num=4, max_iter=15, burn_in=3, random_state=0)
    model.fit(X, y, tree_depth=2, tol=0.5)
    assert len(model.criterion_iter) == 15


def test_unknown_early_stopping_warns(caplog):
    model = lrtree.Lrtree(early_stopping=["low variation", "toto"])
    assert model.early_stopping == [lrtree.LOW_VARIATION]
    assert "toto" in caplog.records[-1].message
    assert lrtree.Lrtree(early_stopping="toto").early_stopping == []
    assert lrtree.Lrtree(early_stopping=True).early_stopping == [
        lrtree.LOW_IMPROVEMENT, lrtree.LOW_VARIATION, lrtree.CHANGED_SEGMENTS]


def test_em_runs_exactly_max_iter(data):
    X, y = data
    model = lrtree.Lrtree(algo="EM", class_num=3, max_iter=4, random_state=0)
    model.fit(X, y, tree_depth=2)
    assert len(model.criterion_iter) == 4
    assert model.predict_proba(X).shape == (1000,)


def test_min_samples_leaf_is_forwarded(data):
    X, y = data
    model = lrtree.Lrtree(class_num=4, max_iter=5, burn_in=0, random_state=0)
    model.fit(X, y, tree_depth=6, min_samples_leaf=100)
    assert model.best_link.min_samples_leaf == 100
    assert model.best_link.tree_.n_node_samples[model.best_link.tree_.children_left == -1].min() >= 100


def test_logreg_kwargs_are_applied(data):
    X, y = data
    model = lrtree.Lrtree(class_num=3, max_iter=3, burn_in=0, random_state=0,
                          logreg_kwargs={"C": 0.01, "max_iter": 50})
    model.fit(X, y, tree_depth=2)
    assert all(m.C == 0.01 and m.max_iter == 50 for m in model.best_logreg.values())


def test_models_are_not_shared_between_iterations(data):
    # the models kept as best must not be refitted by the following iterations
    X, y = data
    model = lrtree.Lrtree(class_num=4, max_iter=30, burn_in=0, random_state=0)
    model.fit(X, y, tree_depth=2)
    before = {label: m.coef_.copy() for label, m in model.best_logreg.items() if hasattr(m, "coef_")}
    model.predict_proba(X)
    for label, coef in before.items():
        np.testing.assert_array_equal(model.best_logreg[label].coef_, coef)
    assert all(isinstance(m, LogRegSegment) for m in model.best_logreg.values())
