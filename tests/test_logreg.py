import numpy as np
import pytest

from lrtree.logreg import LogRegSegment, PossiblyOneClassReg


def test_single_class_predict_returns_labels():
    X = np.zeros((5, 3))
    for label in (0, 1):
        model = PossiblyOneClassReg().fit(X, np.full(5, label))
        assert model.predict(X).tolist() == [label] * 5
        assert model.predict_proba(X)[:, label].tolist() == [1.] * 5
        assert model.n_features_in_ == 3


def test_refit_forgets_single_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    model = PossiblyOneClassReg().fit(X, np.ones(100))
    model.fit(X, (X[:, 0] > 0).astype(int))
    assert model._single_class_label is None
    assert 0 < model.predict_proba(X)[:, 1].mean() < 1


def test_sample_weight_is_used():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
    unweighted = PossiblyOneClassReg().fit(X, y)
    weights = np.where(y == 1, 5., 1.)
    weighted = PossiblyOneClassReg().fit(X, y, sample_weight=weights)
    assert weighted.intercept_[0] > unweighted.intercept_[0]


def test_logreg_segment_forwards_parameters():
    model = LogRegSegment(solver="saga", C=0.01, tol=1e-2)
    assert (model.solver, model.C, model.tol) == ("saga", 0.01, 1e-2)
    assert model.get_params()["C"] == 0.01
    assert model.get_params()["discretization"] is False
    assert "LogRegSegment" in repr(model)
    with pytest.raises(TypeError):
        LogRegSegment(not_a_parameter=1)


@pytest.mark.parametrize("discretization", [False, True])
def test_logreg_segment_fit_predict(discretization):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] ** 2 + 0.3 * rng.normal(size=400) > 1).astype(int)
    model = LogRegSegment(discretization=discretization).fit(X, y)
    assert model.predict_proba(X).shape == (400, 2)
    assert model.predict(X).shape == (400,)
    if discretization:
        # the quadratic effect is only reachable through the discretisation of x0
        assert (model.predict(X) == y).mean() > 0.8


def test_logreg_segment_pure_with_discretization():
    X = np.random.default_rng(0).normal(size=(50, 2))
    model = LogRegSegment(discretization=True).fit(X, np.zeros(50))
    assert model.predict(X).tolist() == [0.] * 50
