import numpy as np
import pandas as pd
import pytest

import lrtree
from lrtree.fit import _segments


@pytest.fixture(scope="module")
def data():
    return lrtree.Lrtree.generate_data(1000, 4, seed=0)[:2]


def test_predict(data):
    X, y = data

    for discretize in (True, False):
        model = lrtree.Lrtree(criterion="aic", class_num=10, max_iter=20, discretize=discretize, random_state=0)
        model.fit(pd.DataFrame(X), y, nb_init=1, tree_depth=2)
        assert model.predict(pd.DataFrame(X)).shape == (1000,)
        probabilities = model.predict_proba(pd.DataFrame(X))
        assert probabilities.shape == (1000,)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert 0 <= model.precision(pd.DataFrame(X), y) <= 1
        with pytest.raises(ValueError):
            model.precision(pd.DataFrame(X), y[:10])


@pytest.mark.parametrize("leaves_as_segment", [False, True])
def test_predict_matches_the_segment_models(data, leaves_as_segment):
    """predict uses the segment (c_map) and the regression the fit used"""
    X, y = data
    model = lrtree.Lrtree(class_num=6, max_iter=25, burn_in=5, leaves_as_segment=leaves_as_segment, random_state=1)
    model.fit(X, y, tree_depth=3)
    segments = _segments(model.best_link, X, leaves_as_segment)
    assert set(segments) <= set(model.best_logreg) | {None}
    expected = np.empty(len(X))
    for label in np.unique(segments):
        expected[segments == label] = model.best_logreg[label].predict_proba(X[segments == label])[:, 1]
    np.testing.assert_allclose(model.predict_proba(X), expected)
    np.testing.assert_array_equal(model.predict(X), (expected > 0.5).astype(int))


def test_predict_on_data_missing_a_segment(data):
    X, y = data
    model = lrtree.Lrtree(class_num=6, max_iter=25, burn_in=5, random_state=1)
    model.fit(X, y, tree_depth=3)
    full = model.predict_proba(X)
    segments = _segments(model.best_link, X, False)
    assert len(np.unique(segments)) > 1
    # only the observations of the last segment: they must still be scored by *their* regression
    last = segments == np.unique(segments)[-1]
    np.testing.assert_allclose(model.predict_proba(X[last]), full[last])
    np.testing.assert_allclose(model.predict_proba(X[:1]), full[:1])


def test_predict_returns_original_labels():
    X, y, _, _ = lrtree.Lrtree.generate_data(500, 4, seed=0)
    model = lrtree.Lrtree(class_num=3, max_iter=5, burn_in=0, random_state=0)
    model.fit(X, np.where(y == 1, 7, 3), tree_depth=2)
    assert set(np.unique(model.predict(X))) <= {3, 7}


def test_predict_with_dataframe_column_names(data):
    X, y = data
    model = lrtree.Lrtree(class_num=4, max_iter=10, burn_in=0, random_state=0)
    model.fit(pd.DataFrame(X, columns=list("abcd")), y, tree_depth=2)
    assert model.column_names == list("abcd")
    np.testing.assert_allclose(model.predict_proba(pd.DataFrame(X, columns=list("abcd"))), model.predict_proba(X))
