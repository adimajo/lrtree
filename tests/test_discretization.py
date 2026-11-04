import numpy as np
import pandas as pd
import pytest

from lrtree.discretization import Processing, apply_discretization, extreme_values, grouping


def test_apply_discretization():
    X = pd.DataFrame({"a": [-5., 0., 0.5, 3., 10.]})
    result = apply_discretization(X, "a", np.array([2., 0.]))
    assert result["a"].tolist() == [1, 1, 2, 3, 3]
    assert np.issubdtype(result["a"].dtype, np.integer)


def test_grouping_merges_similar_levels_only():
    rng = np.random.default_rng(0)
    n = 3000
    levels = rng.choice(["a", "b", "c"], n)
    # "a" and "b" have the same rate of positives, "c" a very different one
    y = (rng.random(n) < np.where(levels == "c", 0.9, 0.1)).astype(int)
    X = pd.DataFrame({"g": levels, "y": y})
    grouped, mapping = grouping(X, "g", "y")
    assert mapping["a"] == mapping["b"] != mapping["c"]
    assert grouped["g"].nunique() == 2
    # the input is left untouched
    assert X["g"].nunique() == 3


def test_extreme_values():
    X = pd.DataFrame({"a": [1., 999., np.nan, 4.], "b": ["NR", "x", "y", "z"]})
    cleaned = extreme_values(X)
    assert cleaned["a"].tolist() == [1., 0., 0., 4.]
    assert cleaned["b"].tolist() == ["missing", "x", "y", "z"]
    assert X["a"].iloc[1] == 999.


@pytest.mark.parametrize("discretize", [False, True])
def test_processing_numerical(discretize):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    X["y"] = (X["a"] + rng.normal(size=300) > 0).astype(int)
    processing = Processing(target="y", discretize=discretize)
    train = processing.fit_transform(X)
    assert len(train) == 300
    if not discretize:
        assert train.shape[1] == 3
    else:
        assert train.shape[1] > 3
    test = processing.transform(X)
    np.testing.assert_allclose(test.to_numpy(), train.to_numpy())
    # the target column is optional when transforming
    assert processing.transform(X.drop(columns="y")).shape == test.shape


def test_processing_categorical_and_unknown_level():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"g": rng.choice(list("abcd"), 500), "x": rng.normal(size=500)})
    X["y"] = ((X["g"] == "a") ^ (rng.random(500) < 0.1)).astype(int)
    processing = Processing(target="y", discretize=False)
    train = processing.fit_transform(X, ["g"])
    assert train.shape[0] == 500
    new = X.iloc[:5].copy()
    new["g"] = "never seen"
    assert processing.transform(new).shape[0] == 5
    with pytest.raises(ValueError):
        Processing(target="y").fit(X, ["missing"])
