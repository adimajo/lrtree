"""Smoke tests of the glmtree benchmark port: tiny settings, the full run is manual (benchmarks/real_data.py)"""
import numpy as np
import pytest

from benchmarks import common


def test_generate_glmtree_data():
    X, y = common.generate_glmtree_data(301, "tree", np.random.default_rng(0))
    assert X.shape == (301, 3)
    assert set(X["x1"].unique()) == {"1", "2", "3", "4", "5", "6"}
    assert set(np.unique(y)) == {0, 1}
    X, y = common.generate_glmtree_data(300, "no tree", np.random.default_rng(0))
    assert list(X.columns) == ["x1", "x2"]
    with pytest.raises(ValueError):
        common.generate_glmtree_data(300, "toto", np.random.default_rng(0))


def test_prepare_handles_unseen_levels_and_missing_values():
    import pandas as pd
    train = pd.DataFrame({"a": [1., 2., np.nan, 4., 5., 6.], "b": ["u", "v", "u", "w", None, "u"]})
    test = pd.DataFrame({"a": [np.nan, 3.], "b": ["never seen", "v"]})
    X_train, X_test = common.prepare(train, test)
    assert X_train.shape[1] == X_test.shape[1] == 1 + 3  # a + 4 levels (u, v, w, missing) minus the first
    assert not np.isnan(X_train).any() and not np.isnan(X_test).any()


def test_run_repetition_and_reports():
    result = common.run_repetition("pima", 1, ["logit", "tree", "lrtree"], sem_iter=4)
    assert list(result["method"]) == ["logit", "tree", "lrtree"]
    assert result["gini"].between(0, 1).all(), result["note"].tolist()
    summary = common.summarize(result)
    assert summary.loc[summary["method"] == "logit", "diff_vs_logit"].iloc[0] == 0
    comparison = common.compare_with_glmtree(result)
    assert set(comparison["glmtree_method"]) == {"logit", "ctree", "glmtree"}
    assert common.qualitative_checks(summary) == [("pima: lrtree does not beat the logistic regression "
                                                   "(as in glmtree)", True)]
