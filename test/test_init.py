import pytest
import numpy as np
import glmtree


def test_init():
    glmtree_instance = glmtree.Glmtree(validation=True, criterion="gini")
    assert not glmtree_instance.test
    assert glmtree_instance.validation
    assert glmtree_instance.criterion == "gini"
    assert glmtree_instance.ratios == (0.7,)
    assert glmtree_instance.class_num == 10
    assert glmtree_instance.max_iter == 100
    assert glmtree_instance.train_rows is None
    assert glmtree_instance.validate_rows is None
    assert glmtree_instance.test_rows is None
    assert glmtree_instance.n == 0


def test_test_arg():
    glmtree.Glmtree(test=True)
    glmtree.Glmtree(test=False)

    with pytest.raises(ValueError):
        glmtree.Glmtree(test="string")

    with pytest.raises(ValueError):
        glmtree.Glmtree(test=12)

    with pytest.raises(ValueError):
        glmtree.Glmtree(test=0.1)


def test_validation_arg():
    glmtree.Glmtree(validation=True)
    glmtree.Glmtree(validation=False)

    with pytest.raises(ValueError):
        glmtree.Glmtree(validation="string")

    with pytest.raises(ValueError):
        glmtree.Glmtree(validation=12)

    with pytest.raises(ValueError):
        glmtree.Glmtree(validation=0.1)


def test_ratios():
    glmtree.Glmtree(ratios=(0.1, 0.5), validation=True, test=True)
    glmtree.Glmtree(ratios=(0.1,), validation=True, test=False)
    glmtree.Glmtree(ratios=(0.1,), validation=False, test=True)
    glmtree.Glmtree(ratios=(0.1,), validation=False, test=False)
    glmtree.Glmtree(validation=False, test=False)

    with pytest.raises(ValueError):
        glmtree.Glmtree(ratios=(0.1,), validation=True, test=True)

    with pytest.raises(ValueError):
        glmtree.Glmtree(ratios=0.1)

    with pytest.raises(ValueError):
        glmtree.Glmtree(ratios=0.1)

    with pytest.raises(ValueError):
        glmtree.Glmtree(ratios=0.1)


def test_criterion():
    glmtree.Glmtree(criterion="aic")
    glmtree.Glmtree(criterion="bic")
    glmtree.Glmtree(criterion="gini")

    with pytest.raises(ValueError):
        glmtree.Glmtree(criterion="wrong criterion")


def test_gini_penalized(caplog):
    glmtree.Glmtree(validation=False,
                    test=True,
                    criterion="gini")
    assert caplog.records[0].message == "Using Gini index on training set might yield an overfitted model."


def test_validation_criterion(caplog):
    glmtree.Glmtree(validation=True,
                    criterion="aic")

    assert caplog.records[0].message == ("No need to penalize the log-likelihood when a validation set is used. Using "
                                         "log-likelihood " \
                                         "instead of AIC/BIC. ")

    glmtree.Glmtree(validation=True,
                    criterion="bic")

    assert caplog.records[0].message == (
        "No need to penalize the log-likelihood when a validation set is used. Using log-likelihood " \
        "instead of AIC/BIC. ")
