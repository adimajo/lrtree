import pytest
import lrtree


def test_init():
    lrtree_instance = lrtree.Lrtree(validation=True, criterion="gini")
    assert not lrtree_instance.test
    assert lrtree_instance.validation
    assert lrtree_instance.criterion == "gini"
    assert lrtree_instance.ratios == (0.7,)
    assert lrtree_instance.class_num == 10
    assert lrtree_instance.max_iter == 100
    assert lrtree_instance.train_rows is None
    assert lrtree_instance.validate_rows is None
    assert lrtree_instance.test_rows is None
    assert lrtree_instance.n == 0


def test_test_arg():
    lrtree.Lrtree(test=True)
    lrtree.Lrtree(test=False)

    with pytest.raises(ValueError):
        lrtree.Lrtree(test="string")

    with pytest.raises(ValueError):
        lrtree.Lrtree(test=12)

    with pytest.raises(ValueError):
        lrtree.Lrtree(test=0.1)


def test_validation_arg():
    lrtree.Lrtree(validation=True)
    lrtree.Lrtree(validation=False)

    with pytest.raises(ValueError):
        lrtree.Lrtree(validation="string")

    with pytest.raises(ValueError):
        lrtree.Lrtree(validation=12)

    with pytest.raises(ValueError):
        lrtree.Lrtree(validation=0.1)


def test_ratios():
    lrtree.Lrtree(ratios=(0.1, 0.5), validation=True, test=True)
    lrtree.Lrtree(ratios=(0.1,), validation=True, test=False)
    lrtree.Lrtree(ratios=(0.1,), validation=False, test=True)
    lrtree.Lrtree(ratios=(0.1,), validation=False, test=False)
    lrtree.Lrtree(validation=False, test=False)

    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=(0.1,), validation=True, test=True)

    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=0.1)

    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=0.1)

    with pytest.raises(ValueError):
        lrtree.Lrtree(ratios=0.1)


def test_criterion():
    lrtree.Lrtree(criterion="aic")
    lrtree.Lrtree(criterion="bic")
    lrtree.Lrtree(criterion="gini")

    with pytest.raises(ValueError):
        lrtree.Lrtree(criterion="wrong criterion")


def test_early_stopping():
    lrtree.Lrtree(early_stopping=["low variation"])
    lrtree.Lrtree(early_stopping="low variation")

    with pytest.raises(ValueError):
        lrtree.Lrtree(early_stopping=[])
    with pytest.raises(ValueError):
        lrtree.Lrtree(early_stopping=['toto'])
    with pytest.raises(ValueError):
        lrtree.Lrtree(early_stopping='toto')


def test_gini_penalized(caplog):
    lrtree.Lrtree(validation=False,
                    test=True,
                    criterion="gini")
    assert caplog.records[0].message == "Using Gini index on training set might yield an overfitted model."


def test_validation_criterion(caplog):
    lrtree.Lrtree(validation=True,
                    criterion="aic")

    assert caplog.records[0].message == ("No need to penalize the log-likelihood when a validation set is used. Using "
                                         "log-likelihood "
                                         "instead of AIC/BIC.")

    lrtree.Lrtree(validation=True,
                    criterion="bic")

    assert caplog.records[0].message == (
        "No need to penalize the log-likelihood when a validation set is used. Using log-likelihood "
        "instead of AIC/BIC.")
