import pandas as pd
import pytest
import lrtree


def test_predict():
    n = 1000
    d = 4
    X, y, _, _ = lrtree.Lrtree.generate_data(n, d)

    # model = lrtree.Lrtree(test=False, validation=False, criterion="aic", class_num=10, max_iter=50,
    #                       data_treatment=True)
    # model.fit(pd.DataFrame(X), y, nb_init=1, tree_depth=2)
    # model.predict(pd.DataFrame(X))
    # model.predict_proba(pd.DataFrame(X))
    # model.precision(pd.DataFrame(X), y)

    model = lrtree.Lrtree(test=False, validation=False, criterion="aic", class_num=10, max_iter=50,
                          data_treatment=False)
    model.fit(pd.DataFrame(X), y, nb_init=1, tree_depth=2)
    model.predict(pd.DataFrame(X))
    model.predict_proba(pd.DataFrame(X))
    model.precision(pd.DataFrame(X), y)
    with pytest.raises(ValueError):
        model.precision(pd.DataFrame(X), y[:10])
