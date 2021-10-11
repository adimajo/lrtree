"""
fit method for the Glmtree class
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def _check_args(X, y):
    """
    Checks input arguments shape.

    :raises: ValueError
    """
    if len(y) != X.shape[0]:
        msg = "X {} and y {} must be of the same length".format(X.shape, len(y))
        logger.error(msg)
        raise ValueError(msg)

    types_data = [i.dtype in ("int32", "float64") for i in X]
    if sum(types_data) != len(types_data):
        msg = "Unsupported data types. Columns of X must be int or float."
        logger.error(msg)
        raise ValueError(msg)

    types_data = [i.dtype in ("int32", "float64") for i in y]
    if sum(types_data) != len(types_data):
        msg = "Unsupported data types. Columns of y must be int or float."
        logger.error(msg)
        raise ValueError(msg)

    if len(y.shape) != 1 or len(np.unique(y)) != 2:
        msg = "y must be composed of one column with the values of two classes (categorical or numeric)"
        logger.error(msg)
        raise ValueError(msg)


def _dataset_split(self):
    """
    Splits the provided dataset into training, validation and test sets
    """
    fst_idx = int(self.ratios[0] * self.n)
    if self.validation and self.test:
        snd_idx = int((self.ratios[0] + self.ratios[1]) * self.n)
        self.train_rows, self.validate_rows, self.test_rows = np.split(np.random.choice(self.n, self.n, replace=False),
                                                                       [fst_idx, snd_idx])
    elif self.validation:
        self.train_rows, self.validate_rows = np.split(np.random.choice(self.n, self.n, replace=False), [fst_idx])
    elif self.test:
        self.train_rows, self.test_rows = np.split(np.random.choice(self.n, self.n, replace=False), [fst_idx])
    else:
        self.train_rows = np.random.choice(self.n, self.n, replace=False)


def _calculate_logreg_c(df, c, c_iter):
    idx = df[c] == np.sort(df[c].unique())[c_iter]
    train_data = df[idx]
    train_data = train_data.drop("c_map", axis=1)
    train_data = train_data.drop("c_hat", axis=1)
    formula = "y~" + "+".join(map(str, train_data.columns[train_data.columns != "y"].to_list()))
    try:
        model = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial())
        model_results = model.fit_regularized(alpha=0.0001, L1_wt=0)
    except PerfectSeparationError as e:
        msg = "Perfect separation in one of the leaves: cannot go further."
        logger.error(msg)
        raise e
    return idx, model_results, model


def _calculate_criterion(self, df, logregs_c_map, model, c_iter, i, idx):
    if self.criterion == "aic":
        if not self.validation:
            self.criterion_iter[i] = self.criterion_iter[i] + 2 * model[c_iter].loglike(logregs_c_map[c_iter].params) - len(logregs_c_map[c_iter].params)
        else:
            y_validate = df[idx & df.index.isin(self.validate_rows)]["y"]
            X_validate = df[idx & df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
            self.criterion_iter[i] = self.criterion_iter[i] + np.sum(
                np.log(df.loc[idx & df.index.isin(self.validate_rows), :]["y"] * logregs_c_map[c_iter].predict(
                    X_validate) + (1 - y_validate) * (1 - y_validate * logregs_c_map[c_iter].predict(X_validate))))
    elif self.criterion == "bic": #On calcule -BIC, qu'on va maximiser
        if not self.validation:
            X_train = df[idx & df.index.isin(self.train_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
            self.criterion_iter[i] = self.criterion_iter[i] + 2 * model[c_iter].loglike(logregs_c_map[c_iter].params) - \
                np.log(len(X_train)) * len(logregs_c_map[c_iter].params)

        else:
            y_validate = df[idx & df.index.isin(self.validate_rows)]["y"]
            X_validate = df[idx & df.index.isin(self.validate_rows)][
                df.columns.difference(["y", "c_map", "c_hat"])]
            self.criterion_iter[i] = self.criterion_iter[i] + np.sum(np.log(
                y_validate * logregs_c_map[c_iter].predict(X_validate) +
                (1 - y_validate) * (1 - y_validate * logregs_c_map[c_iter].predict(X_validate))))
    elif not self.validation:
        a = np.hstack((df["y"], logregs_c_map[c_iter].fittedvalues))
        self.criterion_iter[i] = np.concatenate((self.criterion_iter[i], a))
    else:
        y_validate = df[idx & df.index.isin(self.validate_rows)]["y"]
        X_validate = df[idx & df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
        b = np.hstack((y_validate, logregs_c_map[c_iter].predict(X_validate)))
        self.criterion_iter[i] = np.concatenate((self.criterion_iter[i], b))


def _vectorized_multinouilli(prob_matrix, items):
    """
    A vectorized version of multinouilli sampling.
    .. todo:: check that the number of columns of prob_matrix is the same as the number of elements in items
    :param prob_matrix: A probability matrix of size n (number of training
        examples) * m[j] (the factor levels to sample from).
    :type prob_matrix: numpy.array
    :param list items: The factor levels to sample from.
    :returns: The drawn factor levels for each observation.
    :rtype: numpy.array
    """

    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0]).reshape((-1, 1))
    k = (s < r).sum(axis=1)
    return items[k]


def fit(self, X, y, nb_init):
    """
        Fits the Glmtree object.

        :param numpy.ndarray X:
            array_like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features
        :param numpy.ndarray y:
            Boolean (0/1) labels of the observations. Must be of
            the same length as X
            (numpy "numeric" array).
        """
    _check_args(X, y)

    self.n = len(y)

    _dataset_split(self)

    for k in range(nb_init):
        # Classification init
        X_copy = np.copy(X)
        link = []
        self.criterion_iter=[]
        df = pd.DataFrame(X_copy)
        df = df.add_prefix("par_")
        df["y"] = y
        df["c_map"] = np.random.randint(self.class_num, size=self.n)
        df["c_hat"] = df["c_map"]

        # Start of main logic
        for i in range(self.max_iter):
            logregs_c_hat = np.array([])
            logregs_c_map = np.array([])
            model_c_map = np.array([])
            predictions_log = np.zeros(shape=(self.n, df["c_hat"].nunique()))

            # Getting p(y | x, c_hat) and filling the probs
            for c_iter in range(df["c_hat"].nunique()):
                _, logreg, _ = _calculate_logreg_c(df, "c_hat", c_iter)

                # Statsmodels was used because more simplicity of use of criterions
                logregs_c_hat = np.append(logregs_c_hat, logreg)
                predictions_log[:, c_iter] = logreg.predict(df)

            predictions_log[np.isnan(predictions_log)] = 0

            # Getting p(y | x, c_map) and total AIC calculation
            self.criterion_iter.append([0])

            for c_iter in range(df["c_map"].nunique()):
                idx, logreg, model = _calculate_logreg_c(df, "c_map", c_iter)
                logregs_c_map = np.append(logregs_c_map, logreg)
                model_c_map=np.append(model_c_map, model)
                _calculate_criterion(self, df, logregs_c_map, model_c_map, c_iter, i, idx)

            #logger.info("The " + self.criterion + " criterion for iteration " + str(i) + " is: " + str(self.criterion_iter[i]))

            # Burn in
            if i >= 50 and self.criterion_iter[i] > self.best_criterion :
                self.best_logreg = logregs_c_map
                self.best_link = link
                self.best_criterion = self.criterion_iter[i]


            # Getting p(c_hat | x)
            # TODO add different partition methods support
            # TODO add pass of control parameters
            if df["c_hat"].nunique() > 1:
                clf = DecisionTreeClassifier(max_depth=4).fit(X, df["c_hat"])
                link = clf
            else:
                logger.info("The tree has only its root! Premature end of algorithm.")
                break

            # c_map calculation
            df["c_map"] = np.argmax(link.predict_proba(X), axis=1)

            # choice of the new c_hat
            y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
            masked_predictions_log = np.ma.masked_array(predictions_log, mask=(1 - y_ext)).filled(0) + np.ma.masked_array(
                1 - predictions_log, mask=y_ext).filled(0)
            matrix = np.multiply(link.predict_proba(X), masked_predictions_log)
            row_sums = matrix.sum(axis=1)
            p = matrix / row_sums[:, np.newaxis]
            # # TODO: vectorized multinouilli
            df["c_hat"] = _vectorized_multinouilli(p, df["c_hat"].unique())
