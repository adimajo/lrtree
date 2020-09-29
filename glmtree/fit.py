"""
fit method for the Glmtree class
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier


def _dataset_split(self, X, y):
    """
    Splits the provided dataset into training, validation and test sets

    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    :param numpy.ndarray y:
        Boolean (0/1) labels of the observations. Must be of
        the same length as X
        (numpy "numeric" array).
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


def fit(self, X, y):
    """
        Fits the Glmtree object.

        .. todo:: Refactor due to complexity

        :param numpy.ndarray X:
            array_like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features
        :param numpy.ndarray y:
            Boolean (0/1) labels of the observations. Must be of
            the same length as X
            (numpy "numeric" array).
        """
    if len(y) != X.shape[0]:
        raise ValueError("X {} and y {} must be of the same length".format(X.shape, len(y)))
    self.n = len(y)

    types_data = [i.dtype in ("int32", "float64") for i in X]
    if sum(types_data) != len(types_data):
        raise ValueError("Unsupported data types. Columns of X must be int or float.")

    types_data = [i.dtype in ("int32", "float64") for i in y]
    if sum(types_data) != len(types_data):
        raise ValueError("Unsupported data types. Columns of y must be int or float.")

    if len(y.shape) != 1 or len(np.unique(y)) != 2:
        raise ValueError("y must be composed of one column with the values of two classes (categorical or numeric)")

    self._dataset_split(X, y)

    # Classification init
    current_best = 0
    criterion_iter = []
    link = []
    best_link = []
    best_logreg = None

    df = pd.DataFrame(X)
    df = df.add_prefix("par_")
    df["y"] = y
    df["c_map"] = np.random.randint(self.class_num, size=self.n)
    df["c_hat"] = df["c_map"]

    # Start of main logic
    for i in range(self.max_iter):
        logregs_c_hat = np.array([])
        logregs_c_map = np.array([])
        predictions_log = np.zeros(shape=(self.n, df["c_hat"].nunique()))

        # Getting p(y | x, c_hat) and filling the probs
        for c_iter in range(df["c_hat"].nunique()):
            idx = df["c_hat"] == np.sort(df["c_hat"].unique())[c_iter]
            train_data = df[idx]
            train_data = train_data.drop("c_map", axis=1)
            train_data = train_data.drop("c_hat", axis=1)

            formula = "y~" + "+".join(map(str, train_data.columns[train_data.columns != "y"].to_list()))
            logreg = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial()).fit()

            # Statsmodels was used because more simplicity of use of criterions
            logregs_c_hat = np.append(logregs_c_hat, logreg)
            predictions_log[:, c_iter] = logreg.predict(df)

        predictions_log[np.isnan(predictions_log)] = 0

        # Getting p(y | x, c_map) and total AIC calculation
        criterion_iter.append(np.zeros(2))

        for c_iter in range(df["c_map"].nunique()):
            idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
            train_data = df[idx]
            train_data = train_data.drop("c_map", axis=1)
            train_data = train_data.drop("c_hat", axis=1)

            formula = "y~" + "+".join(map(str, train_data.columns[train_data.columns != "y"].to_list()))
            logreg = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial()).fit()
            logregs_c_map = np.append(logregs_c_map, logreg)

            # TODO move criterion definition to other function
            if self.criterion == "aic":
                if not self.validation:
                    criterion_iter[i] = criterion_iter[i] - logregs_c_map[c_iter].aic
                else:
                    print(c_iter, np.sort(df["c_map"].unique()))
                    y_validate = df[df.index.isin(self.validate_rows)]["y"]
                    X_validate = df[df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
                    criterion_iter[i] = criterion_iter[i] + np.sum(
                        np.log(df.iloc[self.validate_rows, :][idx]["y"] * logregs_c_map[c_iter].predict(X_validate)
                               + (1 - y_validate) * (1 - y_validate * logregs_c_map[c_iter].predict(X_validate))))
            elif self.criterion == "bic":
                if not self.validation:

                    X_train = df[df.index.isin(self.train_rows)][idx][df.columns.difference(["y", "c_map", "c_hat"])]
                    print(X_train.shape, logregs_c_map[c_iter].params.shape)
                    criterion_iter[i] = criterion_iter[i] + 2 * logregs_c_map[c_iter].llf - \
                        np.log(len(X_train)) * len(logregs_c_map[c_iter].params)
                else:
                    y_validate = df[df.index.isin(self.validate_rows)][idx]["y"]
                    X_validate = df[df.index.isin(self.validate_rows)][idx][
                        df.columns.difference(["y", "c_map", "c_hat"])]
                    criterion_iter[i] = criterion_iter[i] + np.sum(np.log(
                        y_validate * logregs_c_map[c_iter].predict(X_validate) +
                        (1 - y_validate) * (1 - y_validate * logregs_c_map[c_iter].predict(X_validate))))
            elif not self.validation:
                print(df["y"].shape, logregs_c_map[c_iter].fittedvalues.shape)  # , criterion_iter[i].shape)
                a = np.hstack((df["y"], logregs_c_map[c_iter].fittedvalues))

                criterion_iter[i] = np.concatenate((criterion_iter[i], a))
            else:
                y_validate = df[df.index.isin(self.validate_rows)][idx]["y"]
                X_validate = df[df.index.isin(self.validate_rows)][idx][df.columns.difference(["y", "c_map", "c_hat"])]

                b = np.hstack((y_validate, logregs_c_map[c_iter].predict(X_validate)))
                print(criterion_iter[i])
                criterion_iter[i] = np.concatenate((criterion_iter[i], b))

        print("The", self.criterion, " criterion for iteration ", i, " is ", criterion_iter[i])

        # Burn in
        if i >= 50 and criterion_iter[i] > criterion_iter[current_best]:
            best_logreg = logregs_c_map
            best_link = link
            current_best = i

        # Getting p(c_hat | x)
        # TODO add different partition methods support
        # TODO add pass of control parameters
        if df["c_hat"].nunique() > 1:
            clf = DecisionTreeClassifier(max_depth=5).fit(X, df["c_hat"])
            link = clf
        else:
            break

        # c_map calculation
        df["c_map"] = np.argmax(link.predict_proba(X), axis=1)

        # choice of the new c_hat
        y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
        masked_predictions_log = np.ma.masked_array(predictions_log, mask=(1 - y_ext)).filled(0) + np.ma.masked_array(
            1 - predictions_log, mask=y_ext).filled(0)
        print(link.predict_proba(X).shape, masked_predictions_log.shape)
        matrix = np.multiply(link.predict_proba(X), masked_predictions_log)
        # matrix = link.predict_proba(X) * (df["y"] * predictions_log + (1 - df["y"]) * (1 - predictions_log))
        row_sums = matrix.sum(axis=1)
        p = matrix / row_sums[:, np.newaxis]
        # TODO refactor
        for j in range(len(df["c_hat"])):
            df["c_hat"][j] = np.random.choice(df["c_hat"].unique(), 1, p=p[j])

    self.best_link = best_link
    self.best_logreg = best_logreg
    self.criterion_iter = criterion_iter
    self.current_best = current_best
