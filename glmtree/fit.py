"""
fit method for the Glmtree class
"""
import glmtree
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from copy import deepcopy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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
    if 'numpy' in str(type(X)):
        types_data = [i.dtype in ("int32", "float64") for i in X]
    else:
        types_data = [X[i].dtype in ("int32", "float64") for i in X.columns]
    if sum(types_data) != len(types_data):
        msg = "Unsupported data types. Columns of X must be int or float."
        logger.error(msg)
        raise ValueError(msg)

    if 'numpy' in str(type(y)):
        types_data = [i.dtype in ("int32", "float64") for i in y]
    else:
        types_data = [y.dtype in ("int32", "float64")]
    # types_data = [i.dtype in ("int32", "float64") for i in y]
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


def _calculate_logreg_poids_c(df, proportion, c_iter):
    """Modèle de regression logisitque sur la feuille c_iter, en utilisant les éléments de df
    avec leur proportion d'appartenance à cette feuille"""

    weights = proportion[:, c_iter]
    y = df['y']
    X = df.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
    try:
        # With L2 reg, a small C leads to a big regularisation
        model = LogisticRegression(random_state=0, C=1000)
        model_results = model.fit(X, y, weights)
    except PerfectSeparationError as e:
        msg = "Perfect separation in one of the leaves: cannot go further."
        logger.error(msg)
        raise e
    return model_results, model


def _calculate_logreg_c(df, c, c_iter, L1_wt=0, cnvrg_tol=1e-2, start_params=None):
    idx = df[c] == np.sort(df[c].unique())[c_iter]
    train_data = df[idx]
    train_data = train_data.drop("c_map", axis=1)
    train_data = train_data.drop("c_hat", axis=1)
    formula = "y~" + "+".join(map(str, train_data.columns[train_data.columns != "y"].to_list()))
    try:
        model = smf.glm(formula=formula, data=train_data, family=sm.families.Binomial())
        model_results = model.fit_regularized(alpha=0.001, L1_wt=L1_wt, start_params=start_params, cnvrg_tol=cnvrg_tol)
    except PerfectSeparationError as e:
        msg = "Perfect separation in one of the leaves: cannot go further."
        logger.error(msg)
        raise e
    return idx, model_results, model


def _calculate_criterion(self, df, logregs_c_map, model, c_iter, i, idx):
    if self.criterion == "aic":
        if not self.validation:
            self.criterion_iter[i] = self.criterion_iter[i] \
                                     + 2 * model[c_iter].loglike(logregs_c_map[c_iter].params) \
                                     - len(logregs_c_map[c_iter].params)
        else:
            y_validate = df[idx & df.index.isin(self.validate_rows)]["y"]
            X_validate = df[idx & df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
            self.criterion_iter[i] = self.criterion_iter[i] + np.sum(
                np.log(df.loc[idx & df.index.isin(self.validate_rows), :]["y"] * logregs_c_map[c_iter].predict(
                    X_validate) + (1 - y_validate) * (1 - y_validate * logregs_c_map[c_iter].predict(X_validate))))
    elif self.criterion == "bic":  # On calcule -BIC, qu'on va maximiser
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



def fit(self, X, y, nb_init=1, tree_depth=10):
    """Fits the Glmtree object.

        :param numpy.ndarray X:
            array_like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features
        :param numpy.ndarray y:
            Boolean (0/1) labels of the observations. Must be of
            the same length as X
            (numpy "numeric" array).
        :param int nb_init:
            Number of different random initializations
        :param int tree_depth:
            Maximum depth of the tree used
        """
    _check_args(X, y)

    self.n = len(y)

    _dataset_split(self)

    params = ["Intercept"] + ["par_" + str(k) for k in range(X.shape[1])]

    if self.algo == 'SEM':
        for k in range(nb_init):
            # Classification init
            X_copy = np.copy(X)
            link = []
            self.criterion_iter = []
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
                # penalty='l1', solver='liblinear'
                models = [LogisticRegression(penalty='l2', C=0.001, tol=1e-2, warm_start=True) for k in
                          range(df["c_hat"].nunique())]

                for c_iter in range(df["c_hat"].nunique()):
                    y = df['y']
                    X = df.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                    model = models[c_iter]
                    logreg = model.fit(X, y)
                    models[c_iter] = model

                    logregs_c_hat = np.append(logregs_c_hat, deepcopy(logreg))
                    to_predict = df.drop(['y', 'c_hat', 'c_map'], axis=1)
                    to_predict = to_predict.to_numpy()
                    predictions_log[:, c_iter] = logreg.predict(to_predict)


                predictions_log[np.isnan(predictions_log)] = 0

                # Getting p(y | x, c_map) and total AIC calculation
                self.criterion_iter.append([0])

                for c_iter in range(df["c_map"].nunique()):
                    # Statsmodels was used because more simplicity of use of criterions
                    idx, logreg, model = _calculate_logreg_c(df, "c_map", c_iter, L1_wt=1)
                    logregs_c_map = np.append(logregs_c_map, deepcopy(logreg))
                    model_c_map = np.append(model_c_map, deepcopy(model))
                    _calculate_criterion(self, df, logregs_c_map, model_c_map, c_iter, i, idx)

                # logger.info("The " + self.criterion + " criterion for iteration " + str(i) + " is: " + str(self.criterion_iter[i]))

                # Burn in
                if i >= 50 and self.criterion_iter[i] > self.best_criterion:
                    self.best_logreg = logregs_c_map
                    self.best_link = link
                    self.best_criterion = self.criterion_iter[i]

                # Getting p(c_hat | x)
                # TODO add different partition methods support
                # TODO add pass of control parameters
                if df["c_hat"].nunique() > 1:
                    clf = DecisionTreeClassifier(max_depth=tree_depth).fit(X, df["c_hat"])
                    link = clf
                else:
                    logger.info("The tree has only its root! Premature end of algorithm.")
                    break

                # c_map calculation
                df["c_map"] = np.argmax(link.predict_proba(X), axis=1)

                # choice of the new c_hat
                y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
                masked_predictions_log = np.ma.masked_array(predictions_log, mask=(1 - y_ext)).filled(0) \
                                         + np.ma.masked_array(1 - predictions_log, mask=y_ext).filled(0)
                matrix = np.multiply(link.predict_proba(X), masked_predictions_log)
                row_sums = matrix.sum(axis=1)
                p = matrix / row_sums[:, np.newaxis]
                # # TODO: vectorized multinouilli
                df["c_hat"] = _vectorized_multinouilli(p, df["c_hat"].unique())

    elif self.algo == 'EM':
        for k in range(nb_init):
            # Random initialisation
            X_copy = np.copy(X)
            self.criterion_iter = []
            df = pd.DataFrame(X_copy)
            df = df.add_prefix("par_")
            df["y"] = y
            # Ajout d'un coefficient constant
            # df["constant_coeff"] = np.ones(self.n)
            df["c_map"] = np.zeros(self.n)
            df["c_hat"] = df["c_map"]  # Not used in this case
            random_init = np.random.random((len(df), self.class_num))
            row_sums = random_init.sum(axis=1)
            proportion = random_init / row_sums[:, np.newaxis]
            # If penalty ='l1', solver='liblinear' or 'saga', default=’lbfgs’
            models=[LogisticRegression(penalty='l1', C=0.001,solver='liblinear', tol=1e-6, warm_start=True) for k in range(self.class_num)]

            # MCMC steps
            for i in range(self.max_iter):
                logregs_c_hat = np.array([])
                logregs_c_map = np.array([])
                model_c_map = np.array([])
                predictions_log = np.zeros(shape=(self.n, self.class_num))

                # Getting p(y | x, c_hat) and filling the probabilities/proportions

                for c_iter in range(self.class_num):
                    # logreg, _ = _calculate_logreg_poids_c(df, proportion, c_iter)
                    weights = proportion[:, c_iter]
                    y = df['y']
                    X = df.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                    model = models[c_iter]
                    logreg = model.fit(X, y, weights)
                    models[c_iter]=model

                    # Statsmodels was used because more simplicity of use of criterions
                    logregs_c_hat = np.append(logregs_c_hat, logreg)
                    to_predict = df.drop(['y', 'c_hat', 'c_map'], axis=1)
                    to_predict = to_predict.to_numpy()
                    predictions_log[:, c_iter] = logreg.predict_proba(to_predict)[:, 1]

                predictions_log[np.isnan(predictions_log)] = 0

                # Getting p(c | x)
                weights = []
                train_data = []
                leaf = []
                for k in range(len(df)):
                    for j in range(self.class_num):
                        leaf.append(j)
                        train_data.append(X[k])
                        weights.append(proportion[k][j])
                clf = DecisionTreeClassifier(max_depth=tree_depth)
                clf.fit(train_data, leaf, weights)
                link = clf

                # New matric of proportions p_theta_1|x
                y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
                # p_theta_y|x
                masked_predictions_log = np.ma.masked_array(predictions_log, mask=(1 - y_ext)).filled(0) \
                                         + np.ma.masked_array(1 - predictions_log, mask=y_ext).filled(0)
                # p_beta_c|x * p_theta_y|x
                matrix = np.multiply(link.predict_proba(X), masked_predictions_log)
                row_sums = matrix.sum(axis=1)
                proportion = matrix / row_sums[:, np.newaxis]

                # c_map calculation
                df["c_map"] = np.argmax(link.predict_proba(X), axis=1)

                # Getting p(y | x, c_map) and total BIC calculation
                self.criterion_iter.append([0])

                # Burn in
                if i >= 50:
                    for c_iter in range(df["c_map"].nunique()):
                        idx, logreg, model = _calculate_logreg_c(df, "c_map", c_iter, L1_wt=1, cnvrg_tol=1e-8)
                        logregs_c_map = np.append(logregs_c_map, logreg)
                        model_c_map = np.append(model_c_map, model)
                        _calculate_criterion(self, df, logregs_c_map, model_c_map, c_iter, i, idx)

                    # Best results
                    if self.criterion_iter[i] > self.best_criterion:
                        self.best_logreg = logregs_c_map
                        self.best_link = link
                        self.best_criterion = self.criterion_iter[i]


def fit_func(X, y, algo='SEM', max_iter=100, tree_depth=5, class_num=10):
    """Creates the glmtree model and fits it to the data
            :param numpy.ndarray X:
                array_like of shape (n_samples, n_features)
                Vector to be scored, where `n_samples` is the number of samples and
                `n_features` is the number of features
            :param numpy.ndarray y:
                Boolean (0/1) labels of the observations. Must be of
                the same length as X
                (numpy "numeric" array).
            :param int max_iter:
                Number of MCMC steps to perform.
            :param int tree_depth:
                Maximum depth of the tree used
            :param int class_num:
                Number of initial discretization intervals for all variables."""
    model = glmtree.Glmtree(algo=algo, test=False, validation=False, criterion="aic", ratios=(0.7,),
                            class_num=class_num,
                            max_iter=max_iter)
    model.fit(X, y, tree_depth=tree_depth)
    return model


def fit_parralized(X, y, algo='SEM', nb_init=5, nb_jobs=-1, max_iter=100, tree_depth=5, class_num=10):
    """A fit function which creates tge model and fits it, where the random initilisations are parallized
            :param numpy.ndarray X:
                array_like of shape (n_samples, n_features)
                Vector to be scored, where `n_samples` is the number of samples and
                `n_features` is the number of features
            :param numpy.ndarray y:
                Boolean (0/1) labels of the observations. Must be of
                the same length as X
                (numpy "numeric" array).
            :param int nb_init:
                Number of different random initializations
            :param int nb_jobs:
                Number of jobs for the Parallelization
                Default : -1, all CPU are used
            :param int max_iter:
                Number of MCMC steps to perform.
            :param int tree_depth:
                Maximum depth of the tree used
            :param int class_num:
                Number of initial discretization intervals for all variables."""
    models = Parallel(n_jobs=nb_jobs)(
        delayed(fit_func)(X, y, algo=algo, max_iter=max_iter, tree_depth=tree_depth, class_num=class_num) for k in
        range(nb_init))
    critere = -np.inf
    best_model = None
    for k in range(nb_init):
        model = models[k]
        criterion=model.best_criterion
        # if type(criterion) is not float or int :
        #     criterion=criterion[0]
        if criterion > critere:
            best_model = model
            critere = model.best_criterion[0]
    return best_model
