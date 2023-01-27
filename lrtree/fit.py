"""
fit module for the Lrtree class
"""
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import lrtree
from lrtree import LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS
from lrtree.discretization import bin_data_cate_train
from lrtree.logreg import LogRegSegment

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

STOPPED_AT_ITERATION = "Stopped at iteration"

pd.options.mode.chained_assignment = None


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
        types_data = [i.dtype in ("int32", "int64", "float32", "float64") for i in X]
    else:
        types_data = [X[i].dtype in ("int32", "int64", "float32", "float64") for i in X.columns]
    if sum(types_data) != len(types_data):
        msg = "Unsupported data types. Columns of X must be int or float."
        logger.error(msg)
        raise ValueError(msg)

    if 'numpy' in str(type(y)):
        types_data = [i.dtype in ("int32", "int64", "float32", "float64") for i in y]
    else:
        types_data = [y.dtype in ("int32", "int64", "float32", "float64")]
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
    Splits the provided dataset into training, validation (to calculate the BIC, and choose the best model) and
    test sets
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


def _calc_criterion(self, df: pd.DataFrame, model_c_map: list, treatment: dict = None) -> float:
    """
    Computes the criterion for this model.

    :param pandas.DataFrame df: The dataframe of the data used
    :param list model_c_map: The list of the models for each leaf.
    :param dict treatment: The dictionary of the data treatment for each leaf.
    :param dict self.column_names: The dictionary of the real column names.
    :returns: The criteria.
    :rtype: float
    """
    lengths_pred = []
    # Computing the Area Under Curve of the ROC curve, which we maximise
    y_true = []
    y_proba = []
    k = 0
    for c_iter in np.unique(df["c_map"]):
        model = model_c_map[k]
        idx = df["c_map"] == c_iter
        if self.validation:
            validate_rows = idx & df.index.isin(self.validate_rows)
            y_validate = df[validate_rows]["y"].tolist()
            X_validate = df[validate_rows].drop(['y', 'c_hat', 'c_map'], axis=1)
        else:
            train_rows = idx & df.index.isin(self.train_rows)
            y_validate = df[train_rows]["y"].tolist()
            X_validate = df[train_rows].drop(['y', 'c_hat', 'c_map'], axis=1)
        if X_validate.shape[0] > 0:
            y_pred = model.predict_proba(X_validate)[:, 1]
            lengths_pred.append(X_validate.shape[0])
            y_true = [*y_true, *y_validate]
            y_proba = [*y_proba, *y_pred]
        k = k + 1
    if self.criterion == "gini":
        return 2 * roc_auc_score(y_true, y_proba) - 1
    elif self.criterion == "aic":
        return - 2 * log_loss(y_true, y_proba, normalize=False, labels=[0, 1]) - np.sum(
            [model.n_features_in_ for model in model_c_map])
    elif self.criterion == "bic":
        return - 2 * log_loss(y_true, y_proba, normalize=False, labels=[0, 1]) - np.sum(
            [np.log(lengths_pred[index]) * model.n_features_in_ for index, model in enumerate(model_c_map)])


def _vectorized_multinouilli(prob_matrix: np.array, items: list) -> np.array:
    """
    A vectorized version of multinouilli sampling.

    :param prob_matrix: A probability matrix of size n (number of training
        examples) * m[j] (the factor levels to sample from).
    :type prob_matrix: numpy.array
    :param list items: The factor levels to sample from.
    :returns: The drawn factor levels for each observation.
    :rtype: numpy.ndarray
    """
    if len(items) != prob_matrix.shape[1]:  # pragma: no cover
        msg = "The number of factors {} and the number of columns of prob_matrix {} must be of the same length".format(
            len(items), prob_matrix.shape)
        logger.error(msg)
        raise ValueError(msg)

    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0]).reshape((-1, 1))
    k = np.sum((s < r), axis=1)
    return items[k]


def find_leaves(link, X):
    """
    Find the leaf "number" for each sample in X

    :param sklearn.tree.DecisionTreeClassifier link: the segmentation model
    :param numpy.ndarray X: the samples x features
    :rtype: numpy.ndarray
    """
    the_tree = link.tree_
    n_nodes = the_tree.node_count
    children_left = the_tree.children_left
    children_right = the_tree.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    return link.decision_path(X)[:, is_leaves]


def _draw_c_hat(link, X_tree, df, predictions_log, c_iter_to_keep):
    tree_pred = link.predict_proba(X_tree)
    y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
    masked_predictions_log = np.ma.masked_array(predictions_log, mask=(1 - y_ext)).filled(0) + np.ma.masked_array(
        1 - predictions_log, mask=y_ext).filled(0)
    matrix = np.multiply(tree_pred, masked_predictions_log[:, c_iter_to_keep])
    row_sums = matrix.sum(axis=1)
    p = matrix / row_sums[:, np.newaxis]
    return tree_pred, _vectorized_multinouilli(p, df["c_hat"].unique()[c_iter_to_keep])


def _fit_tree(self, X_tree, df):
    train_rows = df.index.isin(self.train_rows)
    X = X_tree[train_rows].to_numpy()
    # Building the tree
    clf = DecisionTreeClassifier(max_depth=self.tree_depth, min_impurity_decrease=self.min_impurity_decrease).fit(
        X, df[train_rows]["c_hat"])
    link = clf

    if self.optimal_size and self.validation:
        validate_rows = df.index.isin(self.validate_rows)
        X_validate = X_tree[validate_rows].to_numpy()
        path = clf.cost_complexity_pruning_path(X, df[train_rows]["c_hat"])
        alphas = path.ccp_alphas

        # Tree propositions, with more or less pruning
        best_score = 0
        # Starts from the most complete tree, pruning while it improves the accuracy on the
        # validation test
        for a in range(len(alphas)):
            alpha = alphas[a]
            tree = DecisionTreeClassifier(ccp_alpha=alpha)
            tree.fit(X, df[train_rows]["c_hat"])
            score = tree.score(X_validate, df[validate_rows]["c_hat"])
            # Choosing the tree with the best accuracy on the validation set
            if score > best_score:
                link = tree
    return link


def _fit_sem(self, df, X_tree, models, treatment, i=0, stopping_criterion=False):
    link = []
    df["c_map"] = np.random.randint(self.class_num, size=self.n)
    df["c_hat"] = df["c_map"]

    # Start of main logic
    with tqdm(total=self.max_iter, leave=False, desc="Iterations",
              disable=os.environ.get("TQDM_DISABLE", "False").lower() in ('true', '1', 't')) as pbar:
        while i < self.max_iter and not stopping_criterion:
            # logger.error(f"best: {self.best_link}")
            logger.debug(f"Iteration {i}")
            logregs_c_hat = []
            logregs_c_map = []
            model_c_map = []
            predictions_log = np.zeros(shape=(self.n, df["c_hat"].nunique()))
            c_iter_to_keep = np.ones(predictions_log.shape[1], dtype=bool)
            # Renumbering
            dict_of_values = {v: k for k, v in enumerate(np.unique(df["c_hat"]))}
            df.replace({"c_hat": dict_of_values}, inplace=True)

            # Getting p(y | x, c_hat) and filling the probabilities
            for index, c_iter in enumerate(np.unique(df["c_hat"])):
                idx = df["c_hat"] == c_iter
                train_data = df[idx & df.index.isin(self.train_rows)].drop(['y', 'c_map', 'c_hat'], axis=1)
                if train_data.shape[0] == 0:
                    logger.debug(f"No training data for c_iter {c_iter}, skipping.")
                    c_iter_to_keep[index] = False
                    continue
                y = df[idx & df.index.isin(self.train_rows)]['y']
                models[c_iter].fit(X=train_data, y=y,
                                   categorical=self.categorical)
                logregs_c_hat.append(models[c_iter])
                to_predict = df.drop(['y', 'c_hat', 'c_map'], axis=1)
                predictions_log[:, c_iter] = models[c_iter].predict_proba(to_predict)[:, 1]

            predictions_log[np.isnan(predictions_log)] = 0

            # Getting p(y | x, c_map)
            self.criterion_iter.append(0)
            for c_iter in np.unique(df["c_map"]):
                idx = df["c_map"] == c_iter
                train_data = df[idx & df.index.isin(self.train_rows)]
                y = train_data['y']
                X = train_data.drop(['y', 'c_map', 'c_hat'], axis=1)
                model = LogRegSegment(penalty='l1', solver=self.solver, C=1e-2, tol=1e-2,
                                      warm_start=True, data_treatment=self.data_treatment,
                                      discretization=self.discretization,
                                      column_names=self.column_names)
                logreg = model.fit(X=X, y=y, categorical=self.categorical)
                logregs_c_map.append(logreg)
                model_c_map.append(model)

            # Getting the total criterion, for this model (tree + reg) proposition - Best model yet
            stopping_criterion = _update_criterion(self, i, treatment, df, model_c_map, logregs_c_map, link)

            # Getting p(c_hat | x)
            if df["c_hat"].nunique() > 1:
                link = _fit_tree(self, X_tree, df)
            else:
                logger.info("The tree has only its root! Premature end of algorithm.")
                break
            # logger.error(f"link: {link}")

            # Choice of the new c_hat = random step
            tree_pred, df["c_hat"] = _draw_c_hat(link, X_tree, df, predictions_log, c_iter_to_keep)

            # c_map calculation
            if self.leaves_as_segment:
                path = find_leaves(link, X_tree.to_numpy())
                new_cmap = np.asarray(path.argmax(axis=1)).ravel()
            else:
                new_cmap = tree_pred.argmax(axis=1)
            prop_changed_segments = 1 - np.sum(new_cmap == df['c_map'].values) / len(new_cmap)
            logger.info(f"Proportion of changed segments: {prop_changed_segments:.3}")

            if CHANGED_SEGMENTS in self.early_stopping and i >= self.burn_in and prop_changed_segments <= self.tol:
                # TODO: does not take into account relabelling
                stopping_criterion = True
                logger.info(f"{STOPPED_AT_ITERATION} {i}, the proportion of changed segments is below 'tol'.")
            df["c_map"] = new_cmap
            logger.info(f"Number of distinct segments: {df['c_map'].nunique()}")
            pbar.update(1)
            i += 1


def _fit_em(self, df, models, i=0, stopping_criterion=False):
    df["c_map"] = np.zeros(self.n)
    df["c_hat"] = df["c_map"]  # Not used in this case
    random_init = np.random.random((len(df), self.class_num))
    row_sums = random_init.sum(axis=1)
    proportion = random_init / row_sums[:, np.newaxis]

    # MCMC steps
    with tqdm(total=self.max_iter, leave=False,
              disable=os.environ.get("TQDM_DISABLE", "False").lower() in ('true', '1', 't')) as pbar:
        while i < self.max_iter and not stopping_criterion:
            logregs_c_hat = []
            logregs_c_map = []
            model_c_map = []
            predictions_log = np.zeros(shape=(self.n, self.class_num))

            # Getting p(y | x, c_hat) and filling the probabilities/proportions
            for c_iter in range(self.class_num):
                weights = proportion[:, c_iter]
                y = df['y']
                X = df.drop(['y', 'c_map', 'c_hat'], axis=1)
                model = models[c_iter]
                logreg = model.fit(X=X, y=y, weights=weights)
                models[c_iter] = model
                logregs_c_hat = np.append(logregs_c_hat, logreg)
                to_predict = df.drop(['y', 'c_hat', 'c_map'], axis=1)
                predictions_log[:, c_iter] = logreg.predict_proba(to_predict)[:, 1]

            predictions_log[np.isnan(predictions_log)] = 0

            # Getting p(c | x)
            weights = []
            train_data = []
            leaf = []
            for q in range(len(df)):
                for j in range(self.class_num):
                    leaf.append(j)
                    train_data.append(X.iloc[q, :])
                    weights.append(proportion[q][j])
            clf = DecisionTreeClassifier(max_depth=self.tree_depth, min_impurity_decrease=self.min_impurity_decrease)
            clf.fit(train_data, leaf, weights)
            link = clf

            # New matric of proportions p_theta_1|x
            y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
            # p_theta_y|x
            masked_predictions_log = np.ma.masked_array(predictions_log, mask=(
                1 - y_ext)).filled(0) + np.ma.masked_array(1 - predictions_log, mask=y_ext).filled(0)
            # p_beta_c|x * p_theta_y|x
            matrix = np.multiply(link.predict_proba(X), masked_predictions_log)
            row_sums = matrix.sum(axis=1)
            proportion = matrix / row_sums[:, np.newaxis]

            # c_map calculation
            df["c_map"] = np.argmax(link.predict_proba(X), axis=1)

            # Getting p(y | x, c_map) and total BIC calculation
            self.criterion_iter.append([0])

            for c_iter in range(df["c_map"].nunique()):
                idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                train_data = df[idx]
                y = train_data['y']
                X = train_data.drop(['y', 'c_map', 'c_hat'], axis=1)
                model = LogRegSegment(penalty='l2', solver=self.solver, C=1e-2, tol=1e-3, warm_start=True)
                logreg = model.fit(X=X, y=y)

                logregs_c_map.append(logreg)
                model_c_map.append(model)

            # Getting the total criterion
            self.criterion_iter[i] = _calc_criterion(self, df, model_c_map)
            # Best results
            if self.criterion_iter[i] > self.best_criterion:
                # Stopping when the criterion doesn't really get better anymore
                if self.criterion == "gini" and i >= self.burn_in and abs(
                        self.criterion_iter[i] - self.best_criterion) < self.tol:
                    stopping_criterion = True
                    logger.info(f"{STOPPED_AT_ITERATION} {i}")
                if self.criterion != "gini" and i >= self.burn_in and abs(
                        self.criterion_iter[i] - self.best_criterion) < self.tol * abs(self.best_criterion):
                    stopping_criterion = True
                    logger.info(f"{STOPPED_AT_ITERATION} {i}")

                self.best_logreg = logregs_c_map
                self.best_link = link
                self.best_criterion = self.criterion_iter[i]

            # Stopping when the criterion doesn't vary anymore
            if i >= self.burn_in:
                last_ones = self.criterion_iter[-10: -1]
                variation = np.var(last_ones) ** 0.5
                if self.criterion == "gini" and variation < self.tol:
                    stopping_criterion = True
                    logger.info(f"{STOPPED_AT_ITERATION} {i}")
                if self.criterion != "gini" and variation < self.tol * abs(self.best_criterion):
                    stopping_criterion = True
                    logger.info(f"{STOPPED_AT_ITERATION} {i}")

            pbar.update(1)
            i += 1


def _update_best(self, i, treatment, df, logregs_c_map, link):
    stopping_criterion = False
    # Stopping when the criterion doesn't really get better anymore
    if LOW_IMPROVEMENT in self.early_stopping and self.criterion == "gini" and i >= self.burn_in and abs(
            self.criterion_iter[i] - self.best_criterion) < self.tol:
        stopping_criterion = True
        logger.info(f"{STOPPED_AT_ITERATION} {i}")
    if LOW_VARIATION in self.early_stopping and self.criterion != "gini" and i >= self.burn_in and abs(
            self.criterion_iter[i] - self.best_criterion) < self.tol * abs(self.best_criterion):
        stopping_criterion = True
        logger.info(f"{STOPPED_AT_ITERATION} {i}")

    best_treat = {}
    if self.data_treatment:
        best_treat = {"global": treatment["global"]}
    self.best_treatment = deepcopy(best_treat)
    self.best_logreg = logregs_c_map
    self.best_link = link
    self.best_criterion = self.criterion_iter[i]
    return stopping_criterion


def _update_criterion(self, i, treatment, df, model_c_map, logregs_c_map, link):
    self.criterion_iter[i] = _calc_criterion(self, df, model_c_map, treatment)
    logger.debug(f"{self.criterion} at iteration {i} is {self.criterion_iter[i]:.3}.")

    stopping_criterion = False
    if self.criterion_iter[i] > self.best_criterion:
        stopping_criterion = _update_best(self, i, treatment, df, logregs_c_map, link)
    # Stopping when we reach a tree with only one leaf
    if i > 0 and link == []:
        stopping_criterion = True
        logger.info(f"{STOPPED_AT_ITERATION} {i}, the model is just a logistic regression with no tree.")

    # Stopping when the criterion doesn't vary anymore
    if LOW_IMPROVEMENT in self.early_stopping and i >= self.burn_in:
        last_ones = self.criterion_iter[-10: -1]
        variation = np.var(last_ones) ** 0.5
        if self.criterion == "gini" and variation < self.tol:
            stopping_criterion = True
            logger.info(f"{STOPPED_AT_ITERATION} {i}")
        if self.criterion != "gini" and variation < self.tol * abs(self.best_criterion):
            stopping_criterion = True
            logger.info(f"{STOPPED_AT_ITERATION} {i}")

    return stopping_criterion


def _init_models(self):
    models = {}

    for c_iter in range(self.class_num):
        # If penalty ='l1', solver=self.solver or self.solver (large datasets),
        # default ’sag’, C small leads to stronger regularization
        models[c_iter] = LogRegSegment(penalty='l2', solver=self.solver, C=1e-2, tol=1e-2,
                                       warm_start=True, data_treatment=self.data_treatment,
                                       discretization=self.discretization,
                                       column_names=self.column_names)

    return models


def _init_fit(self, X, y):
    models = _init_models(self)
    self.criterion_iter = []
    df = pd.DataFrame(deepcopy(X))
    if isinstance(X, pd.DataFrame):
        # Dictionary of the correspondence between the column names par_0... and the real names
        self.column_names = {}
        or_col_names = X.columns
        r = 0
        for column in df.columns:
            self.column_names[column] = or_col_names[r]
            r = r + 1
    else:
        df = df.add_prefix("par_")
        self.column_names = {}
    df["y"] = y

    treatment = {}
    if self.data_treatment:
        # Data without treatment (one hot on categorical variables), used for the tree
        # processing = Processing(target=self.target)
        # X_tree = processing.fit_transform(X=X.copy(), categorical=self.categorical)
        X_tree, enc_global = bin_data_cate_train(X.copy(), "y", categorical=self.categorical)
        treatment["global"] = enc_global
    else:
        X_tree = df.drop(['y'], axis=1)
    # if self.categorical:
    #     self.categorical = ["par_" + colname for colname in self.categorical]
    return df, X_tree, models, treatment


def fit(self, X, y, solver: str = "lbfgs", nb_init: int = 1, tree_depth: int = 10,
        min_impurity_decrease: float = 0.0, optimal_size: bool = True, tol: float = 0.005, categorical=None):
    """
    Fits the Lrtree object.

    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
    :param numpy.ndarray y:
        Boolean (0/1) labels of the observations. Must be of
        the same length as X
        (numpy "numeric" array).
    :param str solver:
        sklearn's solver for LogisticRegression (default 'lbfgs')
    :param int nb_init:
        Number of different random initializations
    :param int tree_depth:
        Maximum depth of the tree used
    :param float min_impurity_decrease:
        Parameter used to split (or not) the decision Tree
    :param bool optimal_size:
        Whether to use the tree parameters, or to take the optimal tree (used only with a validation set)
    :param float tol:
        Tolerance to observe an improvement and stop early
    :param list categorical:
        List of names of categorical features
    """
    self.tree_depth = tree_depth
    self.tol = tol
    self.min_impurity_decrease = min_impurity_decrease
    self.optimal_size = optimal_size
    self.solver = solver
    self.categorical = categorical

    if isinstance(X, pd.DataFrame):
        self.column_names = X.columns.to_list()
    elif self.data_treatment and not self.categorical:
        msg = "You did not provide categorical columns name, assuming only numerical columns."
        logger.info(msg)
    else:
        _check_args(X, y)
    if self.data_treatment and type(X) != pd.DataFrame:
        msg = "A numpy array cannot have mixed-type input, so using data_treatment is prohibited"
        logger.error(msg)
        raise ValueError(msg)

    self.n = len(y)

    _dataset_split(self)

    df, X_tree, models, treatment = _init_fit(self, X, y)

    for _ in tqdm(range(nb_init), desc="!= initialisations",
                  disable=os.environ.get("TQDM_DISABLE", "False").lower() in ('true', '1', 't')):
        if self.algo == 'sem':
            _fit_sem(self, df, X_tree, models, treatment)

        elif self.algo == 'em':
            _fit_em(self, df, models)


def _fit_func(fit_kwargs: dict, class_kwargs: dict = None):
    """
    Creates the lrtree model instance and fits it to the data

    :param str algo: either sem or em
    :param str criterion: either AIC (default), BIC or GINI
    :param float min_impurity_decrease: passed to DecisionTree
    :param bool validation: set aside a validation set?
    :param bool optimal_size: Whether to use the tree parameters, or to take the optimal tree
        (used only with a validation set)
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
        Number of initial discretization intervals for all variables.
    :param bool data_treatment:
        Whether to discretize / group levels inside the segments.
    :return: the Lrtree model instance
    :rtype: Lrtree
    """
    if class_kwargs is not None:
        model = lrtree.Lrtree(**class_kwargs)
    else:
        model = lrtree.Lrtree()
    model.fit(**fit_kwargs)
    return model


def _fit_parallelized(fit_kwargs: dict, class_kwargs: dict = None, nb_init: int = 2, nb_jobs: int = -1):
    """
    A fit function which creates the model instance and fits it,
    where the random initializations are parallelized

    :param str algo: either sem or em
    :param str criterion: either AIC (default), BIC or GINI
    :param float min_impurity_decrease: passed to DecisionTree
    :param bool validation: set aside a validation set?
    :param bool optimal_size: Whether to use the tree parameters, or to take the optimal tree
        (used only with a validation set)
    :param numpy.ndarray X:
        array_like of shape (n_samples, n_features)
        Vector to be scored, where `n_samples` is the number of samples and
        `n_features` is the number of features
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
        Number of initial discretization intervals for all variables.
    :param bool data_treatment:
        Whether to discretize / group levels inside the segments.
    :return: the Lrtree model instance
    :rtype: Lrtree
"""
    models = Parallel(n_jobs=nb_jobs)(
        delayed(_fit_func)(fit_kwargs=fit_kwargs, class_kwargs=class_kwargs) for _
        in range(nb_init))
    best_model = models[np.argmax([model.best_criterion for model in models])]
    return best_model
