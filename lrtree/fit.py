"""
fit module for the Lrtree class
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

import lrtree
from lrtree import LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS
from lrtree.logreg import LogRegSegment

STOPPED_AT_ITERATION = "Stopped at iteration"
# Floor of the likelihood of an observation in a segment, so that no observation is left without any segment
MIN_LIKELIHOOD = 1e-12


def _check_args(X, y) -> tuple:
    """
    Checks input arguments shape and types.

    :returns: the labels as a one-dimensional numpy array of 0 / 1, and the two original classes
    :raises: ValueError
    """
    y = np.asarray(y)
    if len(y) != X.shape[0]:
        msg = "X {} and y {} must be of the same length".format(X.shape, len(y))
        logger.error(msg)
        raise ValueError(msg)

    if isinstance(X, pd.DataFrame):
        numeric_x = all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    else:
        numeric_x = np.asarray(X).dtype.kind in "biuf"
    if not numeric_x:
        msg = "Unsupported data types. Columns of X must be int or float."
        logger.error(msg)
        raise ValueError(msg)

    if y.dtype.kind not in "biuf":
        msg = "Unsupported data types. Columns of y must be int or float."
        logger.error(msg)
        raise ValueError(msg)

    classes = np.unique(y)
    if y.ndim != 1 or len(classes) != 2:
        msg = "y must be composed of one column with the values of two classes (categorical or numeric)"
        logger.error(msg)
        raise ValueError(msg)
    # As in glmtree, the larger value is the positive class
    return (y == classes[1]).astype(int), classes


def _dataset_split(self):
    """
    Splits the provided dataset into training, validation (to calculate the BIC, and choose the best model) and
    test sets
    """
    shuffled = self._rng.permutation(self.n)
    fst_idx = int(self.ratios[0] * self.n)
    self.validate_rows, self.test_rows = None, None
    if self.validation and self.test:
        snd_idx = int((self.ratios[0] + self.ratios[1]) * self.n)
        self.train_rows, self.validate_rows, self.test_rows = np.split(shuffled, [fst_idx, snd_idx])
    elif self.validation:
        self.train_rows, self.validate_rows = np.split(shuffled, [fst_idx])
    elif self.test:
        self.train_rows, self.test_rows = np.split(shuffled, [fst_idx])
    else:
        self.train_rows = shuffled


def _new_model(self, **kwargs) -> LogRegSegment:
    """
    A fresh logistic regression for a segment
    """
    return LogRegSegment(discretization=self.data_treatment, **{**self.logreg_kwargs, **kwargs})


def _fit_segments(self, X, y, segments, predictions=None):
    """
    Fits one logistic regression per segment, on the training rows of the segment.

    :param numpy.ndarray X: the features (n, d)
    :param numpy.ndarray y: the 0 / 1 labels
    :param numpy.ndarray segments: the segment of each observation
    :param numpy.ndarray predictions: if provided, filled in place with P(y = 1 | x, segment) of all the
        observations, one column per segment label
    :returns: the models, keyed by segment label
    :rtype: dict
    """
    models = {}
    train = self.train_rows
    train_segments = segments[train]
    for label in np.unique(train_segments):
        rows = train[train_segments == label]
        model = _new_model(self)
        model.fit(X[rows], y[rows])
        models[label] = model
        if predictions is not None:
            predictions[:, label] = model.predict_proba(X)[:, 1]
    return models


def _n_parameters(model) -> int:
    """
    Number of (non zero) parameters of a segment regression, intercept included
    """
    if getattr(model, "_single_class_label", None) is not None:
        return 1
    return int(np.count_nonzero(model.coef_)) + 1


def _calc_criterion(self, X, y, c_map, models: dict, fallback) -> float:
    """
    Computes the criterion for this model, to be maximised.

    On the validation set (if any) it is the Gini index or the log-likelihood; on the training set it is the Gini
    index or the opposite of the AIC / BIC summed over the segments (the BIC penalty of a segment uses the
    logarithm of its size).

    :param numpy.ndarray X: The features
    :param numpy.ndarray y: The labels
    :param numpy.ndarray c_map: The segment of each observation
    :param dict models: The models of each segment, keyed by segment label.
    :param fallback: The model used for the observations of a segment without model.
    :returns: The criteria.
    :rtype: float
    """
    rows = self.validate_rows if self.validation else self.train_rows
    segments = c_map[rows]
    y_true = y[rows]
    y_proba = np.empty(len(rows))
    for label in np.unique(segments):
        in_segment = segments == label
        model = models.get(label, fallback)
        y_proba[in_segment] = model.predict_proba(X[rows[in_segment]])[:, 1]

    if self.criterion == "gini":
        if len(np.unique(y_true)) < 2:
            return -np.inf
        return 2 * roc_auc_score(y_true, y_proba) - 1

    log_likelihood = y_true * np.log(np.clip(y_proba, MIN_LIKELIHOOD, 1)) + (1 - y_true) * np.log(
        np.clip(1 - y_proba, MIN_LIKELIHOOD, 1))
    if self.validation:
        # No need to penalise the log-likelihood on data not used for fitting
        return float(log_likelihood.sum())

    criterion = 0.
    for label in np.unique(segments):
        in_segment = segments == label
        n_parameters = _n_parameters(models.get(label, fallback))
        penalty = 2 * n_parameters if self.criterion == "aic" else np.log(in_segment.sum()) * n_parameters
        criterion -= -2 * log_likelihood[in_segment].sum() + penalty
    return float(criterion)


def _vectorized_multinouilli(prob_matrix: np.ndarray, items: np.ndarray, rng=None) -> np.ndarray:
    """
    A vectorized version of multinouilli sampling.

    :param prob_matrix: A probability matrix of size n (number of training
        examples) * m[j] (the factor levels to sample from).
    :type prob_matrix: numpy.array
    :param numpy.ndarray items: The factor levels to sample from.
    :param numpy.random.Generator rng: the random number generator.
    :returns: The drawn factor levels for each observation.
    :rtype: numpy.ndarray
    """
    items = np.asarray(items)
    if len(items) != prob_matrix.shape[1]:  # pragma: no cover
        msg = "The number of factors {} and the number of columns of prob_matrix {} must be of the same length".format(
            len(items), prob_matrix.shape)
        logger.error(msg)
        raise ValueError(msg)

    rng = np.random.default_rng() if rng is None else rng
    s = prob_matrix.cumsum(axis=1)
    r = rng.random((prob_matrix.shape[0], 1))
    # Rounding errors on the cumulative sum could otherwise point past the last level
    k = np.minimum(np.sum(s < r, axis=1), len(items) - 1)
    return items[k]


def _segments(link: DecisionTreeClassifier, X: np.ndarray, leaves_as_segment: bool, tree_pred=None) -> np.ndarray:
    """
    The segment of each observation: either the most probable class of the tree (MAP) or its leaf.

    This is the single definition of c_map, used when fitting and when predicting.

    :param link: the segmentation tree
    :param numpy.ndarray X: the features
    :param bool leaves_as_segment: leaves of the tree are the segments, as opposed to its MAP prediction
    :param numpy.ndarray tree_pred: the probabilities of the tree on X, if already computed
    """
    if leaves_as_segment:
        return link.apply(X)
    if tree_pred is None:
        tree_pred = link.predict_proba(X)
    # The columns of the probabilities are positions in link.classes_, not segment labels
    return link.classes_[np.argmax(tree_pred, axis=1)]


def _draw_c_hat(link, X, y, predictions, rng):
    """
    Draws the new segments c_hat from p(c | x) * p(y | x, c)

    :param link: the segmentation tree
    :param numpy.ndarray X: the features
    :param numpy.ndarray y: the labels
    :param numpy.ndarray predictions: P(y = 1 | x, c) of each observation, one column per segment label
    :param numpy.random.Generator rng: the random number generator.
    :returns: the probabilities of the tree and the drawn segments
    """
    tree_pred = link.predict_proba(X)
    p = predictions[:, link.classes_]
    likelihood = np.maximum(np.where(y[:, np.newaxis] == 1, p, 1 - p), MIN_LIKELIHOOD)
    matrix = tree_pred * likelihood
    probabilities = matrix / matrix.sum(axis=1, keepdims=True)
    return tree_pred, _vectorized_multinouilli(probabilities, link.classes_, rng)


def _tree_params(self, min_samples_leaf: int = 1) -> dict:
    return dict(max_depth=self.tree_depth, min_impurity_decrease=self.min_impurity_decrease,
                min_samples_leaf=min_samples_leaf, random_state=int(self._rng.integers(2 ** 31 - 1)))


def _fit_tree(self, X, c_hat):
    """
    Fits the segmentation tree p(c_hat | x) on the training rows, then prunes it if requested (and if there is a
    validation set to choose the pruning on).
    """
    train = self.train_rows
    params = _tree_params(self, self.min_samples_leaf)
    link = DecisionTreeClassifier(**params).fit(X[train], c_hat[train])

    if self.optimal_size and self.validation:
        validate = self.validate_rows
        alphas = link.cost_complexity_pruning_path(X[train], c_hat[train]).ccp_alphas

        # Tree propositions, with more or less pruning: keep the one with the best accuracy on the validation set
        best_score = link.score(X[validate], c_hat[validate])
        for alpha in alphas:
            tree = DecisionTreeClassifier(ccp_alpha=alpha, **params).fit(X[train], c_hat[train])
            score = tree.score(X[validate], c_hat[validate])
            if score > best_score:
                link, best_score = tree, score
    return link


def _set_best(self, value, link, models):
    self.best_logreg = models
    self.best_link = link
    self.best_criterion = value


def _small(self, delta, reference) -> bool:
    """
    Whether delta is below tol: absolute for the Gini, relative to reference for the other criteria
    """
    if self.criterion == "gini":
        return abs(delta) < self.tol
    return abs(delta) < self.tol * abs(reference)


def _update_criterion(self, i, value, link, models) -> bool:
    """
    Records the criterion of iteration i, keeps the model if it is the best so far and applies the early stopping
    rules.

    :param int i: the iteration
    :param float value: the criterion of the model (the tree `link` and the regressions `models`)
    :returns: whether the chain should stop
    """
    self.criterion_iter.append(value)
    logger.debug(f"{self.criterion} at iteration {i} is {value:.3}.")
    stopping_criterion = False

    # A model needs a tree to be kept; the first iterations are burnt
    if link is not None:
        if self._fallback is None or value > self._fallback[0]:
            self._fallback = (value, link, models)
        if i >= self.burn_in and value > self.best_criterion:
            previous = self.best_criterion
            _set_best(self, value, link, models)
            # Stopping when the criterion doesn't really get better anymore
            improvement = value - previous
            if LOW_IMPROVEMENT in self.early_stopping and np.isfinite(previous) and _small(
                    self, improvement, previous):
                stopping_criterion = True
                logger.info(f"{STOPPED_AT_ITERATION} {i}, the criterion improved by less than 'tol'.")

    # Stopping when the criterion doesn't vary anymore
    if LOW_VARIATION in self.early_stopping and i >= self.burn_in and np.isfinite(self.best_criterion):
        variation = np.std(self.criterion_iter[-10:])
        if _small(self, variation, self.best_criterion):
            stopping_criterion = True
            logger.info(f"{STOPPED_AT_ITERATION} {i}, the criterion does not vary anymore.")

    return stopping_criterion


def _changed_segments_stop(self, i, new_cmap, c_map) -> bool:
    prop_changed_segments = 1 - np.mean(new_cmap == c_map)
    logger.info(f"Proportion of changed segments: {prop_changed_segments:.3}")
    logger.info(f"Number of distinct segments: {np.unique(new_cmap)}")
    if CHANGED_SEGMENTS in self.early_stopping and i >= self.burn_in and prop_changed_segments <= self.tol:
        # TODO: does not take into account relabelling
        logger.info(f"{STOPPED_AT_ITERATION} {i}, the proportion of changed segments is below 'tol'.")
        return True
    return False


def _fit_sem(self, X, y, fallback):
    """
    One chain of the stochastic EM algorithm: the segments are drawn at each iteration.
    """
    n = self.n
    c_map = self._rng.integers(self.class_num, size=n)
    c_hat = c_map.copy()
    link = None
    self.criterion_iter = []

    with tqdm(total=self.max_iter, leave=False, desc="Iterations") as pbar:
        for i in range(self.max_iter):
            logger.debug(f"Iteration {i}")
            # Getting p(y | x, c_hat) and filling the probabilities
            predictions = np.zeros(shape=(n, self.class_num))
            models_c_hat = _fit_segments(self, X, y, c_hat, predictions)

            # Getting p(y | x, c_map) (the same regressions at the first iteration)
            models_c_map = models_c_hat if np.array_equal(c_map, c_hat) else _fit_segments(self, X, y, c_map)

            # Getting the total criterion, for this model (tree + reg) proposition - Best model yet
            # The tree is the one of the previous iteration: it is the one that produced c_map
            value = _calc_criterion(self, X, y, c_map, models_c_map, fallback)
            if _update_criterion(self, i, value, link, models_c_map):
                break

            # Getting p(c_hat | x)
            if len(np.unique(c_hat[self.train_rows])) < 2:
                logger.info("The tree has only its root! Premature end of algorithm.")
                break
            link = _fit_tree(self, X, c_hat)

            # Choice of the new c_hat = random step
            tree_pred, c_hat = _draw_c_hat(link, X, y, predictions, self._rng)

            # c_map calculation
            new_cmap = _segments(link, X, self.leaves_as_segment, tree_pred)
            stop = _changed_segments_stop(self, i, new_cmap, c_map)
            c_map = new_cmap
            pbar.update(1)
            if stop:
                break


def _fit_em(self, X, y, fallback):
    """
    One chain of the EM algorithm: the segments are soft (a proportion of each observation in each segment).
    """
    n, K = self.n, self.class_num
    train = self.train_rows
    proportion = self._rng.random((n, K))
    proportion /= proportion.sum(axis=1, keepdims=True)
    c_map = None
    self.criterion_iter = []

    with tqdm(total=self.max_iter, leave=False, desc="Iterations") as pbar:
        for i in range(self.max_iter):
            # Getting p(y | x, c) and filling the probabilities/proportions
            predictions = np.zeros(shape=(n, K))
            for c_iter in range(K):
                model = _new_model(self)
                model.fit(X[train], y[train], sample_weight=proportion[train, c_iter])
                predictions[:, c_iter] = model.predict_proba(X)[:, 1]

            # Getting p(c | x): each observation is repeated once per segment, weighted by its proportion. A leaf
            # of m observations therefore holds m * K samples
            link = DecisionTreeClassifier(**_tree_params(self, self.min_samples_leaf * K)).fit(
                np.repeat(X[train], K, axis=0), np.tile(np.arange(K), len(train)),
                sample_weight=proportion[train].ravel())

            # New matrix of proportions p(c | x, y) = p(c | x) * p(y | x, c) / sum_c
            tree_pred = link.predict_proba(X)
            likelihood = np.maximum(np.where(y[:, np.newaxis] == 1, predictions, 1 - predictions), MIN_LIKELIHOOD)
            matrix = tree_pred * likelihood[:, link.classes_]
            proportion = np.zeros((n, K))
            proportion[:, link.classes_] = matrix / matrix.sum(axis=1, keepdims=True)

            # c_map calculation
            new_cmap = _segments(link, X, self.leaves_as_segment, tree_pred)
            stop = c_map is not None and _changed_segments_stop(self, i, new_cmap, c_map)
            c_map = new_cmap

            # Getting p(y | x, c_map) and the criterion: the tree is the one that produced c_map
            models_c_map = _fit_segments(self, X, y, c_map)
            value = _calc_criterion(self, X, y, c_map, models_c_map, fallback)
            stop = _update_criterion(self, i, value, link, models_c_map) or stop
            pbar.update(1)
            if stop:
                break


def _init_fit(self, X, y):
    """
    Prepares the data of a fit: features as a float array, labels as a 0 / 1 array and the global regression, used
    for the observations of a segment without regression.
    """
    self._rng = np.random.default_rng(self.random_state)
    self.criterion_iter = []
    self.best_link, self.best_logreg, self.best_criterion = None, None, -np.inf
    self._fallback = None
    self.column_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
    self.n = len(y)
    _dataset_split(self)
    X = np.asarray(X, dtype=float)
    global_model = _new_model(self)
    global_model.fit(X[self.train_rows], y[self.train_rows])
    return X, global_model


def fit(self, X, y, nb_init: int = 1, tree_depth: int = 10, min_impurity_decrease: float = 0.0,
        optimal_size: bool = True, tol: float = 0.005, min_samples_leaf: int = 1):
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
    :param int min_samples_leaf:
        Minimum number of (training) observations in a leaf of the tree
    """
    self.tree_depth = tree_depth
    self.tol = tol
    self.min_impurity_decrease = min_impurity_decrease
    self.optimal_size = optimal_size
    self.min_samples_leaf = min_samples_leaf

    if self.data_treatment and not isinstance(X, pd.DataFrame):
        msg = "A numpy array cannot have mixed-type input, so using data_treatment is prohibited"
        logger.error(msg)
        raise ValueError(msg)
    y, self.classes_ = _check_args(X, y)

    X, fallback = _init_fit(self, X, y)
    self.global_logreg = fallback

    # The warnings of the optimisers of the segments' regressions are not actionable: scope the filter to the chains
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for _ in tqdm(range(nb_init), desc="!= initialisations"):
            if self.algo == 'sem':
                _fit_sem(self, X, y, fallback)
            elif self.algo == 'em':
                _fit_em(self, X, y, fallback)

    # Chains that never got past the burn-in: keep the best model they went through
    if self.best_link is None and self._fallback is not None:
        _set_best(self, *self._fallback)


def _fit_func(class_kwargs: dict, fit_kwargs: dict):
    """
    Creates the lrtree model instance and fits it to the data

    :param dict class_kwargs: arguments of :class:`lrtree.Lrtree`
    :param dict fit_kwargs: arguments of :meth:`lrtree.Lrtree.fit`, including X and y
    :return: the Lrtree model instance
    :rtype: Lrtree
    """
    model = lrtree.Lrtree(**class_kwargs)
    model.fit(**fit_kwargs)
    return model


def _fit_parallelized(class_kwargs: dict, fit_kwargs: dict, nb_init: int, nb_jobs: int = -1):
    """
    A fit function which creates the model instance and fits it,
    where the random initializations are parallelized

    :param dict class_kwargs: arguments of :class:`lrtree.Lrtree`. If it contains a ``random_state``, each
        initialization gets its own seed derived from it.
    :param dict fit_kwargs: arguments of :meth:`lrtree.Lrtree.fit`, including X and y
    :param int nb_init:
        Number of different random initializations
    :param int nb_jobs:
        Number of jobs for the Parallelization
        Default : -1, all CPU are used
    :return: the Lrtree model instance with the best criterion
    :rtype: Lrtree
    """
    seeds = np.random.SeedSequence(class_kwargs.get("random_state")).generate_state(nb_init)
    models = Parallel(n_jobs=nb_jobs)(
        delayed(_fit_func)({**class_kwargs, "random_state": int(seed)}, fit_kwargs) for seed in seeds)
    return models[int(np.argmax([model.best_criterion for model in models]))]
