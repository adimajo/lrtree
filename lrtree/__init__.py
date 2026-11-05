"""This module is dedicated to logistic regression trees

.. autoclass:: Lrtree
   :members:

    NotFittedError
    _check_input_args
"""
__version__ = "1.1.0"

import numpy as np
from loguru import logger
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.utils.validation import check_is_fitted

LOW_VARIATION = "low variation"
LOW_IMPROVEMENT = "low improvement"
CHANGED_SEGMENTS = "changed segments"


class NotFittedError(SklearnNotFittedError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from NotFittedError from sklearn which
    itself inherits from ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


def _check_ratios(validation: bool, test: bool, ratios):
    """
    Checks the dataset split ratios against the sets that are required
    """
    if not isinstance(ratios, tuple):
        msg = "Ratios must be tuple"
        logger.error(msg)
        raise ValueError(msg)

    if any(i <= 0 for i in ratios):
        msg = "Dataset split ratios should be positive"
        logger.error(msg)
        raise ValueError(msg)

    if sum(ratios) >= 1:
        msg = "Dataset split ratios should be positive numbers with the sum less then 1"
        logger.error(msg)
        raise ValueError(msg)

    if validation and test:
        if len(ratios) != 2:
            msg = ("With validation and test, dataset split ratios should be 2 "
                   "positive numbers with the sum less then 1")
            logger.error(msg)
            raise ValueError(msg)
    elif (validation or test) and len(ratios) != 1:
        msg = ("With either validation or test, dataset split ratios should contain exactly 1 "
               "argument strictly between 0 and 1")
        logger.error(msg)
        raise ValueError(msg)
    elif ratios != (0.7,):
        msg = ("You provided dataset split ratios, but since test "
               "and validation are False, they will not be used")
        logger.warning(msg)


def _check_input_args(algo: str, validation: bool, test: bool, ratios, criterion: str):
    """
    Checks input arguments :code:`algo`, :code:`validation`, :code:`test`, :code:`ratios` and :code:`criterion`

    :param str algo: either "sem" or "em"
    :param bool validation: whether to use validation set
    :param bool test: whether to use test set
    :param tuple ratios: proportion of train / validation / test samples
    :param str criterion: one of "gini", "bic", "aic"
    """
    # The algorithm should be one the ones in the list
    if not isinstance(algo, str):
        msg = "algo must be a string"
        logger.error(msg)
        raise ValueError(msg)

    if algo.lower() not in ("sem", "em"):
        msg = "Algorithm " + algo + " is not supported"
        logger.error(msg)
        raise ValueError(msg)

    # Test is bool
    if not isinstance(test, bool):
        msg = "Test must be boolean"
        logger.error(msg)
        raise ValueError(msg)

    # Validation is bool
    if not isinstance(validation, bool):
        msg = "Validation must be boolean"
        logger.error(msg)
        raise ValueError(msg)

    _check_ratios(validation, test, ratios)

    # The criterion should be one of three from the list
    if not isinstance(criterion, str) or criterion.lower() not in ("gini", "bic", "aic"):
        msg = f"Criterion {criterion} is not supported"
        logger.error(msg)
        raise ValueError(msg)


def _set_early_stopping(early_stopping) -> list:
    """
    Parses the early stopping rules.

    :param early_stopping: bool, or one rule (str) or several (list of str) among "low improvement",
        "low variation" and "changed segments"
    :returns: the list of rules; unknown rules are ignored with a warning
    :rtype: list
    :raises: ValueError if rules are asked for but none of them is recognised
    """
    rules = (LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS)
    if isinstance(early_stopping, bool):
        return list(rules) if early_stopping else []
    if isinstance(early_stopping, str):
        early_stopping = [early_stopping]
    if isinstance(early_stopping, (list, tuple)):
        requested = [str(rule).lower() for rule in early_stopping]
        unknown = [rule for rule in requested if rule not in rules]
        if unknown:
            logger.warning(f"Unrecognized early stopping rule(s) {unknown}, ignored. Choose among {list(rules)}.")
        valid = [rule for rule in requested if rule in rules]
        if not valid:
            msg = f"Unrecognized early stopping rule, must be in {list(rules)}."
            logger.error(msg)
            raise ValueError(msg)
        return valid
    logger.warning(f"Unrecognized early stopping argument {early_stopping!r}, using default (False).")
    return []


class Lrtree:
    """
    The class implements a supervised method based in logistic trees. Its attributes:

    .. attribute:: test

    :type: bool

    Boolean (T/F) specifying if a test set is required.
    If True, the provided data is split to provide 20% of observations in a test set
    and the reported performance is the Gini index on test set.

    .. attribute:: validation

    :type: bool

    Boolean (T/F) specifying if a validation set is required.
    If True, the provided data is split to provide 20% of observations in a validation set
    and the reported performance is the Gini index on the validation set (if no test=False).
    The quality of the model at each step is evaluated using the Gini index on the
    validation set, so criterion must be set to "gini".

    .. attribute:: criterion

    :type: str

    The criterion to be used to assess the goodness-of-fit of the model:
    "bic" or "aic" if no validation set, else "gini".

    .. attribute:: max_iter

    :type: int

    Number of MCMC steps to perform. The more the better, but it may be more intelligent to use
    several MCMCs. Computation time can increase dramatically.

    .. attribute:: num_clas

    :type: int

    Number of initial segments.

    .. attribute:: criterion_iter

    :type: list

    The value of the criterion wished to be optimized over the iterations.

    .. attribute:: best_link

    :type: sklearn.tree.DecisionTreeClassifier

    The best decision tree.

    .. attribute:: best_logreg

    :type: dict

    The best logistic regression of each segment, keyed by segment label (found with best_link).

    .. attribute:: ratios

    :type: tuple

    The float ratio values for splitting of a dataset in test, validation.
    """
    def __init__(self,
                 algo: str = 'SEM',
                 test: bool = False,
                 validation: bool = False,
                 criterion: str = "bic",
                 ratios: tuple = (0.7,),
                 class_num: int = 10,
                 max_iter: int = 100,
                 discretize: bool = False,
                 group: bool = False,
                 leaves_as_segment: bool = False,
                 early_stopping=False,
                 burn_in: int = 30,
                 logreg_kwargs: dict = None,
                 random_state: int = None):
        """
        Initializes self by checking if its arguments are appropriately specified.

        :param str algo:        The algorithm to be used to fit the Lrtree: "SEM" for a stochastic approach or
                                "EM" for a non stochastic expectation/maximization algorithm.
        :param bool test:       Boolean specifying if a test set is required.
                                If True, the provided data is split to provide 20%
                                of observations in a test set and the reported
                                performance is the Gini index on test set.
        :param bool validation: Boolean (T/F) specifying if a validation set is
                                required. If True, the provided data is split to
                                provide 20% of observations in a validation set
                                and the reported performance is the Gini index on
                                the validation set (if no test=False). The quality
                                of the model at each step is evaluated
                                using the Gini index on the validation set, so
                                criterion must be set to "gini".
        :param str criterion:   The criterion to be used to assess the
                                goodness-of-fit of the model: "bic" or
                                "aic" if no validation set, else "gini".
        :param int max_iter:    Number of MCMC steps to perform. The more the
                                better, but it may be more intelligent to use
                                several MCMCs. Computation time can increase
                                dramatically. Defaults to 100.
        :param tuple ratios:    The float ratio values for splitting of a dataset: the first one is the share of the
                                training set, the second one (with both validation and test) is the share of the
                                validation set, the remainder goes to the test set.
                                Sum of values should be less than 1. Defaults to (0.7,)
        :param int class_num:   Number of initial segments. Defaults to 10.
        :param bool discretize: Whether we want the numerical data to be discretized in each leaf.
        :param bool group: Whether we want the categorical data to be merged in each leaf.
        :param bool leaves_as_segment: MAP or leaves-as-segment.
        :param early_stopping: bool (default: False or list of early stopping rules: can be one or several
            from "low improvement", "low variation", "changed segments").
        :param int burn_in: number of iterations to "burn": the best model is picked after them.
        :param dict logreg_kwargs: arguments of the logistic regression of each segment
            (:class:`sklearn.linear_model.LogisticRegression`); scikit-learn's defaults if None.
        :param int random_state: seed of the random number generator, for reproducible fits.
        """
        _check_input_args(algo, validation, test, ratios, criterion)

        # Fit-specific
        self.tree_depth = None
        self.tol = None
        self.min_impurity_decrease = None
        self.optimal_size = None
        self.min_samples_leaf = 1
        self.solver = None
        self.categorical = []

        self.criterion = criterion.lower()
        self.algo = algo.lower()
        self.burn_in = burn_in
        self.early_stopping = _set_early_stopping(early_stopping)

        if not validation and self.criterion == "gini":
            msg = "Using Gini index on training set might yield an overfitted model."
            logger.warning(msg)

        if validation and self.criterion in ("aic", "bic"):
            msg = "No need to penalize the log-likelihood when a validation set is used. Using log-likelihood " \
                  "instead of AIC/BIC."
            logger.warning(msg)

        self.test = test
        self.validation = validation
        self.max_iter = max_iter
        self.class_num = class_num
        self.ratios = ratios
        self.discretize = discretize
        self.group = group
        self.column_names = None
        self._tree_encoder = None
        self.leaves_as_segment = leaves_as_segment
        self.logreg_kwargs = {} if logreg_kwargs is None else dict(logreg_kwargs)
        self.random_state = random_state

        # Init data
        self.train_rows, self.validate_rows, self.test_rows = None, None, None
        self.n = 0

        # Results
        self.best_link = None
        self.best_logreg = None
        self.global_logreg = None
        self.classes_ = None
        self.best_criterion = -np.inf
        self.criterion_iter = []

    def _check_is_fitted(self):
        """
        Perform is_fitted validation for estimator.

        Checks that a tree and the regressions of its segments were found, and that they are fitted; otherwise
        raises a NotFittedError.
        """
        if self.best_link is None or not self.best_logreg:
            raise NotFittedError("This Lrtree instance is not fitted yet. If you did call fit, try increasing "
                                 "`max_iter`: it means it did not find a better solution than the random "
                                 "initialization.")
        try:
            check_is_fitted(self.best_link)
            for logreg in self.best_logreg.values():
                check_is_fitted(logreg)
        except (SklearnNotFittedError, TypeError) as e:
            raise NotFittedError(str(e))

    # Imported methods
    from .fit import fit
    from .generate_data import generate_data
    from .predict import predict
    from .predict import predict_proba
    from .predict import precision
