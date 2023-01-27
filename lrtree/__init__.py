"""This module is dedicated to logistic regression trees

.. autoclass:: Lrtree
   :members:

    NotFittedError
    _check_input_args
"""
__version__ = "1.0.2"

import numpy as np
from loguru import logger
import sklearn as sk

LOW_VARIATION = "low variation"
LOW_IMPROVEMENT = "low improvement"
CHANGED_SEGMENTS = "changed segments"


class NotFittedError(sk.exceptions.NotFittedError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from NotFittedError from sklearn which
    itself inherits from ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


def _check_input_args(algo: str, validation: bool, test: bool, ratios, criterion: str):
    """
    Checks input arguments :code:`algo`, :code:`validation`, :code:`test`, :code:`ratios` and :code:`criterion`

    :param str algo: either "sem" or "em"
    :param bool validation: whether to use validation set
    :param bool test: whether to use test set
    :param tuple ratios: proportion of validation / test samples
    :param str criterion: one of "gini", "bic", "aic"
    """
    # The algorithm should be one the ones in the list
    if type(algo) != str:
        msg = "algo must be a string"
        logger.error(msg)
        raise ValueError(msg)

    if algo.lower() not in ("sem", "em"):
        msg = "Algorithm " + algo + " is not supported"
        logger.error(msg)
        raise ValueError(msg)

    # Test is bool
    if type(test) is not bool:
        msg = "Test must be boolean"
        logger.error(msg)
        raise ValueError(msg)

    # Validation is bool
    if type(validation) is not bool:
        msg = "Validation must be boolean"
        logger.error(msg)
        raise ValueError(msg)

    # Ratios are not correctly defined
    if type(ratios) is not tuple:
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
    elif ratios not in [(0.7,), None]:
        msg = ("You provided dataset split ratios, but since test "
               "and validation are False, they will not be used")
        logger.warning(msg)

    # The criterion should be one of three from the list
    if criterion not in ("gini", "bic", "aic"):
        msg = "Criterion " + criterion + " is not supported"
        logger.error(msg)
        raise ValueError(msg)


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

    .. attribute:: best_reglog:

    :type: list

    The list of the best logistic regression on each segment (found with best_link).

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
                 data_treatment: bool = False,
                 discretization: bool = False,
                 leaves_as_segment: bool = False,
                 early_stopping=False,
                 burn_in: int = 30):
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
        :param tuple ratios:    The float ratio values for splitting of a dataset in test, validation.
                                Sum of values should be less than 1. Defaults to (0.7, 0.3)
        :param int class_num:   Number of initial segments. Defaults to 10.
        :param bool data_treatment: Whether or not we want the data to be discretized/merged categories in each
                                    leaf.
        :param bool leaves_as_segment: MAP or leaves-as-segment.
        :param early_stopping: bool (default: False or list of early stopping rules: can be one or several
            from "low improvement", "low variation", "changed segments").
        :param int burn_in: number of iterations to "burn".
        """
        _check_input_args(algo, validation, test, ratios, criterion)

        # Fit-specific
        self.tree_depth = None
        self.tol = None
        self.min_impurity_decrease = None
        self.optimal_size = None
        self.solver = None

        self.criterion = criterion.lower()
        self.algo = algo.lower()
        self.burn_in = burn_in
        msg = f"Unrecognized early stopping rule, must be in [{[LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS]}]."
        self.early_stopping = []
        if isinstance(early_stopping, bool):
            self.early_stopping = [LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS] if early_stopping else []
        elif isinstance(early_stopping, list):
            early_stopping = [string.lower() for string in early_stopping]
            for el in early_stopping:
                if el in [LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS]:
                    self.early_stopping.append(el)
            if not self.early_stopping:
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(early_stopping, str):
            if early_stopping.lower() in [LOW_IMPROVEMENT, LOW_VARIATION, CHANGED_SEGMENTS]:
                self.early_stopping = [early_stopping.lower()]
            else:
                logger.error(msg)
                raise ValueError(msg)

        if not validation and criterion == "gini":
            msg = "Using Gini index on training set might yield an overfitted model."
            logger.warning(msg)

        if validation and criterion in ("aic", "bic"):
            msg = "No need to penalize the log-likelihood when a validation set is used. Using log-likelihood " \
                  "instead of AIC/BIC."
            logger.warning(msg)

        self.test = test
        self.validation = validation
        self.max_iter = max_iter
        self.class_num = class_num
        self.ratios = ratios
        self.data_treatment = data_treatment
        self.column_names = None
        self.leaves_as_segment = leaves_as_segment
        self.discretization = discretization

        # Init data
        self.train_rows, self.validate_rows, self.test_rows = None, None, None
        self.n = 0

        # Results
        self.best_link = []
        self.best_logreg = None
        self.best_criterion = -np.inf
        self.criterion_iter = []
        self.best_treatment = None

    def _check_is_fitted(self):
        """
        Perform is_fitted validation for estimator.

        Checks if the estimator is fitted by verifying the presence of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError with the given message.
        This utility is meant to be used internally by estimators themselves,
        typically in their own predict / transform methods.
        """
        try:
            for logreg in self.best_logreg:  # pragma: no cover
                if isinstance(logreg, sk.linear_model.LogisticRegression):
                    sk.utils.validation.check_is_fitted(logreg)
            if isinstance(self.best_link, sk.tree.DecisionTreeClassifier):  # pragma: no cover
                sk.utils.validation.check_is_fitted(self.best_link)
        except (sk.exceptions.NotFittedError, TypeError) as e:
            raise NotFittedError(f"{str(e)}. If you did call fit, try increasing `iter`: it means it did not find a "
                                 f"better solution than the random initialization.")

    # Imported methods
    from .fit import fit
    from .generate_data import generate_data
    from .predict import predict
    from .predict import predict_proba
    from .predict import precision
