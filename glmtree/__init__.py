from loguru import logger
import sklearn as sk


class NotFittedError(sk.exceptions.NotFittedError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both NotFittedError from sklearn which
    itself inherits from ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class Glmtree:
    """
    The class implements a supervised method based in logistic trees
    """

    """
        This class implements a supervised multivariate discretization method,
        factor levels grouping and interaction discovery for logistic regression.
        .. attribute:: test
            Boolean (T/F) specifying if a test set is required.
            If True, the provided data is split to provide 20%
            of observations in a test set and the reported
            performance is the Gini index on test set.
           :type: bool
        .. attribute:: validation
            Boolean (T/F) specifying if a validation set is
            required. If True, the provided data is split to
            provide 20% of observations in a validation set
            and the reported performance is the Gini index on
            the validation set (if no test=False). The quality
            of the discretization at each step is evaluated
            using the Gini index on the validation set, so
            criterion must be set to "gini".
            :type: bool
        .. attribute:: criterion
            The criterion to be used to assess the
            goodness-of-fit of the discretization: "bic" or
            "aic" if no validation set, else "gini".
            :type: str
        .. attribute:: max_iter
            Number of MCMC steps to perform. The more the
            better, but it may be more intelligent to use
            several MCMCs. Computation time can increase
            dramatically.
            :type: int
        .. attribute:: num_class
            Number of initial discretization intervals for all
            variables. If :code:`num_class` is bigger than the number of
            factor levels for a given variable in
            X, num_class is set (for this variable
            only) to this variable's number of factor levels.
            :type: int
        .. attribute:: criterion_iter
            The value of the criterion wished to be optimized
            over the iterations.
            :type: list
        .. attribute:: best_link
            The best link function between the original
            features and their quantized counterparts that
            allows to quantize the data after learning.
            :type: list
        .. attribute:: best_reglog:
            The best logistic regression on quantized data found with best_link.
            :type: statsmodels.formula.api.glm
        .. attribute:: ratios
            The line rows corresponding to the splits.
            :type: tuple
        """

    def __init__(self, test: bool = False,
                 validation: bool = False,
                 criterion: str = "bic",
                 ratios: tuple = (0.7,),
                 class_num: int = 10,
                 max_iter: int = 100):
        """
                Initializes self by checking if its arguments are appropriately specified.
                    :param bool test:   Boolean specifying if a test set is required.
                                        If True, the provided data is split to provide 20%
                                        of observations in a test set and the reported
                                        performance is the Gini index on test set.
                :param bool validation: Boolean (T/F) specifying if a validation set is
                                        required. If True, the provided data is split to
                                        provide 20% of observations in a validation set
                                        and the reported performance is the Gini index on
                                        the validation set (if no test=False). The quality
                                        of the discretization at each step is evaluated
                                        using the Gini index on the validation set, so
                                        criterion must be set to "gini".
                :param str criterion:   The criterion to be used to assess the
                                        goodness-of-fit of the discretization: "bic" or
                                        "aic" if no validation set, else "gini".
                :param int max_iter:    Number of MCMC steps to perform. The more the
                                        better, but it may be more intelligent to use
                                        several MCMCs. Computation time can increase
                                        dramatically. Defaults to 100.
                :param tuple ratios:    The float ratio values for splitting of a dataset in test, validation.
                                        Sum of values should be less than 1. Defaults to (0.7, 0.3)
                :param int class_num:   Number of initial separation classes for all
                                        variables. If :code:`class_num` is bigger than the number of
                                        factor levels for a given variable in
                                        :code:`X`, :code:`class_num` is set (for this variable
                                        only) to this variable's number of factor levels. Defaults to 10.
                """
        # Test is bool
        if not type(test) is bool:
            raise ValueError("Test must be boolean")

        # Validation is bool
        if not type(validation) is bool:
            raise ValueError("Validation must be boolean")

        # Ratios are not correctly defined
        if not type(ratios) is tuple:
            raise ValueError("Ratios must be tuple")

        if any(i <= 0 for i in ratios):
            raise ValueError("Dataset split ratios should be positive")

        if sum(ratios) >= 1:
            raise ValueError("Dataset split ratios should be positive numbers with the sum less when 1")

        if validation and test:
            if len(ratios) != 2:
                raise ValueError("Dataset split ratios should be 2 positive numbers with the sum less when 1")
        elif validation or test:
            if len(ratios) != 1:
                raise ValueError("Dataset split ratios should contain 1 argument strictly between 0 and 1")
        else:
            message = "Dataset split ratios will not be used"
            logger.warning(message)

        # The criterion should be one of three from the list
        if criterion not in ("gini", "bic", "aic"):
            raise ValueError("Criterion ", criterion, " is not supported")

        if not validation and criterion == "gini":
            message = "Using Gini index on training set might yield an overfitted model."
            logger.warning(message)

        if validation and criterion in ("aic", "bic"):
            message = "No need to penalize the log-likelihood when a validation set is used. Using log-likelihood " \
                      "instead of AIC/BIC. "
            logger.warning(message)

        self.test = test
        self.validation = validation
        self.criterion = criterion
        self.max_iter = max_iter
        self.class_num = class_num
        self.ratios = ratios

        # Init data
        self.train_rows, self.validate_rows, self.test_rows = None, None, None
        self.n = 0

        # Results
        self.best_link, self.best_logreg = [], None
        self.criterion_iter, self.current_best = None, None

    def check_is_fitted(self):
        """Perform is_fitted validation for estimator.
        Checks if the estimator is fitted by verifying the presence of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError with the given message.
        This utility is meant to be used internally by estimators themselves,
        typically in their own predict / transform methods.
        """
        try:
            sk.utils.validation.check_is_fitted(self.best_logreg)
            for link in self.best_link:
                if isinstance(link, sk.linear_model.LogisticRegression):
                    sk.utils.validation.check_is_fitted(link)
        except sk.exceptions.NotFittedError as e:
            raise NotFittedError(str(e) + " If you did call fit, try increasing iter: "
                                          "it means it did not find a better solution than the random initialization.")

    # Imported methods
    from .fit import fit, _dataset_split
    from .utils import _generate_test_data
