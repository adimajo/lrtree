"""
fit method for the Glmtree class
"""
import glmtree
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


class OneClassReg():
    def __init__(self, *args, **kwargs):
        self._single_class_label = None
        self.n_features_in_ = None

    def fit(self, X, y, sample_weight=None):
        self._single_class_label = y.iloc[0]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._single_class_label)

    def predict_proba(self, X):
        if self._single_class_label == 1:
            return np.full((X.shape[0], 2), [0, 1])
        else:
            return np.full((X.shape[0], 2), [1, 0])


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
    Splits the provided dataset into training, validation (to calculate the BIC, and choose the best model) and test sets
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


def calc_criterion(self, df, model_c_map):
    """
    Computes the criterion for this model.
        :param pandas.Dataframe df: The dataframe of the data used
        :param list model_c_map: The list of the models for each leaf.
        :returns: The criteria.
        :rtype: float
        """
    criterion = 0
    if self.criterion == "gini":
        # Computing the Area Under Curve of the ROC curve, which we maximise
        y_true = []
        y_proba = []
        k = 0
        for c_iter in range(df["c_map"].nunique()):
            model = model_c_map[k]
            idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
            if self.validation:
                y_validate = df[idx & df.index.isin(self.validate_rows)]["y"].tolist()
                X_validate = df[idx & df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
            else:
                y_validate = df[idx & df.index.isin(self.train_rows)]["y"].tolist()
                X_validate = df[idx & df.index.isin(self.train_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
            if X_validate.shape[0] > 0:
                pred = model.predict_proba(X_validate.to_numpy())
                y_pred = [pred[i][1] for i in range(len(pred))]
                y_true = [*y_true, *y_validate]
                y_proba = [*y_proba, *y_pred]
            k = k + 1
        criterion = roc_auc_score(y_true, y_proba)

    elif self.criterion == "aic":
        # Computing -AIC, which we maximise
        if self.validation:
            k = 0
            for c_iter in range(df["c_map"].nunique()):
                model = model_c_map[k]
                idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                y_validate = df[idx & df.index.isin(self.validate_rows)]["y"]
                X_validate = df[idx & df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
                if X_validate.shape[0] > 0:
                    y_pred = model.predict_proba(X_validate.to_numpy())
                    criterion = criterion - 2 * log_loss(y_validate, y_pred, normalize=False, labels=[0, 1]) \
                                - model.n_features_in_
                k = k + 1
        else:
            k = 0
            for c_iter in range(df["c_map"].nunique()):
                model = model_c_map[k]
                idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                y_true = df[idx & df.index.isin(self.train_rows)]["y"]
                X_pred = df[idx & df.index.isin(self.train_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
                y_pred = model.predict_proba(X_pred.to_numpy())
                criterion = criterion - 2 * log_loss(y_true, y_pred, normalize=False, labels=[0, 1]) \
                            - model.n_features_in_
                k = k + 1

    elif self.criterion == "bic":
        # Computing -BIC, which we maximise
        if self.validation:
            k = 0
            for c_iter in range(df["c_map"].nunique()):
                model = model_c_map[k]
                idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                y_validate = df[idx & df.index.isin(self.validate_rows)]["y"]
                X_validate = df[idx & df.index.isin(self.validate_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
                if X_validate.shape[0] > 0:
                    y_pred = model.predict_proba(X_validate.to_numpy())
                    criterion = criterion - 2 * log_loss(y_validate, y_pred, normalize=False, labels=[0, 1]) \
                                - np.log(len(X_validate)) * model.n_features_in_
                k = k + 1
        else:
            k = 0
            for c_iter in range(df["c_map"].nunique()):
                model = model_c_map[k]
                idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                y_true = df[idx & df.index.isin(self.train_rows)]["y"]
                X_pred = df[idx & df.index.isin(self.train_rows)][df.columns.difference(["y", "c_map", "c_hat"])]
                y_pred = model.predict_proba(X_pred.to_numpy())
                criterion = criterion - 2 * log_loss(y_true, y_pred,
                                                     normalize=False,
                                                     labels=[0, 1]) - \
                            np.log(len(X_pred)) * model.n_features_in_
                k = k + 1

    return criterion


def _vectorized_multinouilli(prob_matrix, items):
    """
    A vectorized version of multinouilli sampling.
    :param prob_matrix: A probability matrix of size n (number of training
        examples) * m[j] (the factor levels to sample from).
    :type prob_matrix: numpy.array
    :param list items: The factor levels to sample from.
    :returns: The drawn factor levels for each observation.
    :rtype: numpy.array
    """
    if len(items) != prob_matrix.shape[1]:
        msg = "The number of factors {} and the number of columns of prob_matrix {} must be of the same length".format(
            len(items), prob_matrix.shape)
        logger.error(msg)
        raise ValueError(msg)

    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0]).reshape((-1, 1))
    k = np.sum((s < r), axis=1)
    return items[k]


def fit(self, X, y, nb_init=1, tree_depth=10, min_impurity_decrease=0.0, OptimalSize=True):
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
        :param float min_impurity_decrease:
            Parameter used to split (or not) the decision Tree
        :param bool OptimalSize:
            Whether to use the tree parameters, or to take the optimal tree (used only with a validation set)
        """
    _check_args(X, y)

    self.n = len(y)

    _dataset_split(self)

    if self.algo == 'SEM':
        for k in range(nb_init):
            # Classification init
            Stop = False
            i = 0
            X_copy = np.copy(X)
            link = []
            self.criterion_iter = []
            df = pd.DataFrame(X_copy)
            df = df.add_prefix("par_")
            df["y"] = y
            df["c_map"] = np.random.randint(self.class_num, size=self.n)
            df["c_hat"] = df["c_map"]

            models = {}
            for c_iter in range(self.class_num):
                # If penalty ='l1', solver='liblinear' or 'saga' (large datasets), default ’lbfgs’, C small leads to stronger regularization
                models[c_iter] = LogisticRegression(penalty='l2', C=0.1, tol=1e-2, warm_start=True)

            # Start of main logic
            while i < self.max_iter and not Stop:
                print(i)
                logregs_c_hat = []
                logregs_c_map = []
                model_c_map = []
                coeff_adaptive_c_map = []
                predictions_log = np.zeros(shape=(self.n, df["c_hat"].nunique()))

                # Getting p(y | x, c_hat) and filling the probabilites
                for c_iter in range(df["c_hat"].nunique()):
                    idx = df["c_hat"] == np.sort(df["c_hat"].unique())[c_iter]
                    train_data = df[idx & df.index.isin(self.train_rows)]
                    y = train_data['y']
                    X = train_data.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                    if y.nunique() == 1:
                        # A model for when there is only one class (not a regression)
                        model = OneClassReg()
                        logreg = model.fit(X, y)
                    else:
                        model = models[c_iter]
                        logreg = model.fit(X, y)
                        models[c_iter] = model

                    logregs_c_hat.append(logreg)
                    to_predict = df.drop(['y', 'c_hat', 'c_map'], axis=1).to_numpy()
                    predictions_log[:, c_iter] = logreg.predict(to_predict)

                predictions_log[np.isnan(predictions_log)] = 0

                # Getting p(y | x, c_map)
                self.criterion_iter.append(0)

                for c_iter in range(df["c_map"].nunique()):
                    idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                    train_data = df[idx & df.index.isin(self.train_rows)]
                    y = train_data['y']
                    X = train_data.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()

                    if y.nunique() == 1:
                        model = OneClassReg()
                        logreg = model.fit(X, y)
                        # coeff = [1 for i in range(len(X[0]))]
                    else:
                        # model_estim = LogisticRegression(penalty='l2', solver='saga', C=0.1, tol=1e-2, warm_start=False, max_iter=50)
                        # model_estim.fit(X, y)
                        # # Estimation of the coefficients, to use as (inverse of) weight in adaptive lasso
                        # coeff = model_estim.coef_[0]
                        # for b in range(len(X[0])):
                        #     X[:, b] = X[:, b] / abs(coeff[b])
                        # If penalty ='l1', solver='liblinear' or 'saga' (large datasets), default ’lbfgs’, C small leads to stronger regularization
                        model = LogisticRegression(penalty='l1', solver='saga', C=0.01, tol=1e-2, warm_start=False)
                        logreg = model.fit(X, y)

                    logregs_c_map.append(logreg)
                    model_c_map.append(model)
                    # coeff_adaptive_c_map.append(coeff)

                # Gettting the total criterion, for this model (tree + reg) proposition
                self.criterion_iter[i] = calc_criterion(self, df, model_c_map)

                # Best model yet
                if self.criterion_iter[i] > self.best_criterion:
                    # Stopping when the criterion doesn't really get better anymore
                    if self.criterion == "gini" and i >= 30 and abs(
                            self.criterion_iter[i] - self.best_criterion) < 0.005:
                        Stop = True
                        print("Stopped at iteration", i)
                    if self.criterion != "gini" and i >= 30 and abs(
                            self.criterion_iter[i] - self.best_criterion) < 0.01 * abs(self.best_criterion):
                        Stop = True
                        print("Stopped at iteration", i)
                    self.best_logreg = logregs_c_map
                    # self.best_coeff = coeff_adaptive_c_map
                    self.best_link = link
                    self.best_criterion = self.criterion_iter[i]

                # Stopping when we reach a tree with only one leaf
                if i > 0 and link == []:
                    Stop = True
                    print("Stopped at iteration", i, "the model is juste a logistic regression with no tree.")

                # Stopping when the criterion doesn't vary anymore
                if i >= 20:
                    last_ones = self.criterion_iter[-10: -1]
                    variation = np.var(last_ones) ** 0.5
                    if self.criterion == "gini" and variation < 0.005:
                        Stop = True
                        print("Stopped at iteration", i)
                    if self.criterion != "gini" and variation < 0.01 * abs(self.best_criterion):
                        Stop = True
                        print("Stopped at iteration", i)

                # Getting p(c_hat | x)
                if df["c_hat"].nunique() > 1:
                    X = df.drop(['y', 'c_map', 'c_hat'], axis=1)
                    X = X[df.index.isin(self.train_rows)].to_numpy()
                    # Building the tree
                    clf = DecisionTreeClassifier(max_depth=tree_depth, min_impurity_decrease=min_impurity_decrease).fit(
                        X, df[df.index.isin(self.train_rows)]["c_hat"])
                    link = clf

                    if OptimalSize and self.validation:
                        X_validate = df.drop(['y', 'c_map', 'c_hat'], axis=1)
                        X_validate = X_validate[df.index.isin(self.validate_rows)]
                        path = clf.cost_complexity_pruning_path(X, df[df.index.isin(self.train_rows)]["c_hat"])
                        alphas = path.ccp_alphas

                        # Tree propositions, with more or less pruning
                        best_score = 0
                        a = 0
                        Improving = True
                        # Starts from the most complete tree, pruning while it improves the accuracy on the validation test
                        while Improving and a < len(alphas):
                            alpha = alphas[a]
                            tree = DecisionTreeClassifier(ccp_alpha=alpha)
                            tree.fit(X, df[df.index.isin(self.train_rows)]["c_hat"])
                            score = tree.score(X_validate.to_numpy(), df[df.index.isin(self.validate_rows)]["c_hat"])
                            # Choosing the tree with the best accuracy on the validation set
                            if score > best_score:
                                link = tree
                            # When pruning the tree starts to make us lose accuracy, we stop
                            else:
                                Improving = False
                            a = a + 1

                else:
                    logger.info("The tree has only its root! Premature end of algorithm.")
                    break

                # Choice of the new c_hat = random step
                X = df.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                y_ext = np.array([df["y"], ] * predictions_log.shape[1]).transpose()
                masked_predictions_log = np.ma.masked_array(predictions_log, mask=(1 - y_ext)).filled(0) \
                                         + np.ma.masked_array(1 - predictions_log, mask=y_ext).filled(0)
                matrix = np.multiply(link.predict_proba(X), masked_predictions_log)
                row_sums = matrix.sum(axis=1)
                p = matrix / row_sums[:, np.newaxis]
                df["c_hat"] = _vectorized_multinouilli(p, df["c_hat"].unique())

                # c_map calculation
                X = df.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                df["c_map"] = np.argmax(link.predict_proba(X), axis=1)

                i = i + 1


    elif self.algo == 'EM':
        for k in range(nb_init):
            # Random initialisation
            Stop = False
            i = 0
            X_copy = np.copy(X)
            self.criterion_iter = []
            df = pd.DataFrame(X_copy)
            df = df.add_prefix("par_")
            df["y"] = y
            df["c_map"] = np.zeros(self.n)
            df["c_hat"] = df["c_map"]  # Not used in this case
            random_init = np.random.random((len(df), self.class_num))
            row_sums = random_init.sum(axis=1)
            proportion = random_init / row_sums[:, np.newaxis]

            models = {}
            for c_iter in range(self.class_num):
                # If penalty ='l1', solver='liblinear' or 'saga', default=’lbfgs’
                models[c_iter] = LogisticRegression(penalty='l2', C=1, tol=1e-2, warm_start=True)

            # MCMC steps
            while i < self.max_iter and not Stop:
                logregs_c_hat = []
                logregs_c_map = []
                model_c_map = []
                predictions_log = np.zeros(shape=(self.n, self.class_num))

                # Getting p(y | x, c_hat) and filling the probabilities/proportions

                for c_iter in range(self.class_num):
                    weights = proportion[:, c_iter]
                    y = df['y']
                    X = df.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                    model = models[c_iter]
                    logreg = model.fit(X, y, weights)
                    models[c_iter] = model

                    logregs_c_hat = np.append(logregs_c_hat, logreg)
                    to_predict = df.drop(['y', 'c_hat', 'c_map'], axis=1).to_numpy()
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
                clf = DecisionTreeClassifier(max_depth=tree_depth, min_impurity_decrease=min_impurity_decrease)
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

                for c_iter in range(df["c_map"].nunique()):
                    idx = df["c_map"] == np.sort(df["c_map"].unique())[c_iter]
                    train_data = df[idx]
                    y = train_data['y']
                    X = train_data.drop(['y', 'c_map', 'c_hat'], axis=1).to_numpy()
                    if y.nunique() == 1:
                        model = OneClassReg()
                        logreg = model.fit(X, y)
                    else:
                        model = LogisticRegression(penalty='l2', C=1, tol=1e-2, warm_start=False)
                        logreg = model.fit(X, y)

                    logregs_c_map.append(logreg)
                    model_c_map.append(model)

                # Getting the total criterion
                self.criterion_iter[i] = calc_criterion(self, df, model_c_map)
                # Best results
                if self.criterion_iter[i] > self.best_criterion:
                    # Stopping when the criterion doesn't really get better anymore
                    if self.criterion == "gini" and i >= 30 and abs(
                            self.criterion_iter[i] - self.best_criterion) < 0.005:
                        Stop = True
                        print("Stopped at iteration", i)
                    if self.criterion != "gini" and i >= 30 and abs(
                            self.criterion_iter[i] - self.best_criterion) < 0.01 * abs(self.best_criterion):
                        Stop = True
                        print("Stopped at iteration", i)

                    self.best_logreg = logregs_c_map
                    self.best_link = link
                    self.best_criterion = self.criterion_iter[i]

                # Stopping when the criterion doesn't vary anymore
                if i >= 20:
                    last_ones = self.criterion_iter[-10: -1]
                    variation = np.var(last_ones) ** 0.5
                    if self.criterion == "gini" and variation < 0.005:
                        Stop = True
                        print("Stopped at iteration", i)
                    if self.criterion != "gini" and variation < 0.01 * abs(self.best_criterion):
                        Stop = True
                        print("Stopped at iteration", i)

                i = i + 1


def fit_func(X, y, algo='SEM', criterion="aic", max_iter=100, tree_depth=5, class_num=10, validation=False,
             min_impurity_decrease=0.0, OptimalSize=True):
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
    model = glmtree.Glmtree(algo=algo, test=False, validation=validation, criterion=criterion, ratios=(0.7,),
                            class_num=class_num, max_iter=max_iter)
    model.fit(X, y, tree_depth=tree_depth, min_impurity_decrease=min_impurity_decrease, OptimalSize=OptimalSize)
    return model


def fit_parralized(X, y, algo='SEM', criterion="aic", nb_init=5, nb_jobs=-1, max_iter=100, tree_depth=5, class_num=10,
                   min_impurity_decrease=0.0, OptimalSize=True,
                   validation=False):
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
        delayed(fit_func)(X, y, algo=algo, criterion=criterion, max_iter=max_iter, tree_depth=tree_depth,
                          class_num=class_num, validation=validation, min_impurity_decrease=min_impurity_decrease,
                          OptimalSize=OptimalSize) for k
        in range(nb_init))
    critere = -np.inf
    best_model = None
    for k in range(nb_init):
        model = models[k]
        criterion = model.best_criterion
        if criterion > critere:
            best_model = model
            critere = model.best_criterion
    return best_model
