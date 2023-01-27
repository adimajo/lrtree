import os
import zipfile
# os.environ['LOGURU_LEVEL'] = 'ERROR'
# os.environ['TQDM_DISABLE'] = '1'
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from tqdm import tqdm  # noqa: E402
from typing import Tuple  # noqa: E402
from lrtree.discretization import Processing  # noqa: E402
from lrtree.fit import _fit_parallelized  # noqa: E402
from operator import itemgetter  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402


targets = {"adult": "Target", "german": "Target", "fraud": "Class"}


def main():
    results_df = []
    for data in ["german", "adult", "fraud"]:
        results_exp = []
        for seed in tqdm(range(0, 600, 30), desc="Seeds"):
            X_train, X_val, X_test, labels_train, labels_val, labels_test, categorical = get_data(data, seed)
            lrtree_test = run_lrtree(X_train, X_val, X_test, labels_train, labels_val, labels_test, categorical)
            reglog_test, tree_test, boost_test, forest_test = run_other_models(
                X_train, X_val, X_test, labels_train, labels_val, labels_test)
            results_exp.append([data, lrtree_test, reglog_test, tree_test, boost_test, forest_test])
        results_df.append(pd.DataFrame(results_exp, columns=[
            "Dataset", "[MODEL]", "Logistic regression", "Decision tree", "Boosting", "Random Forest"]).agg(
            ["mean", "std"]))
    # TODO: add changes from workstation


def get_from_uci(train: str, test: str = None, **kwargs) -> (pd.DataFrame, pd.DataFrame):
    filename_train = train.split('/')[-1]
    if not os.path.isfile(filename_train):
        original_train = pd.read_csv(train, **kwargs)
        original_train.to_pickle(filename_train)
    else:
        original_train = pd.read_pickle(filename_train)
    if test is not None:
        filename_test = test.split('/')[-1]
        if not os.path.isfile(filename_test):
            original_test = pd.read_csv(test, **kwargs, skiprows=1)
            original_test.to_pickle(filename_test)
        else:
            original_test = pd.read_pickle(filename_test)
    else:
        original_test = None
    return original_train, original_test


def split_dataset(original: pd.DataFrame, seed: int, ratio: Tuple[float, float] = (0.6, 0.2)) ->\
        (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits dataset into training and testing sets

    :param pandas.DataFrame original: dataset to split
    :param int seed: random number generator seed
    :param float ratio: ratio of rows to keep in train dataset
    :return: training dataset, test dataset
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
    """
    train_rows = np.random.default_rng(seed).choice(len(original), int(ratio[0] * len(original)), replace=False)
    # train_rows.sort()
    left_rows = np.array(list(set(range(len(original))).difference(train_rows)))
    # left_rows.sort()
    validation_rows = np.random.default_rng(seed).choice(len(left_rows), int(ratio[1] * len(original)), replace=False)
    # validation_rows.sort()
    validation_rows = left_rows[validation_rows]
    test_rows = np.array(list(set(left_rows).difference(validation_rows)))
    original_train = original.loc[train_rows, :]
    original_val = original.loc[validation_rows, :]
    original_test = original.loc[test_rows, :]
    return original_train, original_val, original_test


def get_adult_data(target: str, seed: int):
    """
    Retrieve Adult dataset from UCI repository, transform Target to 0-1 and delete redundant information.

    :param str target: target feature name
    :param int seed: random number generator seed
    :return: train and test sets (with labels), train and test labels, list of categorical features' names
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, list)
    """
    features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", target]
    categorical = ["Workclass", "Martial Status", "Occupation", "Relationship", "Race", "Sex", "Country"]

    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    original_train, original_test = get_from_uci(
        train=train_url, test=test_url, names=features, sep=r'\s*,\s*', engine='python', na_values="?")

    original = pd.concat([original_train.reset_index(drop=True),
                          original_test.reset_index(drop=True)], axis=0).reset_index(drop=True)

    original[target] = original[target].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original[target] = original[target].astype(int)
    del original["Education"]
    original_train, original_val, original_test = split_dataset(original, seed)
    return original_train.reset_index(drop=True), original_val.reset_index(drop=True),\
        original_test.reset_index(drop=True), original_train[target].reset_index(drop=True),\
        original_val[target].reset_index(drop=True), original_test[target].reset_index(drop=True), categorical


def get_german_data(target: str, seed: int):
    """
    Retrieve german dataset from UCI, transform Target to 0-1.

    :param str target: target feature name
    :param int seed: random number generator seed
    :return: train and test sets (with labels), train and test labels, list of categorical features' names
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, list)
    """
    features = ["Status", "Duration", "History", "Purpose", "Amount", "Savings", "Employment", "Installment",
                "Personal", "Other", "Resident", "Property", "Age", "Plans", "Housing", "Number", "Job", "Maintenance",
                "Telephone", "Foreign", target]
    categorical = ["Status", "History", "Purpose", "Savings", "Employment", "Installment", "Personal", "Other",
                   "Property", "Plans", "Housing", "Job", "Telephone", "Foreign"]
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    original, _ = get_from_uci(
        train=url, test=None, names=features, sep=r' ', engine='python', index_col=False)
    original[target] = original[target] - 1
    original_train, original_val, original_test = split_dataset(original, seed)
    return original_train.reset_index(drop=True), original_val.reset_index(drop=True),\
        original_test.reset_index(drop=True), original_train[target].reset_index(drop=True),\
        original_val[target].reset_index(drop=True), original_test[target].reset_index(drop=True), categorical


def get_fraud_data(target: str, seed: int):
    """
    Retrieve german dataset from Kaggle, delete Time info.

    :param str target: target feature name
    :param int seed: random number generator seed
    :return: train and test sets (with labels), train and test labels, list of categorical features' names
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, list)
    """
    from kaggle.api import KaggleApi  # noqa: E402
    api = KaggleApi()
    api.authenticate()
    categorical = None
    api.dataset_download_cli("mlg-ulb/creditcardfraud", force=False)  # for some reason, unzip redownloads every time
    effective_path = api.get_default_download_dir(
        'datasets', "mlg-ulb", "creditcardfraud")
    outfile = os.path.join(effective_path, "creditcardfraud.zip")
    with zipfile.ZipFile(outfile) as z:
        z.extractall(effective_path)
    original = pd.read_csv("creditcard.csv", engine='python')
    del original['Time']
    original_train, original_val, original_test = split_dataset(original, seed)
    return original_train.reset_index(drop=True), original_val.reset_index(drop=True),\
        original_test.reset_index(drop=True), original_train[target].reset_index(drop=True),\
        original_val[target].reset_index(drop=True), original_test[target].reset_index(drop=True), categorical


def get_data(dataset: str, seed: int = 0):
    logger.info(f"Getting {dataset} dataset and preprocessing.")
    processing = Processing(target=targets[dataset])
    func_to_call = f"get_{dataset}_data"
    func_to_call = globals()[func_to_call]
    original_train, original_val, original_test, labels_train, labels_val, labels_test, categorical = func_to_call(
        target=targets[dataset], seed=seed)
    if dataset == "fraud":
        X_train = original_train.drop(targets[dataset], axis=1)
        X_val = original_val.drop(targets[dataset], axis=1)
        X_test = original_test.drop(targets[dataset], axis=1)
    else:
        X_train = processing.fit_transform(X=original_train, categorical=categorical)
        X_val = processing.transform(original_val)
        X_test = processing.transform(original_test)
    return X_train, X_val, X_test, labels_train, labels_val, labels_test, categorical


def run_benchmark(X_train, X_val, X_test, labels_train: np.ndarray, labels_val: np.ndarray, labels_test: np.ndarray,
                  categorical: list, class_num: int = 5, data_treatment: bool = False,
                  leaves_as_segment: bool = False, optimal_size: bool = False,
                  tree_depth: int = 2) -> Tuple[float, float]:
    """
    Run a benchmark experiment of Lrtree against other models on test set.

    :return: Performance of Lrtree, LogisticRegression, DecitionTree, XGboost and RandomForest on test set
    :rtype: (float, float)
    """
    model = _fit_parallelized(
        nb_init=8,
        class_kwargs={
            "criterion": "aic",
            "algo": 'SEM',
            "class_num": class_num,
            "max_iter": 300,
            "validation": False,
            "data_treatment": data_treatment,
            "leaves_as_segment": leaves_as_segment,
            "early_stopping": "changed segments"},
        fit_kwargs={
            "X": X_train,
            "y": labels_train,
            "optimal_size": optimal_size,
            "tree_depth": tree_depth,
            "tol": 1e-9,
            "categorical": categorical})

    return roc_auc_score(labels_val, model.predict_proba(X_val)),\
        roc_auc_score(labels_test, model.predict_proba(X_test))


def run_lrtree(X_train, X_val, X_test, labels_train, labels_val, labels_test, categorical):
    results_lrtree = []
    for class_num in range(5, 6):
        for data_treatment in [False]:
            for leaves_as_segment in [True]:
                for optimal_size in [True]:
                    for tree_depth in range(3, 5):
                        lrtree_test = run_benchmark(
                            X_train, X_val, X_test, labels_train, labels_val, labels_test, categorical,
                            class_num, data_treatment, leaves_as_segment, optimal_size,
                            tree_depth)
                        results_lrtree.append(lrtree_test)
    return max(results_lrtree, key=itemgetter(0))[1]


def run_logreg(X_cv, labels_cv, X_test, labels_test):
    logger.info("Fitting Logistic Regression.")
    alphas = np.logspace(-2, 0, num=21)
    l1_ratios = np.linspace(0, 1, 11)
    pipeline = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=alphas, cv=10, penalty='elasticnet', scoring="roc_auc", solver="saga", n_jobs=-1, l1_ratios=l1_ratios))
    pipeline.fit(X_cv, labels_cv)
    return roc_auc_score(labels_test, pipeline.predict_proba(X_test)[:, 1])


def run_tree(X_train, X_val, X_test, labels_train, labels_val, labels_test):
    logger.info("Fitting Decision Tree")
    best_tree = None
    model_tree = DecisionTreeClassifier(random_state=0)
    path = model_tree.cost_complexity_pruning_path(X_train, labels_train)
    alphas = path.ccp_alphas

    # Tree propositions, with more or less pruning
    best_score = 0
    # Starts from the most complete tree, pruning while it improves the accuracy on the
    # validation test
    for a in range(len(alphas)):
        alpha = alphas[a]
        tree = DecisionTreeClassifier(ccp_alpha=alpha)
        tree.fit(X_train, labels_train)
        score = tree.score(X_val, labels_val)
        # Choosing the tree with the best accuracy on the validation set
        if score > best_score:
            best_score = score
            best_tree = tree

    return roc_auc_score(labels_test, best_tree.predict_proba(X_test)[:, 1])


def run_gradientboosting(X_cv, labels_cv, X_test, labels_test):
    logger.info("Fitting Gradient Boosting")

    param_grid = {
        'learning_rate': [0.01, 0.1, 0.5, 2.0],
        'n_estimators': [100, 300, 1000],
        'subsample': [0.5, 0.75, 1.0],
        'min_samples_split': [10, 30],
        'min_samples_leaf': [1, 5, 20],
        'max_depth': [2, 5, 10],
        'max_features': ['log2', 'sqrt', None]
    }

    model_boost = GradientBoostingClassifier()
    grid_search = GridSearchCV(estimator=model_boost,
                               param_grid=param_grid,
                               cv=10,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_cv, labels_cv)
    return roc_auc_score(labels_test, grid_search.predict_proba(X_test)[:, 1])


def run_randomforest(X_cv, labels_cv, X_test, labels_test):
    logger.info("Fitting Random forest")

    param_grid = {
        'n_estimators': [10, 100, 1000],
        'min_samples_split': [2, 10, 30],
        'min_samples_leaf': [5, 20],
        'max_depth': [None, 2, 5],
        'max_features': ['log2', 'sqrt', None],
        'ccp_alpha': [0.0, 0.1, 1.0]
    }

    model_boost = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=model_boost,
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_cv, labels_cv)
    return roc_auc_score(labels_test, grid_search.predict_proba(X_test)[:, 1])


def run_other_models(X_train, X_val, X_test, labels_train, labels_val, labels_test):
    # CV is train + val
    X_cv = pd.concat([X_train, X_val]).reset_index(drop=True)
    labels_cv = pd.concat([labels_train, labels_val]).reset_index(drop=True)

    # Reglog, decision tree, boosting, random forest
    reglog_test = run_logreg(X_cv, labels_cv, X_test, labels_test)
    tree_test = run_tree(X_train, X_val, X_test, labels_train, labels_val, labels_test)
    boost_test = run_gradientboosting(X_cv, labels_cv, X_test, labels_test)
    forest_test = run_randomforest(X_cv, labels_cv, X_test, labels_test)

    return reglog_test, tree_test, boost_test, forest_test


if __name__ == "__main__":
    main()
