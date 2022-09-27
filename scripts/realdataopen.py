import os
import zipfile
os.environ['LOGURU_LEVEL'] = 'ERROR'
os.environ['TQDM_DISABLE'] = '1'
import numpy as np
import pandas as pd
from kaggle.api import KaggleApi
from loguru import logger
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from lrtree.discretization import Processing
from lrtree.fit import _fit_parallelized

api = KaggleApi()
api.authenticate()


targets = {"adult": "Target", "german": "Target", "fraud": "Class"}


def get_from_uci(train: str, test: str = None, **kwargs) -> (pd.DataFrame, pd.DataFrame):
    filename_train = train.split('/')[-1]
    if not os.path.isfile(filename_train):
        original_train = pd.read_csv(train,  **kwargs)
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


def split_dataset(original: pd.DataFrame, seed: int, ratio: float = 0.7) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits dataset into training and testing sets

    :param pandas.DataFrame original: dataset to split
    :param int seed: random number generator seed
    :param float ratio: ratio of rows to keep in train dataset
    :return: training dataset, test dataset
    :rtype: (pandas.DataFrame, pandas.DataFrame)
    """
    train_rows = np.random.default_rng(seed).choice(len(original), int(ratio * len(original)), replace=False)
    test_rows = np.array(list(set(range(len(original))).difference(train_rows)))
    original_train = original.loc[train_rows, :]
    original_test = original.loc[test_rows, :]
    return original_train, original_test


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

    original_train[target] = original_train[target].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original_train[target] = original_train[target].astype(int)
    original_test[target] = original_test[target].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original_test[target] = original_test[target].astype(int)
    del original_train["Education"]
    del original_test["Education"]
    original_train, original_test = split_dataset(original, seed)
    return original_train, original_test, original_train[target], original_test[target], categorical


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
    original_train, original_test = split_dataset(original, seed)
    return original_train, original_test, original_train[target].reset_index(drop=True),\
        original_test[target].reset_index(drop=True), categorical


def get_fraud_data(target: str):
    """
    Retrieve german dataset from Kaggle, delete Time info.

    :param str target: target feature name
    :param int seed: random number generator seed
    :return: train and test sets (with labels), train and test labels, list of categorical features' names
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, list)
    """
    categorical = None
    api.dataset_download_cli("mlg-ulb/creditcardfraud", force=False)  # for some reason, unzip redownloads every time
    effective_path = api.get_default_download_dir(
        'datasets', "mlg-ulb", "creditcardfraud")
    outfile = os.path.join(effective_path, "creditcardfraud.zip")
    with zipfile.ZipFile(outfile) as z:
        z.extractall(effective_path)
    original = pd.read_csv("creditcard.csv", engine='python')
    del original['Time']
    original_train, original_test = split_dataset(original, seed)
    return original_train, original_test, original_train[target].reset_index(drop=True),\
        original_test[target].reset_index(drop=True), categorical


def get_data(dataset: str, seed: int = 0):
    logger.info(f"Getting {dataset} dataset and preprocessing.")
    processing = Processing(target=targets[dataset])
    func_to_call = f"get_{dataset}_data"
    func_to_call = globals()[func_to_call]
    original_train, original_test, labels_train, labels_test, categorical = func_to_call(target=targets[dataset],
                                                                                         seed=seed)
    if dataset == "fraud":
        X_train = original_train.drop(targets[dataset], axis=1)
        X_test = original_test.drop(targets[dataset], axis=1)
    else:
        X_train = processing.fit_transform(X=original_train, categorical=categorical)
        X_test = processing.transform(original_test)
    return X_train, X_test, labels_train, labels_test, categorical


def run_benchmark(X_train, X_test, labels_train: np.ndarray, labels_test: np.ndarray,
                  categorical: list, class_num: int = 5, data_treatment: bool = False,
                  leaves_as_segment: bool = False, optimal_size: bool = False,
                  tree_depth: int = 2) -> float:
    """
    Run a benchmark experiment of Lrtree against other models on test set.

    :param str dataset: name of the dataset (one of 'adult', 'german', 'fraud')
    :param int seed: random number generator seed
    :return: Performance of Lrtree, LogisticRegression, DecitionTree, XGboost and RandomForest on test set
    :rtype: pandas.DataFrame
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
            "X": original_train,
            # "X": X_train,
            "y": labels_train,
            "optimal_size": optimal_size,
            "tree_depth": tree_depth,
            "tol": 1e-9,
            "categorical": categorical})

    return 2 * roc_auc_score(labels_test, model.predict_proba(X_test)) - 1


def run_other_models(X_train, X_test, labels_train, labels_test):
    logger.info("Fitting Logistic Regression.")
    modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', max_iter=100)
    modele_regLog.fit(X_train, labels_train)
    reglog_test = 2 * roc_auc_score(labels_test, modele_regLog.predict_proba(X_test)[:, 1]) - 1

    logger.info("Fitting Decision Tree")
    model_tree = DecisionTreeClassifier(min_samples_leaf=500, random_state=0)
    model_tree.fit(X_train, labels_train)
    tree_test = 2 * roc_auc_score(labels_test, model_tree.predict_proba(X_test)[:, 1]) - 1

    logger.info("Fitting Gradient Boosting")
    model_boost = GradientBoostingClassifier(min_samples_leaf=100, random_state=0)
    model_boost.fit(X_train, labels_train)
    boost_test = 2 * roc_auc_score(labels_test, model_boost.predict_proba(X_test)[:, 1]) - 1

    logger.info("Fitting Random forest")
    model_forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=100, random_state=0)
    model_forest.fit(X_train, labels_train)
    forest_test = 2 * roc_auc_score(labels_test, model_forest.predict_proba(X_test)[:, 1]) - 1
    return reglog_test, tree_test, boost_test, forest_test


if __name__ == "__main__":
    results_df = []
    for data in ["german", "adult", "fraud"]:
        results_exp = []
        for seed in tqdm(range(0, 600, 30), desc="Seeds"):
            results_lrtree = []
            X_train, X_test, labels_train, labels_test, categorical = get_data(data, seed)
            reglog_test, tree_test, boost_test, forest_test = run_other_models(
                X_train, X_test, labels_train, labels_test)
            for class_num in range(2, 10):
                for data_treatment in [False]:
                    for leaves_as_segment in [True, False]:
                        for optimal_size in [True, False]:
                            for tree_depth in range(2, 10):
                                lrtree_test = run_benchmark(
                                    X_train, X_test, labels_train, labels_test, categorical,
                                    class_num, data_treatment, leaves_as_segment, optimal_size,
                                    tree_depth
                                )
                                results_lrtree.append(lrtree_test)
            lrtree_test = max(results_lrtree)
            results_exp.append([lrtree_test, reglog_test, tree_test, boost_test, forest_test])
        results_df.append(pd.DataFrame(results_exp, columns=["[MODEL]", "Logistic regression", "Decision tree",
                                                             "Boosting", "Random Forest"]).agg(["mean", "std"]))
