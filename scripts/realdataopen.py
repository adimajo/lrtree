import os
import zipfile
os.environ['LOGURU_LEVEL'] = 'INFO'
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

from lrtree.discretization import Processing
from lrtree.fit import _fit_parallelized, _fit_func

api = KaggleApi()
api.authenticate()


targets = {"adult": "Target", "german": "Target", "fraud": "Class"}


def split_dataset(original: pd.DataFrame, ratio: float = 0.7) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits dataset into training and testing sets

    :param pandas.DataFrame original: dataset to split
    :param float ratio: ratio of rows to keep in train dataset
    :return: training dataset, test dataset
    :rtype: (pandas.DataFrame, pandas.DataFrame)
    """
    train_rows = np.random.default_rng(123).choice(len(original), int(ratio * len(original)), replace=False)
    test_rows = np.array(list(set(range(len(original))).difference(train_rows)))
    original_train = original.loc[train_rows, :]
    original_test = original.loc[test_rows, :]
    return original_train, original_test


def get_adult_data(target: str):
    """
    Retrieve Adult dataset from UCI repository, transform Target to 0-1 and delete redundant information.

    :return: train and test sets (with labels), train and test labels, list of categorical features' names
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, list)
    """
    features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", target]
    categorical = ["Workclass", "Martial Status", "Occupation", "Relationship", "Race", "Sex", "Country"]

    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*',
                                 engine='python', na_values="?")
    original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*',
                                engine='python', na_values="?", skiprows=1)

    original_train[target] = original_train[target].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original_train[target] = original_train[target].astype(int)
    original_test[target] = original_test[target].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original_test[target] = original_test[target].astype(int)
    del original_train["Education"]
    del original_test["Education"]
    return original_train, original_test, original_train[target], original_test[target], categorical


def get_german_data(target: str):
    """
    Retrieve german dataset from UCI, transform Target to 0-1.

    :return: train and test sets (with labels), train and test labels, list of categorical features' names
    :rtype: (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, list)
    """
    features = ["Status", "Duration", "History", "Purpose", "Amount", "Savings", "Employment", "Installment",
                "Personal", "Other", "Resident", "Property", "Age", "Plans", "Housing", "Number", "Job", "Maintenance",
                "Telephone", "Foreign", target]
    categorical = ["Status", "History", "Purpose", "Savings", "Employment", "Installment", "Personal", "Other",
                   "Property", "Plans", "Housing", "Job", "Telephone", "Foreign"]
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    original = pd.read_csv(url, names=features, sep=r' ', engine='python', index_col=False)
    original[target] = original[target] - 1
    original_train, original_test = split_dataset(original)
    return original_train, original_test, original_train[target].reset_index(drop=True),\
        original_test[target].reset_index(drop=True), categorical


def get_fraud_data(target: str):
    """
    Retrieve german dataset from Kaggle, delete Time info.

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
    original_train, original_test = split_dataset(original)
    return original_train, original_test, original_train[target].reset_index(drop=True),\
        original_test[target].reset_index(drop=True), categorical


def run_benchmark(dataset: str) -> pd.DataFrame:
    """
    Run a benchmark experiment of Lrtree against other models on test set.

    :param str dataset: name of the dataset (one of 'adult', 'german', 'fraud')
    :param str target: name of the target feature
    :return: Performance of Lrtree, LogisticRegression, DecitionTree, XGboost and RandomForest on test set
    :rtype: pandas.DataFrame
    """
    logger.info(f"Getting {dataset} dataset and preprocessing.")
    processing = Processing(target=targets[dataset])
    func_to_call = f"get_{dataset}_data"
    func_to_call = globals()[func_to_call]
    original_train, original_test, labels_train, labels_test, categorical = func_to_call(target=targets[dataset])
    if dataset == "fraud":
        X_train = original_train.drop(targets[dataset], axis=1)
        X_test = original_test.drop(targets[dataset], axis=1)
    else:
        X_train = processing.fit_transform(X=original_train, categorical=categorical)
        X_test = processing.transform(original_test)
    model = _fit_func(
        class_kwargs={
            "criterion": "gini",
            "algo": 'SEM',
            "class_num": 4,
            "max_iter": 200,
            "validation": True,
            "data_treatment": True,
            "leaves_as_segment": True,
            "early_stopping": "changed segments"},
        fit_kwargs={
            "X": X_train,
            "y": labels_train,
            "optimal_size": False,
            "tree_depth": 2,
            "tol": 1e-9,
            "categorical": categorical})

    lrtree_test = 2 * roc_auc_score(labels_test, model.predict_proba(X_test)) - 1
    logger.info("Fitting Logistic Regression.")
    modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto',
                                                    max_iter=100)
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
    return lrtree_test, reglog_test, tree_test, boost_test, forest_test


if __name__ == "__main__":
    results = []
    for data in ["german", "adult", "fraud"]:
        lrtree_test, reglog_test, tree_test, boost_test, forest_test = run_benchmark(data)
        results.append([lrtree_test, reglog_test, tree_test, boost_test, forest_test])
