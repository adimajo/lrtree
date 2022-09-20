import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from lrtree import Lrtree
from lrtree.discretization import Processing
from lrtree.fit import _fit_parallelized
from scripts.traitement_data import cacf_data
from kaggle.api import KaggleApi

api = KaggleApi()
api.authenticate()


BASE_DIR = r"N:\Projets02\GRO_STAGES\GRO_STG_2021_09 - Logistic Regression Trees\Segmentation_scores"


def get_adult_data():
    features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]
    categorical = ["Workclass", "Martial Status", "Occupation", "Relationship", "Race", "Sex", "Country"]

    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*',
                                 engine='python', na_values="?")
    original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*',
                                engine='python', na_values="?", skiprows=1)

    original_train["Target"] = original_train["Target"].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original_train["Target"] = original_train["Target"].astype(int)
    original_test["Target"] = original_test["Target"].replace('<=50K', '0').replace('>50K', '1').replace(
        '<=50K.', '0').replace('>50K.', '1')
    original_test["Target"] = original_test["Target"].astype(int)
    del original_train["Education"]
    del original_test["Education"]
    return original_train, original_test, original_train["Target"], original_test["Target"], features, categorical


def split_dataset(original: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train_rows = np.random.default_rng().choice(len(original), int(0.70 * len(original)), replace=False)
    test_rows = np.array(list(set(range(len(original))).difference(train_rows)))
    original_train = original.loc[train_rows, :]
    original_test = original.loc[test_rows, :]
    return original_train, original_test


def get_german_data():
    features = ["Status", "Duration", "History", "Purpose", "Amount", "Savings", "Employment", "Installment", "Personal", "Other",
                "Resident", "Property", "Age", "Plans", "Housing", "Number", "Job", "Maintenance", "Telephone",
                "Foreign", "Target"]
    categorical = ["Status", "History", "Purpose", "Savings", "Employment", "Installment", "Personal", "Other", "Property",
                   "Plans", "Housing", "Job", "Telephone", "Foreign"]
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    original = pd.read_csv(url, names=features, sep=r' ', engine='python', index_col=False)
    original["Target"] = original["Target"] - 1
    original_train, original_test = split_dataset(original)
    return original_train, original_test, original_train["Target"].reset_index(drop=True),\
        original_test["Target"].reset_index(drop=True), features, categorical


def get_fraud_data():
    features = [f"V{num}" for num in range(1, 29)] + ["Amount"]
    categorical = None
    api.dataset_download_cli("mlg-ulb/creditcardfraud", unzip=True)
    original = pd.read_csv("creditcard.csv", engine='python')
    del original['Time']
    original_train, original_test = split_dataset(original)
    return original_train, original_test, original_train["Target"].reset_index(drop=True),\
        original_test["Target"].reset_index(drop=True), features, categorical


def run_benchmark(dataset: str, target: str):
    logger.info(f"Getting {dataset} dataset and preprocessing.")
    processing = Processing(target=target)
    func_to_call = f"get_{dataset}_data"
    func_to_call = globals()[func_to_call]
    original_train, original_test, labels_train, labels_test, _, categorical = func_to_call()
    X_train = processing.fit_transform(X=original_train, categorical=categorical)
    X_test = processing.transform(original_test)
    model = Lrtree(criterion="gini", algo='SEM', class_num=6,
                   max_iter=200, validation=True, data_treatment=False,
                   leaves_as_segment=False, early_stopping="changed segments")
    logger.info("Fitting lrtree.")
    model.fit(X=X_train, y=labels_train, optimal_size=True, tree_depth=10, nb_init=1, tol=1e-9)
    logger.info(2 * roc_auc_score(labels_test, model.predict_proba(X_test)) - 1)
    logger.info("Fitting Logistic Regression.")
    modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto',
                                                    max_iter=100)
    modele_regLog.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, modele_regLog.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Decision Tree")
    model_tree = DecisionTreeClassifier(min_samples_leaf=500, random_state=0)
    model_tree.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_tree.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Gradient Boosting")
    model_boost = GradientBoostingClassifier(min_samples_leaf=100, random_state=0)
    model_boost.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_boost.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Random forest")
    model_forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=100, random_state=0)
    model_forest.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_forest.predict_proba(X_test)[:, 1]) - 1)


if __name__ == "__main__":
    run_benchmark("adult", "Target")
    logger.info("Getting adult dataset and preprocessing.")
    processing = Processing(target="Target")
    original_train, original_test, labels_train, labels_test, _, categorical = get_adult_data()
    X_train = processing.fit_transform(X=original_train, categorical=categorical)
    X_test = processing.transform(original_test)
    model = Lrtree(criterion="gini", algo='SEM', class_num=6,
                   max_iter=200, validation=True, data_treatment=False,
                   leaves_as_segment=False, early_stopping="changed segments")
    logger.info("Fitting lrtree.")
    model.fit(X=X_train, y=labels_train, optimal_size=True, tree_depth=10, nb_init=1, tol=1e-9)
    logger.info(2 * roc_auc_score(labels_test, model.predict_proba(X_test)) - 1)
    logger.info("Fitting Logistic Regression.")
    modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto',
                                                    max_iter=100)
    modele_regLog.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, modele_regLog.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Decision Tree")
    model_tree = DecisionTreeClassifier(min_samples_leaf=500, random_state=0)
    model_tree.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_tree.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Gradient Boosting")
    model_boost = GradientBoostingClassifier(min_samples_leaf=100, random_state=0)
    model_boost.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_boost.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Random forest")
    model_forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=100, random_state=0)
    model_forest.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_forest.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Getting german dataset and preprocessing.")
    processing = Processing(target="Target")
    original_train, original_test, labels_train, labels_test, _, categorical = get_german_data()
    X_train = processing.fit_transform(X=original_train, categorical=categorical)
    X_test = processing.transform(original_test)
    model = Lrtree(criterion="gini", algo='SEM', class_num=6,
                   max_iter=200, validation=True, data_treatment=False,
                   leaves_as_segment=False, early_stopping="changed segments")
    logger.info("Fitting lrtree.")
    model.fit(X=X_train, y=labels_train, optimal_size=True, tree_depth=10, nb_init=1, tol=1e-9)
    logger.info(2 * roc_auc_score(labels_test, model.predict_proba(X_test)) - 1)
    logger.info("Fitting Logistic Regression.")
    modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto',
                                                    max_iter=100)
    modele_regLog.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, modele_regLog.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Decision Tree")
    model_tree = DecisionTreeClassifier(min_samples_leaf=500, random_state=0)
    model_tree.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_tree.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Gradient Boosting")
    model_boost = GradientBoostingClassifier(min_samples_leaf=100, random_state=0)
    model_boost.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_boost.predict_proba(X_test)[:, 1]) - 1)

    logger.info("Fitting Random forest")
    model_forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=100, random_state=0)
    model_forest.fit(X_train, labels_train)
    logger.info(2 * roc_auc_score(labels_test, model_forest.predict_proba(X_test)[:, 1]) - 1)
