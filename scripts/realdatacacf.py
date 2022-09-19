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

BASE_DIR = r"N:\Projets02\GRO_STAGES\GRO_STG_2021_09 - Logistic Regression Trees\Segmentation_scores"

data_crca = pd.read_sas(os.path.join(BASE_DIR, "base_crca.sas7bdat"))
data_lcl = pd.read_sas(os.path.join(BASE_DIR, "base_lcl.sas7bdat"))
data_edm = pd.read_sas(os.path.join(BASE_DIR, "base_edm.sas7bdat"))
data_cc = pd.read_sas(os.path.join(BASE_DIR, "base_cc.sas7bdat"))
data_gd = pd.read_sas(os.path.join(BASE_DIR, "base_gd_vf.sas7bdat"))
data_auto = pd.read_sas(os.path.join(BASE_DIR, "base_auto_vf.sas7bdat"))
data_instit = pd.read_sas(os.path.join(BASE_DIR, "base_inst_vf.sas7bdat"))


def get_cacf_data():
    # score2 entre 168 et 391 pour crca ?
    # segments = grscor2 = [b'1A', b'1C', b'1D', b'1J', b'1M', b'1P', b'1U', b'1Y', b'2U', b'2L', b'3L']
    common = ['DOFFR', 'DNAISS', 'DNACJ', 'DEMBA', 'AMEMBC', 'DCLEM', 'HABIT', 'SITFAM', 'CSP', 'CSPCJ',
              'TOP_COEMP', 'CPCL', 'PROD', 'SPROD', 'CPROVS', 'MREVNU', 'MREVCJ', 'MREVAU', 'MRCJAU', 'MCDE', 'CREDAC',
              'APPORT', 'ENDEXT', 'NBENF', 'MLOYER', 'MT_LOYER', 'MT_CHRG', 'MT_PENS_DU', 'ECJCOE', 'RFMD', 'AMCIRC',
              'NATB', 'CVFISC', 'cible', 'grscor2', 'score2']
    used = ['DNAISS', 'DNACJ', 'DEMBA', 'AMEMBC', 'DCLEM', 'HABIT', 'SITFAM', 'CSP', 'CSPCJ', 'TOP_COEMP', 'CPCL',
            'PROD', 'SPROD', 'CPROVS', 'MREVNU', 'MREVCJ', 'MREVAU', 'MRCJAU', 'MCDE', 'CREDAC', 'APPORT', 'ENDEXT',
            'NBENF', 'MLOYER', 'MT_LOYER', 'MT_CHRG', 'MT_PENS_DU', 'ECJCOE', 'RFMD', 'AMCIRC', 'NATB', 'CVFISC']
    cate = ['HABIT', 'SITFAM', 'CSP', 'CSPCJ', 'TOP_COEMP', 'PROD', 'SPROD', 'CPROVS', 'NBENF', 'ECJCOE', 'NATB',
            'CVFISC', 'CPCL']
    data = pd.concat([data_crca[common], data_lcl[common], data_edm[common], data_cc[common], data_gd[common],
                      data_auto[common], data_instit[common]], ignore_index=True)
    data['CPCL'] = data['CPCL'].str.slice(0, 2)
    data = cacf_data(data)
    train_rows = np.random.choice(len(data), 100000, replace=False)
    data_train = data[data.index.isin(train_rows)]
    data_val = data.drop(train_rows)
    y_train = data_train["cible"].astype(np.int32).to_numpy()
    y_val = data_val["cible"].astype(np.int32).to_numpy()
    return data, data_train, data_val, y_train, y_val, common, used, cate


if __name__ == "__main__":
    data, data_train, data_val, y_train, y_val, common, used, cate = get_cacf_data()
    processing = Processing(target="cible")
    X_train = processing.fit_transform(X=data_train[used + ["cible"]], categorical=cate)
    X_test = processing.transform(data_val)

    model = Lrtree(criterion="gini", algo='SEM', class_num=8,
                   max_iter=200, validation=True, data_treatment=False)
    model.fit(X=X_train, y=y_train, optimal_size=True, tree_depth=3)

    # model = _fit_parallelized(X_train, y_train, criterion="gini", algo='SEM', nb_init=8, tree_depth=3, class_num=8,
    #                           max_iter=200, validation=True, data_treatment=False, optimal_size=True)

    tree.plot_tree(model.best_link, feature_names=processing.labels)
    plt.show()
    plt.close()
    text_representation = tree.export_text(model.best_link)
    logger.info(text_representation)
    logger.info(f"Nb segments : {len(model.best_logreg)}")

    logger.info("Régression logistique")
    modele_regLog = linear_model.LogisticRegression(random_state=0, solver='liblinear', multi_class='auto',
                                                    max_iter=100)
    modele_regLog.fit(X_train, y_train)

    logger.info("Arbre de décision")
    model_tree = DecisionTreeClassifier(min_samples_leaf=500, random_state=0)
    model_tree.fit(X_train, y_train)

    logger.info("Gradient Boosting")
    model_boost = GradientBoostingClassifier(min_samples_leaf=100, random_state=0)
    model_boost.fit(X_train, y_train)

    logger.info("Random forest")
    model_forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=100, random_state=0)
    model_forest.fit(X_train, y_train)

    y_total_train = []
    y_total_proba = []
    for seg in np.unique(data["grscor2"]):
        logger.info(seg)
        sub_data = data[data["grscor2"] == seg]
        X_test = traitement_val(sub_data[used], enc, scaler, merged_cat, discret_cat)
        y_test = sub_data['cible'].to_numpy()
        y_proba = model.predict_proba(X_test)
        logger.info(f"SEM: {roc_auc_score(y_test, y_proba)}")
        y_total_train = [*y_total_train, *y_test]
        y_total_proba = [*y_total_proba, *y_proba]
    logger.info(f"SEM total: {roc_auc_score(y_total_train, y_total_proba)}")

    y_total_train = []
    y_total_proba = []
    for base in [data_gd, data_cc, data_edm, data_instit, data_auto, data_crca, data_lcl]:
        sub_base = cacf_data(base[common])
        X_test = traitement_val(sub_base[used], enc, scaler, merged_cat, discret_cat)
        y_test = sub_base['cible'].to_numpy()
        y_proba = model.predict_proba(X_test)
        logger.info(f"SEM: {roc_auc_score(y_test, y_proba)}")
        y_total_train = [*y_total_train, *y_test]
        y_total_proba = [*y_total_proba, *y_proba]
    logger.info(f"SEM total: {roc_auc_score(y_total_train, y_total_proba)}")
