import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn import tree

from lrtree import Lrtree
from lrtree.fit import _fit_parallelized

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['LOGURU_LEVEL'] = 'ERROR'

bon_critere1 = []
bon_critere2 = []

split1 = []
split2 = []
BIC = []
bonne_forme = []
bon_arbre = []
theta_arbre = []

# Affichage de l'arbre obtenu, None, texte ou image
affichage = None

n_data = [100, 300, 500, 700, 1000, 2000, 3000, 5000, 8000, 10000]
n_iter = [80, 90, 100, 110, 120, 130, 140, 150]
n_para = range(5, 7)
n_experiments = 5

X, y, theta, BIC_oracle = Lrtree.generate_data(n_data[-1], 3, seed=1)

for n in n_para:
    logger.info(f"Nombre de chaînes parallèles : {n}")
    BIC_i = []
    forme_arbre = 0
    arbre = 0
    theta_i = []
    split1_i = [[], [], []]
    split2_i = [[], [], []]

    for k in range(n_experiments):
        X_train = deepcopy(X)
        y_train = deepcopy(y)
        model = _fit_parallelized(X_train, y_train,
                                  nb_init=n, tree_depth=2,
                                  class_num=4, max_iter=n_iter[0])
        BIC_i.append(model.best_criterion)
        if affichage == 'texte':
            text_representation = tree.export_text(model.best_link)
            logger.info(text_representation)
        elif affichage == 'image':
            tree.plot_tree(model.best_link)
            plt.show()
        treee = model.best_link.tree_
        feature = treee.feature
        threshold = treee.threshold

        split1_i[feature[0]].append(threshold[0])
        # 0, 1 et 4 sont les nodes de split
        split2_i[feature[1]].append(threshold[1])
        split2_i[feature[4]].append(threshold[4])

        if len(feature) == 7 and ((feature == [0, 1, -2, -2,
                                               1, -2, -2]).all() or (feature == [1, 0, -2, -2, 0, -2, -2]).all()):
            forme_arbre += 1
            if 0.1 > threshold[0] > -0.1 and 0.1 > threshold[1] > -0.1 and \
                    0.1 > threshold[4] > -0.1:
                arbre += 1
                theta_model = []
                log_reg = model.best_logreg
                for i in range(len(log_reg)):
                    theta_model.append([log_reg[i].intercept_,
                                        log_reg[i].coef_[0][0], log_reg[i].coef_[0][1], log_reg[i].coef_[0][2]])
                theta_i.append(theta_model)
    logger.info(f"BIC moyen pour {n} chaînes : {np.mean(BIC_i)}")
    BIC.append(BIC_i)
    theta_arbre.append(theta_i)
    bonne_forme.append(forme_arbre / n_experiments)
    bon_arbre.append(arbre / n_experiments)
    bon_critere1.append(1 - len(split1_i[2]) / n_experiments)
    bon_critere2.append(1 - len(split2_i[2]) / (n_experiments * 2))
    split1.append(split1_i[0] + split1_i[1])
    split2.append(split2_i[0] + split2_i[1])

logger.info(f"Bonne forme = {bonne_forme}")
logger.info(f"Bon arbre = {bon_arbre}")
logger.info(f"Theta = {theta_arbre}")
logger.info(f"BIC = {BIC}")
logger.info(f"Bon split racine = {bon_critere1}")
logger.info(f"Bon split noeud = {bon_critere2}")
logger.info(f"Split racine = {split1}")
logger.info(f"Split noeud = {split2}")

mean_bic = np.array([-np.mean(bic) for bic in BIC])
std_bic = np.array([np.std(bic) for bic in BIC])

std_forme = np.array([np.sqrt(forme * (1 - np.sqrt(forme)) / n_experiments) for forme in bonne_forme])
std_arbre = np.array([np.sqrt(arbre * (1 - np.sqrt(arbre)) / n_experiments) for arbre in bon_arbre])
std_split1 = np.array([np.sqrt(split_root * (1 - np.sqrt(split_root)) / n_experiments) for split_root in split1])
std_split2 = np.array([np.sqrt(split_leaf * (1 - np.sqrt(split_leaf)) / n_experiments) for split_leaf in split2])

plt.plot(n_para, mean_bic)
plt.fill_between(n_para, mean_bic - std_bic, mean_bic + std_bic, alpha=0.5)
plt.plot(n_para, [-BIC_oracle] * len(n_para))
plt.xlabel('Number of initialisations')
plt.ylabel('BIC')
plt.show()

plt.plot(n_para, bonne_forme)
plt.fill_between(n_para,
                 np.maximum(np.array([0] * len(bonne_forme)), bonne_forme - std_forme),
                 np.minimum(np.array([1] * len(bonne_forme)), bonne_forme + std_forme), alpha=0.5)
plt.xlabel('Number of initialisations')
plt.ylabel('Proportion of trees of the right form')
plt.show()

plt.plot(n_para, bon_arbre)
plt.fill_between(n_para,
                 np.maximum(np.array([0] * len(bon_arbre)), bon_arbre - std_arbre),
                 np.minimum(np.array([1] * len(bon_arbre)), bon_arbre + std_arbre), alpha=0.5)
plt.xlabel('Number of initialisations')
plt.ylabel('Proportion of correct trees')
plt.show()
