import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import tree

from lrtree import Lrtree
from lrtree.fit import _fit_parallelized

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['LOGURU_LEVEL'] = 'ERROR'


# Affichage de l'arbre obtenu, None, texte ou image
affichage = None
n_experiments = 5

hyperparameters_to_test = {
    "#samples": [100, 300, 500, 700, 1000, 2000, 3000, 5000, 8000, 10000],
    "#iterations": [40, 60, 80, 100, 120, 140, 160, 200],
    "#chains": range(1, 10)
}

results = {k: pd.DataFrame(
    data=np.array([[np.nan] * len(v)] * 7).T,
    columns=["BIC oracle"] + [critere + " moyen" for critere in ["BIC", "Arbre", "Forme"]] + [
        critere + " standard" for critere in ["BIC", "Arbre", "Forme"]],
    index=v) for k, v in hyperparameters_to_test.items()}


def one_experiment(X, y, n_init, n_iter):
    split1 = [[], [], []]
    split2 = [[], [], []]
    forme = 0
    arbre = 0
    theta_model = []

    model = _fit_parallelized(X, y, nb_init=n_init, tree_depth=2, class_num=4, max_iter=n_iter)
    if affichage == 'texte':
        text_representation = tree.export_text(model.best_link)
        logger.info(text_representation)
    elif affichage == 'image':
        tree.plot_tree(model.best_link)
        plt.show()
    treee = model.best_link.tree_
    feature = treee.feature
    threshold = treee.threshold

    split1[feature[0]].append(threshold[0])
    # 0, 1 et 4 sont les nodes de split
    split2[feature[1]].append(threshold[1])
    split2[feature[4]].append(threshold[4])

    if len(feature) == 7 and ((feature == [0, 1, -2, -2,
                                           1, -2, -2]).all() or (feature == [1, 0, -2, -2, 0, -2, -2]).all()):
        forme = 1
        if 0.1 > threshold[0] > -0.1 and 0.1 > threshold[1] > -0.1 and \
                0.1 > threshold[4] > -0.1:
            arbre = 1
            # log_reg = model.best_logreg
            # for i in range(len(log_reg)):
            #     theta_model.append([log_reg[i].intercept_,
            #                         log_reg[i].coef_[0][0], log_reg[i].coef_[0][1], log_reg[i].coef_[0][2]])

    return model.best_criterion, forme, arbre, split1, split2, theta_model


if __name__ == "__main__":
    for hyperparameter_to_test in hyperparameters_to_test.keys():
        logger.info(f'Sensitivity w.r.t. {hyperparameter_to_test}')

        for hyperparameter in hyperparameters_to_test[hyperparameter_to_test]:
            logger.info(f'{hyperparameter_to_test} {hyperparameter}')
            n_samples = hyperparameter if hyperparameter_to_test == "#samples" else 1000
            n_iter = hyperparameter if hyperparameter_to_test == "#iterations" else 100
            n_init = hyperparameter if hyperparameter_to_test == "#chains" else 5
            X, y, theta, BIC_oracle = Lrtree.generate_data(n_samples, 3, seed=1)

            criteria, formes, arbres, splits1, splits2, thetas = [], [], [], [], [], []

            for k in range(n_experiments):
                criterion, forme, arbre, split1, split2, theta_model = one_experiment(X, y, n_init, n_iter)
                criteria.append(criterion)
                formes.append(forme)
                arbres.append(arbre)
                splits1.append(split1)
                splits2.append(split2)

            results[hyperparameter_to_test].loc[hyperparameter] = [BIC_oracle,
                                                                   -np.mean(criteria), np.mean(formes), np.mean(arbres),
                                                                   np.std(criteria), np.std(formes), np.std(arbres)]

            # splits1.append(split1[0] + split1[1])
            # splits2.append(split2[0] + split2[1])
            # bon_critere1.append(1 - len(split1[2]) / n_experiments)
            # bon_critere2.append(1 - len(split2[2]) / (n_experiments * 2))

    for var in ["BIC", "Arbre", "Forme"]:
        for label, result_df in results.items():
            plt.figure()
            plt.plot(result_df.index, result_df[var + " moyen"])
            plt.fill_between(result_df.index, result_df[var + " moyen"] - result_df[var + " standard"],
                             result_df[var + " moyen"] + result_df[var + " standard"], alpha=0.5)
            if var == "BIC":
                plt.plot(result_df.index, result_df["BIC oracle"])
            plt.xlabel(label)
            plt.ylabel(var)
            plt.show()

    # TODO: clipping
    # np.maximum(np.array([0] * len(bonne_forme)), bonne_forme - std_forme),
    # np.minimum(np.array([1] * len(bonne_forme)), bonne_forme + std_forme), alpha=0.5)
