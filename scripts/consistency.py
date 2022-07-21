import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from tqdm import tqdm
from loguru import logger
from sklearn import tree

from lrtree import Lrtree
from lrtree.fit import _fit_parallelized

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
logger.remove()
logger.add(tqdm.write)
os.environ['LOGURU_LEVEL'] = 'ERROR'

# Affichage de l'arbre obtenu, None, texte ou image
affichage = None
n_experiments = 100
hyperparameters_to_test = {
    "#samples": [30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 230, 260, 290, 320, 350, 400, 450, 500, 550,
                 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2300, 2600, 2900, 3200, 3500, 4000, 4500, 5000,
                 5500, 6000],
    "#iterations": [10, 30, 50, 75, 100],
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

        for hyperparameter in tqdm(hyperparameters_to_test[hyperparameter_to_test]):
            logger.info(f'{hyperparameter_to_test} = {hyperparameter}')
            n_samples = hyperparameter if hyperparameter_to_test == "#samples" else 2000
            n_iter = hyperparameter if hyperparameter_to_test == "#iterations" else 80
            n_init = hyperparameter if hyperparameter_to_test == "#chains" else 5
            X, y, theta, BIC_oracle = Lrtree.generate_data(n_samples, 3, seed=1)

            criteria, formes, arbres, splits1, splits2, thetas = [], [], [], [], [], []

            for k in tqdm(range(n_experiments), leave=False):
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
            low = result_df[var + " moyen"] - result_df[var + " standard"]
            high = result_df[var + " moyen"] + result_df[var + " standard"]
            if var != "BIC":
                low = np.maximum(np.array([0] * len(low)), low)
                high = np.minimum(np.array([1] * len(high)), high)
            plt.fill_between(result_df.index, low, high, alpha=0.5)
            if var == "BIC":
                plt.plot(result_df.index, result_df["BIC oracle"])
            plt.xlabel(label)
            plt.ylabel(var)
            plt.savefig(os.path.join(BASE_DIR, f"pictures/{var}_{label}.png"))
            tikzplotlib.save(os.path.join(BASE_DIR, f"tikz/{var}_{label}.tex"))
