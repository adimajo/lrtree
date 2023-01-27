import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from loguru import logger
from sklearn import tree
from tqdm import tqdm
from pathlib import Path

from lrtree import Lrtree
from lrtree.fit import _fit_parallelized

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logger.remove()
logger.add(tqdm.write)
affichage = None  # Affichage de l'arbre obtenu, None, texte ou image
test = os.environ.get("DEBUG", False)

if test:
    hyperparameters_to_test = {
        "#samples": [30, 120],
        "#iterations": [30, 200],
        "#chains": range(4, 6)
    }
    seeds = [4]
    n_experiments = 10
else:
    hyperparameters_to_test = {
        "#samples": [30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 230, 260, 290, 320, 350, 400, 450, 500,
                     550, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2300, 2600, 2900, 3200, 3500, 4000,
                     4500, 5000, 5500, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 30000],
        "#iterations": [10, 20, 30, 50, 75, 200],
        "#chains": range(1, 8)
    }
    seeds = [1324, 4567, 95]
    n_experiments = 200

results = {True: {k: pd.DataFrame(
    data=np.array([[np.nan] * len(v)] * 7).T,
    columns=["BIC oracle"] + [critere + " moyen" for critere in ["BIC",
                                                                 "Correct depth, features & splits (%)",
                                                                 "Correct depth & features (%)"]] + [
        critere + " standard" for critere in ["BIC",
                                              "Correct depth, features & splits (%)",
                                              "Correct depth & features (%)"]],
    index=v) for k, v in hyperparameters_to_test.items()}}

results[False] = deepcopy(results[True])


def one_experiment(X, y, n_init, n_iter, leaves_as_segment):
    split1 = [[], [], []]
    split2 = [[], [], []]
    forme = 0
    arbre = 0
    theta_model = []

    model = _fit_parallelized(class_kwargs={'class_num': 4, 'max_iter': n_iter,
                                            'leaves_as_segment': leaves_as_segment},
                              fit_kwargs={'X': X, 'y': y, 'tree_depth': 2},
                              nb_init=n_init,
                              nb_jobs=-1)
    if affichage == 'texte':
        text_representation = tree.export_text(model.best_link)
        logger.info(text_representation)
    elif affichage == 'image':
        tree.plot_tree(model.best_link)
        plt.show()
    if model.best_link:
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


def main():
    for seed in tqdm(seeds, desc="Seeds"):
        logger.info(f'Seed {seed}')
        for leaves_as_segment in tqdm([True, False], leave=False, desc="Types of segments"):
            logger.info("Leaves as segments" if leaves_as_segment else "MAP segments")
            for hyperparameter_to_test in tqdm(hyperparameters_to_test.keys(), leave=False, desc="Hyperparameters"):
                logger.info(f'Sensitivity w.r.t. {hyperparameter_to_test}')

                for hyperparameter in tqdm(hyperparameters_to_test[hyperparameter_to_test], leave=False,
                                           desc="Current hyperparameter values"):
                    logger.info(f'{hyperparameter_to_test} = {hyperparameter}')
                    n_samples = hyperparameter if hyperparameter_to_test == "#samples" else 6000
                    n_iter = hyperparameter if hyperparameter_to_test == "#iterations" else 100
                    n_init = hyperparameter if hyperparameter_to_test == "#chains" else 5
                    X, y, theta, BIC_oracle = Lrtree.generate_data(n_samples, 3, seed=seed)

                    # criteria, formes, arbres, splits1, splits2, thetas = [], [], [], [], [], []
                    criteria, formes, arbres, splits1, splits2 = [], [], [], [], []

                    for k in tqdm(range(n_experiments), leave=False, desc="Experiments"):
                        criterion, forme, arbre, split1, split2, theta_model = one_experiment(X, y, n_init, n_iter,
                                                                                              leaves_as_segment)
                        criteria.append(criterion)
                        formes.append(forme)
                        arbres.append(arbre)
                        splits1.append(split1)
                        splits2.append(split2)

                    results[leaves_as_segment][hyperparameter_to_test].loc[hyperparameter] = [
                        BIC_oracle,
                        -np.mean(criteria), np.mean(formes), np.mean(arbres),
                        np.std(criteria), np.std(formes), np.std(arbres)]

                    # splits1.append(split1[0] + split1[1])
                    # splits2.append(split2[0] + split2[1])
                    # bon_critere1.append(1 - len(split1[2]) / n_experiments)
                    # bon_critere2.append(1 - len(split2[2]) / (n_experiments * 2))

        for var in ["BIC", "Correct depth, features & splits (%)", "Correct depth & features (%)"]:
            for label, _ in results[True].items():
                plt.figure()
                for leaves_as_segment in [True, False]:
                    plt.plot(results[leaves_as_segment][label].index, results[leaves_as_segment][label][var + " moyen"],
                             label="Leaves as segments" if leaves_as_segment else "MAP segments")
                    low = results[leaves_as_segment][label][var + " moyen"] - results[
                        leaves_as_segment][label][var + " standard"]
                    high = results[leaves_as_segment][label][var + " moyen"] + results[
                        leaves_as_segment][label][var + " standard"]
                    if var != "BIC":
                        low = np.maximum(np.array([0] * len(low)), low)
                        high = np.minimum(np.array([1] * len(high)), high)
                    plt.fill_between(results[leaves_as_segment][label].index, low, high, alpha=0.5)
                if var == "BIC":
                    plt.plot(results[leaves_as_segment][label].index, results[leaves_as_segment][label]["BIC oracle"],
                             label="Oracle")
                else:
                    plt.plot(results[leaves_as_segment][label].index, np.array([1] * len(high)),
                             label="Oracle")

                plt.xlabel(label)
                plt.ylabel(var)
                plt.legend(loc='lower right')
                Path(os.path.join(BASE_DIR, "pictures/")).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(BASE_DIR,
                                         f"pictures/{var}_{label}_{seed}.png"))
                Path(os.path.join(BASE_DIR, "tikz/")).mkdir(parents=True, exist_ok=True)
                tikzplotlib.save(os.path.join(BASE_DIR,
                                              f"tikz/{var}_{label}_{seed}.tex"))


if __name__ == "__main__":
    main()
