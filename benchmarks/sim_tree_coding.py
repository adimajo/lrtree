"""
Why does lrtree trail glmtree on the 'tree' simulation? The true segments are pairs of levels of the categorical
``x1``. glmtree's ctree splits on subsets of levels; lrtree needs numeric features, so ``x1`` is dummy coded and a
CART tree needs several splits to isolate a pair of levels. This compares dummy coding with a numeric coding of ``x1``
(the levels are ordered like the clusters).

Usage: python -m benchmarks.sim_tree_coding [repeats]
"""
import sys
import warnings

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.model_selection import train_test_split

from benchmarks import common


def one(rep: int, coding: str) -> list:
    logger.remove()
    warnings.simplefilter("ignore")
    X, y = common.load_dataset("sim_tree", rep)
    if coding == "numeric":
        X = X.assign(x1=X["x1"].astype(float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=1000 + rep)
    X_train, X_test = common.prepare(X_train, X_test)
    rows = []
    for name in ("logit", "tree", "lrtree"):
        result = common.METHODS[name](X_train, X_test, y_train, rep, 100)
        rows.append({"x1": coding, "method": name, "gini": common.gini(y_test, result["p"]),
                     "segments": result["segments"]})
    return rows


if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    runs = Parallel(n_jobs=-1)(delayed(one)(rep, coding) for rep in range(1, repeats + 1)
                               for coding in ("onehot", "numeric"))
    results = pd.DataFrame([row for rows in runs for row in rows])
    print(results.groupby(["x1", "method"]).agg(gini=("gini", "mean"), segments=("segments", "mean")).round(3))
