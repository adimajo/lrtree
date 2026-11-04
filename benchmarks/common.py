"""
Shared pieces of the benchmark (port of glmtree's ``inst/benchmarks/common.R``): datasets, competitors and the
repetition runner.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from lrtree import Lrtree

DATA_DIR = Path(__file__).parent / "data"
REFERENCE_DIR = Path(__file__).parent / "reference"
REAL_DATASETS = {
    "german": "German credit (UCI Statlog): 1000 loans, 7 numeric + 13 categorical features",
    "pima": "Pima Indians diabetes (MASS::Pima.tr + Pima.te): 532 women, 7 numeric features",
    "default": "ISLR::Default, simulated credit card default: 10000 customers, 3 features",
    "creditcard": "AER::CreditCard: 1319 card applications (share and expenditure removed: they leak the outcome)",
    "caravan": "ISLR::Caravan: 5822 insurance customers, 85 features, 6% positives",
}
SIMULATED_DATASETS = {
    "sim_tree": "Simulation ('tree'): 3 segments, each with its own logistic regression, n = 3000",
    "sim_notree": "Simulation ('no tree'): 3 clusters of features, a single logistic regression, n = 3000",
}
# Name of the glmtree method each of our methods stands for
REFERENCE_METHODS = {"logit": "logit", "tree": "ctree", "forest": "forest", "lrtree": "glmtree",
                     "lrtree_valid": "glmtree_valid"}


# ---- datasets -------------------------------------------------------------------------

def generate_glmtree_data(n: int, scenario: str, rng: np.random.Generator) -> tuple:
    """
    Port of glmtree's ``generateData``: returns the features and the labels.

    - "tree": three clusters, each with its own pair of levels of the categorical ``x1`` and its own logistic
      regression on (x2, x3)
    - "no tree": three apparent clusters of the features, but a single logistic regression
    """
    n_clusters = 3
    # equal sized clusters, the remainder goes to the last one
    cluster = np.concatenate([np.repeat(np.arange(1, n_clusters + 1), n // n_clusters),
                              np.full(n % n_clusters, n_clusters)])
    if scenario == "tree":
        theta = np.array([[0, -1, 0.5], [0, -0.5, 1.5], [0, 1, -0.5]])
        x1 = 2 * cluster - 1 + rng.binomial(1, 0.5, n)
        x2 = rng.normal(0, 1.5, n)
        x3 = rng.normal(0, 1.5, n)
        log_odd = theta[cluster - 1, 0] + theta[cluster - 1, 1] * x2 + theta[cluster - 1, 2] * x3
        X = pd.DataFrame({"x1": x1.astype(str), "x2": x2, "x3": x3})
    elif scenario == "no tree":
        theta = np.array([3, 0.5, -1])
        x1 = rng.normal(3 * cluster, 1)
        x2 = rng.normal(3 * cluster, 1)
        log_odd = theta[0] + theta[1] * x1 + theta[2] * x2
        X = pd.DataFrame({"x1": x1, "x2": x2})
    else:
        raise ValueError(f"Invalid scenario {scenario}.")
    y = rng.binomial(1, 1 / (1 + np.exp(-log_odd)))
    return X, y


def load_dataset(name: str, rep: int) -> tuple:
    """
    The features and labels of a dataset; the simulated ones are redrawn for each repetition.
    """
    if name in SIMULATED_DATASETS:
        scenario = "tree" if name == "sim_tree" else "no tree"
        return generate_glmtree_data(3000, scenario, np.random.default_rng(90000 + rep))
    data = pd.read_csv(DATA_DIR / f"{name}.csv.gz")
    return data.drop(columns="y"), data["y"].to_numpy()


# ---- helpers ----------------------------------------------------------------------------

def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def min_bucket(n_train: int) -> int:
    """Leaves of at least 5% of the training data"""
    return max(20, int(0.05 * n_train))


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Turns the features into numeric arrays (lrtree needs numeric features), fitting everything on the training data:
    median imputation of numeric features, missing level for categorical ones, unseen levels in the test set
    replaced by the most frequent level, dummy coding (first level dropped) and standardisation of the numerical
    features (the optimisers of scikit-learn, unlike R's ``glm``, are not scale invariant).
    """
    train, test = train.copy(), test.copy()
    numeric = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 1]
    categorical = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    for column in numeric:
        median = train[column].median()
        train[column] = train[column].fillna(median)
        test[column] = test[column].fillna(median)
        mean, std = train[column].mean(), train[column].std(ddof=0)
        train[column] = (train[column] - mean) / std
        test[column] = (test[column] - mean) / std
    pieces_train, pieces_test = [train[numeric]], [test[numeric]]
    for column in categorical:
        train_values = train[column].fillna("missing").astype(str)
        levels = sorted(train_values.unique())
        mode = train_values.value_counts().index[0]
        test_values = test[column].fillna("missing").astype(str)
        test_values = test_values.where(test_values.isin(levels), mode)
        for values, pieces in ((train_values, pieces_train), (test_values, pieces_test)):
            dummies = pd.get_dummies(pd.Categorical(values, categories=levels), prefix=column, dtype=float)
            dummies.index = values.index
            pieces.append(dummies.iloc[:, 1:])
    return pd.concat(pieces_train, axis=1).to_numpy(float), pd.concat(pieces_test, axis=1).to_numpy(float)


# ---- competitors ------------------------------------------------------------------------

def method_logit(X_train, X_test, y_train, rep, sem_iter):
    model = LogisticRegression(C=np.inf, max_iter=2000).fit(X_train, y_train)
    return {"p": model.predict_proba(X_test)[:, 1], "segments": 1}


def method_tree(X_train, X_test, y_train, rep, sem_iter):
    # stand-in for partykit::ctree
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=min_bucket(len(y_train)), random_state=rep)
    tree.fit(X_train, y_train)
    return {"p": tree.predict_proba(X_test)[:, 1], "segments": tree.get_n_leaves()}


def method_forest(X_train, X_test, y_train, rep, sem_iter):
    # stand-in for ranger; probability forests of ranger have leaves of at least 10 observations
    forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, n_jobs=1, random_state=rep)
    forest.fit(X_train, y_train)
    return {"p": forest.predict_proba(X_test)[:, 1], "segments": np.nan}


def _lrtree(X_train, X_test, y_train, rep, sem_iter, **kwargs):
    model = Lrtree(class_num=5, max_iter=sem_iter, burn_in=min(50, sem_iter // 4), random_state=5000 + rep,
                   **kwargs)
    model.fit(X_train, y_train, tree_depth=4, min_samples_leaf=min_bucket(len(y_train)), optimal_size=False)
    return {"p": model.predict_proba(X_test), "segments": len(model.best_logreg),
            "best_iteration": int(np.argmax(model.criterion_iter))}


def method_lrtree(X_train, X_test, y_train, rep, sem_iter):
    return _lrtree(X_train, X_test, y_train, rep, sem_iter, criterion="bic")


def method_lrtree_valid(X_train, X_test, y_train, rep, sem_iter):
    return _lrtree(X_train, X_test, y_train, rep, sem_iter, criterion="gini", validation=True, ratios=(0.75,))


METHODS = {"logit": method_logit, "tree": method_tree, "forest": method_forest, "lrtree": method_lrtree,
           "lrtree_valid": method_lrtree_valid}


# ---- one repetition ---------------------------------------------------------------------

def run_repetition(dataset_name: str, rep: int, methods: list, sem_iter: int) -> pd.DataFrame:
    """
    One stratified 70/30 split of a dataset and all the methods on it (same protocol as glmtree's benchmark).
    """
    logger.remove()
    X, y = load_dataset(dataset_name, rep)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=1000 + rep)
    X_train, X_test = prepare(X_train, X_test)
    rows = []
    for name in methods:
        row = {"dataset": dataset_name, "rep": rep, "method": name, "gini": np.nan, "segments": np.nan,
               "best_iteration": np.nan, "secs": np.nan, "note": ""}
        start = time.perf_counter()
        try:
            result = METHODS[name](X_train, X_test, y_train, rep, sem_iter)
            row["secs"] = time.perf_counter() - start
            ok = ~np.isnan(result["p"])
            row["gini"] = gini(y_test[ok], result["p"][ok])
            row["segments"] = result["segments"]
            row["best_iteration"] = result.get("best_iteration", np.nan)
        except Exception as error:  # a failing method is reported in the results, as in the R benchmark
            row["secs"] = time.perf_counter() - start
            row["note"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---- summaries ----------------------------------------------------------------------------

def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per dataset and method: mean / sd of the Gini and the paired difference with the logistic regression.
    """
    rows = []
    for dataset, sub in results.groupby("dataset", sort=False):
        reference = sub[sub["method"] == "logit"].set_index("rep")["gini"]
        for method, part in sub.groupby("method", sort=False):
            difference = part.set_index("rep")["gini"] - reference
            rows.append({"dataset": dataset, "method": method, "n_ok": int(part["gini"].notna().sum()),
                         "gini_mean": part["gini"].mean(), "gini_sd": part["gini"].std(),
                         "diff_vs_logit": difference.mean(),
                         "diff_se": difference.std() / np.sqrt(difference.notna().sum()),
                         "segments": part["segments"].mean(), "secs": part["secs"].mean()})
    return pd.DataFrame(rows)


def compare_with_glmtree(results: pd.DataFrame) -> pd.DataFrame:
    """
    Mean Gini of our methods next to the corresponding method of glmtree (``benchmarks/reference``).

    A method agrees with glmtree when the means differ by at most max(0.03, 2 combined standard errors).
    """
    reference = pd.read_csv(REFERENCE_DIR / "glmtree_results.csv")
    rows = []
    for (dataset, method), part in results.groupby(["dataset", "method"], sort=False):
        if method not in REFERENCE_METHODS:
            continue
        ref = reference[(reference["dataset"] == dataset) & (reference["method"] == REFERENCE_METHODS[method])]
        if ref.empty:
            continue
        se = np.sqrt(part["gini"].var() / part["gini"].notna().sum() + ref["gini"].var() / ref["gini"].notna().sum())
        delta = part["gini"].mean() - ref["gini"].mean()
        rows.append({"dataset": dataset, "method": method, "glmtree_method": REFERENCE_METHODS[method],
                     "python_gini": part["gini"].mean(), "glmtree_gini": ref["gini"].mean(), "delta": delta,
                     "tolerance": max(0.03, 2 * se), "agrees": bool(abs(delta) <= max(0.03, 2 * se)),
                     "python_segments": part["segments"].mean(), "glmtree_segments": ref["segments"].mean()})
    return pd.DataFrame(rows)


def qualitative_checks(summary: pd.DataFrame) -> list:
    """
    The qualitative claims of the glmtree benchmark, for the datasets in ``summary``: (description, holds) pairs.
    """
    def diff(dataset):
        sub = summary[(summary["dataset"] == dataset) & (summary["method"] == "lrtree")]
        return float(sub["diff_vs_logit"].iloc[0]) if len(sub) else None

    claims = [("sim_tree", "sim_tree: lrtree beats the logistic regression by at least 0.30", lambda d: d >= 0.30),
              ("sim_notree", "sim_notree: lrtree is on par with the logistic regression (|diff| <= 0.01)",
               lambda d: abs(d) <= 0.01)]
    claims += [(name, f"{name}: lrtree does not beat the logistic regression (as in glmtree)", lambda d: d <= 0.01)
               for name in ("german", "pima", "caravan")]
    return [(description, bool(holds(diff(dataset)))) for dataset, description, holds in claims
            if diff(dataset) is not None]
