[![PyPI version](https://badge.fury.io/py/lrtree.svg)](https://badge.fury.io/py/lrtree)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lrtree.svg)](https://pypi.python.org/pypi/lrtree/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/lrtree)](https://img.shields.io/pypi/dm/lrtree)
![Python package](https://github.com/adimajo/lrtree/workflows/Python%20package/badge.svg)
[![codecov](https://codecov.io/gh/adimajo/lrtree/branch/master/graph/badge.svg)](https://codecov.io/gh/adimajo/lrtree)

# Logistic regression trees

Table of Contents
-----------------

* [Documentation](https://adimajo.github.io/lrtree)
* [Installation instructions](#-installing-the-package)
* [Theory](https://adimajo.github.io/logistic_trees.html)
* [Some examples](#-utilization)
* [Open an issue](https://github.com/adimajo/lrtree/issues/new/choose)
* [References](#-references)
* [Contribute](#-contribute)

## Motivation

The goal of `lrtree` is to build decision trees with logistic regressions at their leaves, so that the resulting model mixes non parametric VS parametric and stepwise VS linear approaches to have the best predictive results, yet maintaining interpretability.

This is the implementation of glmtree as described in *Formalization and study of statistical problems in Credit Scoring*, Ehrhardt A. (see [manuscript](https://github.com/adimajo/manuscrit_these) or [web article](https://adimajo.github.io/logistic_trees.html))

## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This code is supported on Python 3.10, 3.11, 3.12 and 3.13.

### Installing the package

#### Installing the development version

If `git` is installed on your machine, you can use:

```PowerShell
pip install git+https://github.com/adimajo/lrtree.git
```

If `git` is not installed, you can also use:

```PowerShell
pip install --upgrade https://github.com/adimajo/lrtree/archive/master.tar.gz
```

#### Installing through the `pip` command

You can install a stable version from [PyPi](https://pypi.org/project/lrtree/) by using:

```PowerShell
pip install lrtree
```

To run the provided scripts, `lrtree-consistency` and `lrtree-realdata`, you need a
few additional dependencies:
```bash
pip install lrtree[scripts]
```

#### Installation guide for Anaconda

The installation with the `pip` command **should** work. If not, please raise an issue.

#### For people behind proxy(ies)...

A lot of people, including myself, work behind a proxy at work...

A simple solution to get the package is to use the `--proxy` option of `pip`:

```PowerShell
pip --proxy=http://username:password@server:port install lrtree
```

where *username*, *password*, *server* and *port* should be replaced by your own values.

If environment variables `http_proxy` and / or `https_proxy` and / or (unfortunately depending on applications...) 
`HTTP_PROXY` and `HTTPS_PROXY` are set, the proxy settings should be picked up by `pip`.

Over the years, I've found [CNTLM](http://cntlm.sourceforge.net/) to be a great tool in this regard.

## Authors

* [Adrien Ehrhardt](https://adimajo.github.io)
* [Vincent Vandewalle](https://sites.google.com/site/vvandewa/)
* [Philippe Heinrich](http://math.univ-lille1.fr/~heinrich/)
* [Christophe Biernacki](http://math.univ-lille1.fr/~biernack/)
* Dmitri Gaynullin
* Elise Bayraktar

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research has been financed by [Crédit Agricole Consumer Finance](https://www.ca-consumerfinance.com/en.html) through a CIFRE PhD.

This research was supported by [Inria Lille - Nord-Europe](https://www.inria.fr/centre/lille) and [Lille University](https://www.univ-lille.fr/en/home/) as part of a PhD.

## References

Ehrhardt, A. (2019), [Formalization and study of statistical problems in Credit Scoring: Reject inference, discretization and pairwise interactions, logistic regression trees](https://hal.archives-ouvertes.fr/tel-02302691) ([PhD thesis](https://github.com/adimajo/manuscrit_these)).

## Contribute

You can clone this project using:

```PowerShell
git clone https://github.com/adimajo/lrtree.git
```

You can install the package with its development dependencies in a virtual environment using
[uv](https://docs.astral.sh/uv/) (or plain `python -m venv` and `pip`):

```bash
uv venv .venv && uv pip install -e ".[test,docs,bench]"
```

## Tests

The tests are based on `pytest` and are stored in the `tests` folder. They can all be launched with the command
`pytest` at the root of the project.
The test coverage is computed with `coverage run --source lrtree -m pytest`, followed by `coverage report`
(or `coverage html`, which creates the `htmlcov` folder from which `index.html` can be opened).
The code is linted with `flake8` (configuration in `.flake8`).

## Documentation

You can build the documentation with `sphinx-build docs/source docs/build/html` (needs the `docs` extra).

## Benchmarks

`benchmarks/` reproduces the benchmark of the R package [glmtree](https://github.com/adimajo/glmtree) in Python,
see [benchmarks/README.md](benchmarks/README.md).

## Utilization
The package provides [sklearn-like](https://scikit-learn.org/stable) interface.

Loading [sample data](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) for regression task:

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

The trained model consists of a fitted `sklearn.tree.DecisionTreeClassifier` class for segmentation of a data and
`sklearn.linear_model.LogisticRegression` regressions for each node a of a tree in a form of python list.

The snippet to train the model and make a prediction:
```python
from lrtree import Lrtree

model = Lrtree(criterion="bic", ratios=(0.7,), class_num=2, max_iter=100)

# Fitting the model
model.fit(X_train, y_train)

# Make a prediction on a fitted model
model.predict(X_test)
```

If you installed the additional dependencies for scripts, you can also run directly from the command line:
```bash
LOGURU_LEVEL="ERROR" DEBUG="True" lrtree-consistency
```

or
```bash
LOGURU_LEVEL="ERROR" TQDM_DISABLE="1" lrtree-realdata
```

Beware: if you don't set `LOGURU_LEVEL` then it is implicitly set on DEBUG
which will yield a lot of prints. Also, both scripts will take very long
to complete as they test the consistency of the method for various
hyperparameters and run cross-validation on 3 real datasets respectively.
