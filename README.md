[![PyPI version](https://badge.fury.io/py/lrtree.svg)](https://badge.fury.io/py/lrtree)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lrtree.svg)](https://pypi.python.org/pypi/lrtree/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/lrtree)](https://img.shields.io/pypi/dm/lrtree)
[![Build Status](https://travis-ci.org/adimajo/lrtree.svg?branch=master)](https://travis-ci.org/adimajo/lrtree)
![Python package](https://github.com/adimajo/lrtree/workflows/Python%20package/badge.svg)
[![codecov](https://codecov.io/gh/adimajo/lrtree/branch/master/graph/badge.svg)](https://codecov.io/gh/adimajo/lrtree)

# Logistic regression trees

Table of Contents
-----------------

* [Documentation](https://adimajo.github.io/lrtree)
* [Installation instructions](#-installing-the-package)
* [Theory](#-use-case-example)
* [Some examples](#-the-lrtree-package)
* [Open an issue](https://github.com/adimajo/lrtree/issues/new/choose)
* [References](#-references)
* [Contribute](#-contribute)

## Motivation

The goal of `lrtree` is to build decision trees with logistic regressions at their leaves, so that the resulting model mixes non parametric VS parametric and stepwise VS linear approaches to have the best predictive results, yet maintaining interpretability.

This is the implementation of glmtree as described in *Formalization and study of statistical problems in Credit Scoring*, Ehrhardt A. (see [manuscript](https://github.com/adimajo/manuscrit_these) or [web article](https://adimajo.github.io/logistic_trees.html))

## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This code is supported on Python 3.7, 3.8, 3.9.

### Installing the package

#### Installing the development version

If `git` is installed on your machine, you can use:

```PowerShell
pipenv install git+https://github.com/adimajo/lrtree.git
```

If `git` is not installed, you can also use:

```PowerShell
pipenv install --upgrade https://github.com/adimajo/lrtree/archive/master.tar.gz
```

#### Installing through the `pip` command

You can install a stable version from [PyPi](https://pypi.org/project/lrtree/) by using:

```PowerShell
pip install lrtree
```

#### Installation guide for Anaconda

The installation with the `pip` or `pipenv` command **should** work. If not, please raise an issue.

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

You can install all dependencies, including development dependencies, using (note that 
this command requires `pipenv` which can be installed by typing `pip install pipenv`):

```PowerShell
pipenv install -d
```

You can build the documentation by going into the `docs` directory and typing `make html`.

You can run the tests by typing `coverage run -m pytest`, which relies on packages 
[coverage](https://coverage.readthedocs.io/en/coverage-5.2/) and [pytest](https://docs.pytest.org/en/latest/).

To run the tests in different environments (one for each version of Python), install `pyenv` (see [the instructions here](https://github.com/pyenv/pyenv)),
install all versions you want to test (see [tox.ini](tox.ini)), e.g. with `pyenv install 3.7.0` and run 
`pipenv run pyenv local 3.7.0 [...]` (and all other versions) followed by `pipenv run tox`.
 
## Python Environment 
The project uses `pipenv`. [An interesting resource](https://realpython.com/pipenv-guide/).

To download all the project dependencies in order to then port them to a machine that had limited access to the internet, you must use the command
`pipenv lock -r > requirements.txt` which will transform the `Pipfile` into a `requirements.txt`.

## Installation
To install a virtual environment as well as all the necessary dependencies, you must use the `pipenv install` command for production use 
or the command `pipenv install -d` for development use.

## Tests

The tests are based on `pytest` and are stored in the `tests` folder. They can all be launched with the command
`pytest` in at the root of the project.
The test coverage can be calculated thanks to the `coverage` package, which is also responsible for launching the tests. 
The command to use is `coverage run -m pytest`. We can then obtain a graphic summary in the form of an HTML page 
using the `coverage html` command which creates or updates the `htmlcov` folder from which we can open the `index.html` file.

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
