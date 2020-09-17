# The `glmtree` package
This repository contains the implementation of Glmtree algorithm for data classification.

Glmtree is a logistic regression trees based approach that is described in [this thesis paper](https://hal.archives-ouvertes.fr/tel-02302691/) .


## Python Environment 
The project uses `pipenv`. [An interesting resource](https://realpython.com/pipenv-guide/).

To download all the project dependencies in order to then port them to a machine that had limited access to the internet, you must use the command
`pipenv lock -r> requirements.txt` which will transform the `Pipfile` into a `requirements.txt`.


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
from glmtree import *

model = Glmtree(criterion="bic", ratios=(0.7,), class_num=2, max_iter=100)

# Fitting the model
model.fit(X_train, y_train)

# Make a prediction on a fitted model
model.predict(X_test)
```


