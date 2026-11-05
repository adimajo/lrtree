# lrtree

Python package (from a PhD thesis): logistic regression trees, i.e. a decision tree whose leaves / segments hold
logistic regressions, fitted by a stochastic EM (SEM) algorithm. Entry point: `lrtree.Lrtree`
(`fit`, `predict`, `predict_proba`, `precision`). It is the Python port of the R package `../glmtree`
(same conventions, same benchmark: see `benchmarks/`). Sister packages: `../glmdisc`, `../glmdisc_python`
(tooling in `pyproject.toml`, `.flake8` and the GitHub workflow is copied from the latter).

## Layout

| File | Role |
|---|---|
| `lrtree/__init__.py` | `Lrtree` class (arguments, `_check_input_args`, `_set_early_stopping`, `_check_is_fitted`), `NotFittedError`, rule constants |
| `lrtree/fit.py` | `fit()`, the SEM chain `_fit_sem` and EM chain `_fit_em`, criterion, tree fit/pruning, c_hat draw, `_segments`, `_fit_parallelized` |
| `lrtree/predict.py` | `predict`, `predict_proba`, `precision` (vectorised, one regression per segment present in X) |
| `lrtree/logreg.py` | `PossiblyOneClassReg` (pure segments), `LogRegSegment` (forwards sklearn kwargs; optional per-segment discretisation) |
| `lrtree/discretization.py` | MDLP discretisation, chi2 level merging (`grouping`, `green_clust`), `Processing` class, one-hot/scaling |
| `lrtree/generate_data.py` | toy data (`Lrtree.generate_data`, static) with a known 4-leaf tree |
| `benchmarks/` | Python port of glmtree's benchmark (`real_data.py`, `common.py`, `export_datasets.R`, `data/`, `reference/`, `results/`) |
| `lrtree/_command_line/` | the `lrtree-consistency` and `lrtree-realdata` console scripts (`pip install lrtree[scripts]`); `realdataopen` downloads UCI / Kaggle data |
| `scripts/` | experiments on private datasets (Kaggle, Vertica, `N:\` paths): not runnable here, only kept importable |
| `tests/` | pytest; `conftest.py` forwards loguru records to `caplog` |

## Key design points (read before changing the SEM loop)

- `c_hat` is the *sampled* segment, `c_map` the tree's segment (`_segments`: `link.classes_[argmax(proba)]`, or
  `link.apply(X)` with `leaves_as_segment`). `_segments` is the single definition of c_map, used by fit **and**
  predict. Never use the argmax position as a label: `link.classes_` can be a strict subset of the labels.
- Segment labels are integers kept for the whole chain (never recoded). `best_logreg` is a `dict` label -> model;
  observations of a label without model fall back to `global_logreg`.
- The criterion of iteration `i` is computed on `c_map` with the tree of iteration `i - 1` (the one that produced
  it), and that pair (tree, models) is what is stored as best. A model needs a tree, hence `max_iter=1` gives no
  usable model (`predict` raises `NotFittedError`). Iterations before `burn_in` cannot be the best, unless the
  chain never gets past it (then the best model seen is kept).
- The likelihood in the draw of `c_hat` uses `predict_proba[:, 1]` (never hard labels), floored at `1e-12`.
- Criteria are maximised: Gini; log-likelihood on a validation set; `-AIC` / `-BIC` summed over segments on the
  training set (the BIC penalty of a segment uses the log of its size, like glmtree). `ratios[0]` is the **train**
  share (glmtree's `proportions` is the held-out share).
- Segment models are fresh `LogRegSegment` objects at each iteration: the best ones must not be refitted later.
- Randomness: everything goes through `self._rng = default_rng(random_state)`, created at the start of `fit`
  (same `random_state` = same fit). `_fit_parallelized` derives one seed per initialisation.
- `categorical=[names]` (argument of `fit`), `discretize` and `group` (arguments of `Lrtree`) need a DataFrame. The
  regressions of the segments get the raw features (object array, selected by name) and treat them *inside each
  segment* (`Processing`, owned by `LogRegSegment`: merging of levels, MDLP discretisation, one-hot, scaling); the
  segmentation tree gets numerical features as they are and one-hot encoded categorical ones (`_tree_features`,
  encoder fitted once in `_init_fit`). Missing levels are a level of their own (`as_categories`).
- `fit(solver=...)` overrides the solver of `logreg_kwargs`; the positional order of `fit` is upstream's
  (`X, y, solver, nb_init, ...`), my additions (`min_samples_leaf`) come last.

## Commands

```bash
uv venv .venv && uv pip install -e ".[test,docs,bench]"
.venv/bin/pytest -q                                  # DeprecationWarning / FutureWarning are errors
.venv/bin/flake8 .                                   # config in .flake8 (scripts/ has old findings)
.venv/bin/coverage run --source lrtree -m pytest && .venv/bin/coverage report
.venv/bin/python -m build && .venv/bin/twine check dist/*
.venv/bin/sphinx-build docs/source docs/build/html   # needs the docs extra
.venv/bin/python scripts/profiling.py                # times one fit (n = 10000)
# benchmark: glmtree's protocol in Python (~25 min on 8 cores); writes results.csv, summary.txt, comparison.csv
.venv/bin/python -m benchmarks.real_data 20 100 benchmarks/results
# datasets of the benchmark come from R packages; they are committed, regenerate only if needed
R_LIBS_USER=$HOME/R/library Rscript benchmarks/export_datasets.R benchmarks/data
```

Pitfalls:
- Tests and experiments must pass `random_state`; the chain is stochastic. A single chain often lands in a local
  optimum: use `nb_init` / `_fit_parallelized` (see `lrtree/_command_line/consistency.py`).
- Fit time is dominated by scikit-learn's `LogisticRegression` (one fit per segment and iteration): vectorising
  our own code does not change the wall time much. Profile before optimising.
- Everything is numpy inside the chain (no feature names, so no sklearn feature-name warnings); DataFrames are
  converted positionally in `fit` and `predict`.
- pandas 3 copy-on-write: no chained assignment, no `inplace=True` on a column; sklearn 1.8 deprecates
  `penalty=` in `LogisticRegression` (tests use `C=` only).
- `creditcard.csv` at the repo root is a 144 MB untracked Kaggle file: it is gitignored, never commit it.
- `.gitlab-ci.yml` and `.pypirc` target internal infrastructure (python 3.8 image, pipenv, `python setup.py
  sdist bdist_wheel`) and are deliberately untouched: hence `Pipfile`, `requirements*.txt` and the `setup.py` shim
  (metadata are in `pyproject.toml`) are kept.
- `tikzplotlib` (used by `_command_line/consistency.py`) does not import with matplotlib 3.11: not a test dependency.

## Conventions

- flake8, 120 columns; numpydoc/Sphinx docstrings (`:param type name:`); loguru for logging (no `print`).
- Keep the public names `Lrtree`, `NotFittedError` and the `fit`/`predict` signatures stable; the private
  helpers in `fit.py` are imported by the tests.
- No claims in the docs / README / NEWS without a number from `benchmarks/results/`.
- Authors: Adrien Ehrhardt, Dmitry Gaynullin, Elise Bayraktar.
