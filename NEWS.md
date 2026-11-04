# lrtree 1.1.0

This release makes `lrtree` work again on current Python stacks (Python 3.10-3.13, pandas 3, NumPy 2,
scikit-learn 1.4-1.9) and fixes several errors in the SEM / EM chains found by comparing with `glmtree`. Models
fitted with 1.0.0 give different results and the objects are not compatible.

## Breaking changes

* `Lrtree.best_logreg` is a `dict` keyed by segment label (it was a list indexed by position, which was wrong
  whenever a segment was absent from the data to predict). `best_link` is `None` (not `[]`) before fitting.
* `Lrtree.best_treatment` is gone: the per-segment treatment lives in the `Processing` object of each
  `LogRegSegment`.
* `burn_in` iterations can no longer be selected as the best model (they only counted for early stopping before).
* `predict` returns the labels given when fitting (it returned 0 / 1).
* Python >= 3.10, pandas >= 2.0, scikit-learn >= 1.4. `setup.py`, `Pipfile` and Travis are replaced by
  `pyproject.toml` and a GitHub Actions workflow.

## Bug fixes

* **`predict` / `predict_proba`** used the position of a segment among those *present in the data* to pick its
  regression, and `link.predict` (a class label) where fit used leaf positions with `leaves_as_segment=True`. A
  single `_segments()` helper is now used by both. They also crashed on pandas 3.
* **SEM drew the segments from hard 0 / 1 predictions** instead of probabilities, so likelihoods collapsed to
  0 / 1 and the rows of the draw could be 0 / 0 (`invalid value encountered in divide`).
* The tree columns were taken as segment labels (`argmax` is a position in `link.classes_`) and the likelihood
  columns were selected with `df["c_hat"].unique()`, which is in order of appearance: both misaligned segments
  as soon as a segment disappeared.
* `LogRegSegment` silently dropped all its keyword arguments (`penalty`, `solver`, `C`, `tol`...).
* `PossiblyOneClassReg.predict` returned `None` for pure segments, kept its single class after a refit, and
  EM called `fit` with a signature that did not exist.
* Cost-complexity pruning never updated its best score (the last alpha won) and dropped `max_depth`.
* The BIC penalty was indexed by position, not segment (wrong when a segment had no observations); the AIC
  penalty lacked its factor 2 and ignored the intercept. With a validation set, AIC / BIC are now the plain
  log-likelihood, as announced by the warning.
* Early stopping rules were crossed (`low improvement` only for the Gini, `low variation` only for AIC / BIC);
  EM never incremented its iteration counter inside the loop. A second `fit` kept the best criterion and the
  criterion history of the first one.
* `_check_args` was skipped when `data_treatment=True`, iterated over every element of `y`, and a `pd.Series`
  `y` with a non-default index was misaligned.
* `grouping` assigned through `.iloc` with a boolean mask (an error on pandas 3), `np.in1d`, `np.NaN`,
  `OneHotEncoder(sparse=...)` and chained `inplace` calls were removed. `extreme_values` (sentinels such as 999
  become missing then 0) is now opt-in in `Processing` (`clean_extremes`), and the column names of the in-house
  credit datasets are only handled with `legacy_schema=True`.
* `import lrtree` no longer sets `warnings.filterwarnings` and a pandas option globally; the convergence
  warnings are silenced only while fitting.

## Performance

* The tree data of EM is built with `np.repeat` / `np.tile` instead of a double Python loop, the likelihood
  masks use `np.where`, `predict` is vectorised (no `deepcopy`, no per-row `.loc`), the data are converted once
  per fit and the c_hat regressions are reused as c_map regressions when the two segmentations are equal.
* The fit time is dominated by scikit-learn's `LogisticRegression`: on `scripts/profiling.py` (n = 10000,
  100 iterations, 4 segments) a fit takes 4.8 s before and after. EM and `predict` could not be timed before
  because they raised errors.

## New

* `Lrtree(random_state=...)` for reproducible fits; `Lrtree(logreg_kwargs=...)` to configure the regressions;
  `fit(min_samples_leaf=...)`; `Lrtree.generate_data` uses `numpy.random.default_rng`.
* `benchmarks/`: Python port of glmtree's benchmark with a comparison of the results (see `benchmarks/README.md`).
* `CLAUDE.md` describes the design and the commands.
