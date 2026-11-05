# Reproducing glmtree's benchmark in Python

This is a port of `../glmtree/inst/benchmarks/real_data.R` (R package [glmtree](https://github.com/adimajo/glmtree)):
same datasets, same protocol (20 random stratified 70/30 splits, Gini = 2 AUC - 1 on the held-out 30%, leaves of at
least 5% of the training data, depth 4, 5 initial segments, 100 SEM iterations, BIC on the training set, and a
variant with a 25% validation set and the Gini).

```bash
.venv/bin/python -m benchmarks.real_data 20 100 benchmarks/results      # about 25 minutes on 8 cores
.venv/bin/python -m benchmarks.sim_tree_coding                          # see "Where the gap comes from"
```

`results/` holds the raw results (`results.csv`), the summary in glmtree's format (`summary.txt`) and the
comparison with glmtree (`comparison.csv`); `reference/` holds glmtree's own results, copied from
`../glmtree/inst/benchmarks/results`. The real datasets (R packages scorecard, MASS, ISLR, AER) were exported once
with `export_datasets.R` to `data/*.csv.gz` (350 KB in total); the simulations are redrawn from a port of
`generateData()`.

## Differences with the R protocol

- **Competitors**: `tree` (CART) stands for `partykit::ctree`, `forest` (scikit-learn) for `ranger`; `mob` has no
  Python equivalent and is omitted.
- **Features**: lrtree needs numeric features, so categorical ones are dummy coded (first level dropped, fitted on
  the training set), and numerical ones are standardised for every method. glmtree handles factors natively and
  `glm` is scale invariant.
- **Splits and seeds** are drawn with NumPy: the splits are not the ones of the R run, so only statistical
  agreement can be expected, not identical numbers.
- The regressions of lrtree are scikit-learn's default `LogisticRegression` (L2, `C=1`); `glm` is unpenalised.

## Results (20 splits, 100 iterations)

Mean Gini on the held-out 30%. "agrees" in `comparison.csv` means a difference of at most max(0.03, 2 standard errors).

| dataset | lrtree | glmtree | segments (lrtree / glmtree) | lrtree valid. | glmtree valid. |
|---|---|---|---|---|---|
| sim_tree | 0.597 | 0.650 | 4.0 / 4.1 | 0.577 | 0.644 |
| sim_notree | 0.670 | 0.672 | 3.1 / 1.6 | 0.668 | 0.670 |
| german | 0.487 | 0.514 | 3.8 / 1.5 | 0.467 | 0.393 |
| pima | 0.639 | 0.697 | 3.7 / 1.2 | 0.638 | 0.633 |
| default | 0.893 | 0.894 | 3.8 / 1.7 | 0.892 | 0.893 |
| creditcard | 0.619 | 0.667 | 3.2 / 1.4 | 0.633 | 0.668 |
| caravan | 0.363 | 0.413 | 3.2 / 1.3 | 0.337 | 0.379 |

- The qualitative claims of glmtree's benchmark **are reproduced**: on `sim_tree` lrtree beats the logistic
  regression by +0.355 (glmtree: +0.403), on `sim_notree` it is on par (-0.002), and on `german`, `pima` and
  `caravan` it does not beat the logistic regression.
- The logistic regression agrees with glmtree's on all 7 datasets; the `tree` and `forest` stand-ins on 10 of 14
  comparisons.
- **The numbers are not reproduced for lrtree itself**: 7 of the 14 lrtree comparisons agree within tolerance.
  lrtree is lower than glmtree on 12 of 14: by 0.03 to 0.07 on 8 of them and by less than 0.005 on the 4 others
  (`sim_notree`, `default`). It is higher on `german` with a validation set (0.467 against 0.393) and on `pima`
  with a validation set (+0.005). On the real datasets it keeps about 3.5 segments against 1.5 for glmtree.

## Where the gap comes from

- **Categorical features (explained for sim_tree).** The segments of `sim_tree` are pairs of levels of the
  categorical `x1`. ctree splits on subsets of levels; a CART tree on dummies needs several splits.
  `benchmarks.sim_tree_coding` (10 splits) codes `x1` as a number instead: lrtree goes from 0.595 to 0.633
  (glmtree: 0.650) and the CART stand-in from 0.384 to 0.580 (ctree: 0.568).
- **The regression penalty is not the cause.** With `logreg_kwargs={"C": np.inf}` (no penalty, like `glm`) the Gini on
  `sim_tree`, `pima` and `creditcard` (10 splits) is unchanged or lower (0.593 / 0.628 / 0.615 against 0.595 /
  0.664 / 0.616 with the default).
- **Not established**: the extra segments on numeric-only data such as `pima` (3.7 against 1.2). ctree stops
  splitting when no split is significant (`alpha = 0.1`), CART only stops on depth and leaf size, and the BIC of the
  training set then keeps finer segmentations. Nothing was tuned to close the gap; making the segmentation tree
  conditional-inference would be the next experiment.

The fit time of lrtree is 3 to 16 s per fit here (glmtree: 3 to 160 s; both measured with several fits running
in parallel), mostly in scikit-learn's `LogisticRegression`.
