"""
Benchmark of lrtree against a single logistic regression and other tree-based methods, on simulated data and real
datasets. Python port of glmtree's ``inst/benchmarks/real_data.R``.

Usage:  python -m benchmarks.real_data [repeats] [iter] [outdir] [--datasets a b] [--methods x y] [--jobs n]

  repeats  number of random stratified 70/30 splits per dataset (default 20)
  iter     number of SEM iterations of lrtree (default 100)
  outdir   where to write results.csv, summary.txt and comparison.csv (default: benchmarks/results)

The script is not part of the package tests: it takes a few tens of minutes.
"""
import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from benchmarks.common import (METHODS, REAL_DATASETS, SIMULATED_DATASETS, compare_with_glmtree,
                               qualitative_checks, run_repetition, summarize)


def run(repeats: int, sem_iter: int, datasets: list, methods: list, jobs: int) -> pd.DataFrame:
    tasks = [(dataset, rep) for dataset in datasets for rep in range(1, repeats + 1)]
    runs = Parallel(n_jobs=jobs)(
        delayed(run_repetition)(dataset, rep, methods, sem_iter) for dataset, rep in tqdm(tasks, desc="Repetitions"))
    return pd.concat(runs, ignore_index=True)


def write_report(results: pd.DataFrame, outdir: Path, repeats: int, sem_iter: int) -> str:
    summary = summarize(results)
    comparison = compare_with_glmtree(results)
    results.to_csv(outdir / "results.csv", index=False)
    comparison.to_csv(outdir / "comparison.csv", index=False)
    rounded = summary.round({"gini_mean": 3, "gini_sd": 3, "diff_vs_logit": 3, "diff_se": 3, "segments": 1,
                             "secs": 1})
    lines = [
        f"lrtree benchmark: {repeats} random stratified 70/30 splits per dataset, {sem_iter} SEM iterations",
        "gini = 2*AUC-1 on the held-out 30%; diff_vs_logit = paired difference with the plain logistic regression "
        "(mean, standard error)",
        "segments = number of leaves / segments; tree-based methods use leaves of at least 5% of the training "
        "data and depth 4",
        "tree and forest stand for glmtree's ctree and ranger; mob has no Python equivalent and is omitted",
        "", rounded.to_string(index=False), "",
        "Comparison with the results of glmtree (same protocol, R); agrees = |delta| <= max(0.03, 2 se)", "",
        comparison.round(3).to_string(index=False), "", "Qualitative claims of glmtree's benchmark", ""]
    lines += [f"[{'PASS' if holds else 'FAIL'}] {description}" for description, holds in qualitative_checks(summary)]
    report = "\n".join(lines)
    (outdir / "summary.txt").write_text(report + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repeats", nargs="?", type=int, default=20)
    parser.add_argument("iter", nargs="?", type=int, default=100)
    parser.add_argument("outdir", nargs="?", default="benchmarks/results")
    parser.add_argument("--datasets", nargs="+", default=[*SIMULATED_DATASETS, *REAL_DATASETS])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # the paired differences are with respect to the logistic regression
    methods = ["logit"] + [method for method in args.methods if method != "logit"]
    results = run(args.repeats, args.iter, args.datasets, methods, args.jobs)
    print(write_report(results, outdir, args.repeats, args.iter))


if __name__ == "__main__":
    main()
