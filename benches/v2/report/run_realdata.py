"""Real-dataset case-study driver.

For each (dataset, estimator) cell:
  1. Load X, y (and event for Cox).
  2. Run K-fold CV with a deep λ-grid: per fold, fit on train, evaluate
     deviance on test for every λ.
  3. Select λ_min by CV-deviance, λ_1se by glmnet's 1-SE rule.
  4. Refit on the full data at λ_min; record support size, β, runtime.

Output: one JSONL row per (dataset, estimator) plus an env.json sidecar.
Aggregator + figure builders (`F6_realdata_boxplots`, `T5_realdata`)
consume these directly.

Usage:
  python -m benches.v2.report.run_realdata --out-dir benches/v2/results/realdata
  python -m benches.v2.report.run_realdata --dataset riboflavin --estimator lasso
  python -m benches.v2.report.run_realdata --use-synthetic   # offline mode
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from benches.problems import Problem
from benches.v2.metrics import deviance as dev
from benches.v2.report import capture_env


# (dataset_id, family, [estimator names])
CASE_STUDIES: list[tuple[str, str, list[str]]] = [
    ("riboflavin", "gaussian",  ["lasso", "mcp", "scad", "elastic_net"]),
    ("leukemia",   "logistic",  ["lasso", "mcp", "elastic_net"]),
    ("bardet",     "gaussian",  ["group_lasso", "group_mcp"]),
    ("pbc",        "cox",       ["lasso", "mcp"]),
]


def _load_dataset(dataset_id: str, use_synthetic: bool) -> Problem:
    if dataset_id == "riboflavin":
        from benches.v2.datasets import riboflavin
        return riboflavin.load(use_synthetic=use_synthetic)
    if dataset_id == "leukemia":
        from benches.v2.datasets import leukemia
        return leukemia.load(use_synthetic=use_synthetic)
    if dataset_id == "bardet":
        from benches.v2.datasets import bardet
        return bardet.load(use_synthetic=use_synthetic)
    if dataset_id == "pbc":
        from benches.v2.datasets import tcga_brca
        return tcga_brca.load(use_synthetic=use_synthetic)
    raise KeyError(dataset_id)


def _lambda_grid(problem: Problem, n_lambdas: int = 50,
                 lambda_min_ratio: float | None = None) -> np.ndarray:
    """Grid construction follows glmnet's recipe.

    `lambda_min_ratio` defaults to 0.01 when n < p (high-dim regime),
    else 1e-3. This is the empirical glmnet convention and avoids the
    saturated-active-set tail in n≪p problems.
    """
    if lambda_min_ratio is None:
        lambda_min_ratio = 0.01 if problem.x.shape[0] < problem.x.shape[1] else 1e-3
    n = problem.x.shape[0]
    if problem.family == "gaussian":
        y_centered = problem.y - problem.y.mean()
    elif problem.family == "logistic":
        y_centered = problem.y - problem.y.mean()
    elif problem.family == "cox":
        y_centered = problem.meta["event"].astype(float) - problem.meta["event"].astype(float).mean()
    elif problem.family == "poisson":
        y_centered = problem.y - problem.y.mean()
    else:
        y_centered = problem.y - problem.y.mean()
    lambda_max = float(np.max(np.abs(problem.x.T @ y_centered)) / n)
    return np.geomspace(lambda_max, lambda_max * lambda_min_ratio, n_lambdas)


def _kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def _fit_path(problem: Problem, estimator: str, lambda_grid: np.ndarray,
              tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Return (coef_path (n_lambdas, p), intercept_path (n_lambdas,)).

    Routes to the skein runner; future variants can plug in skglm/etc.
    """
    from benches.v2.runners import skein_runner
    res = skein_runner.fit(problem, penalty=estimator, lambda_grid=lambda_grid,
                           tol=tol)
    intercept_path = res.intercept_path
    if intercept_path is None:
        intercept_path = np.zeros(res.coef_path.shape[0])
    return np.asarray(res.coef_path), np.asarray(intercept_path)


def _eval_deviance(problem: Problem, coef_path: np.ndarray,
                   intercept_path: np.ndarray,
                   x_test: np.ndarray, y_test: np.ndarray,
                   event_test: np.ndarray | None = None) -> np.ndarray:
    """Return deviance(λ) over the test fold."""
    eta = coef_path @ x_test.T + intercept_path[:, None]
    devs = np.zeros(eta.shape[0])
    for k in range(eta.shape[0]):
        devs[k] = dev.for_family(problem.family, y_test, eta[k], event=event_test)
    return devs


def run_one(dataset_id: str, estimator: str, *,
            cv_folds: int = 5, seed: int = 0,
            use_synthetic: bool = False,
            n_lambdas: int = 50) -> dict:
    problem = _load_dataset(dataset_id, use_synthetic=use_synthetic)
    grid = _lambda_grid(problem, n_lambdas=n_lambdas)

    fold_devs: list[np.ndarray] = []
    fold_supp: list[int] = []
    event = problem.meta.get("event") if problem.meta else None

    n = problem.x.shape[0]
    t_cv0 = time.perf_counter()
    for train_idx, test_idx in _kfold_indices(n, cv_folds, seed=seed):
        # Build per-fold sub-problem so the skein runner sees a self-consistent
        # design. Carry event status for Cox.
        sub_meta = dict(problem.meta or {})
        if event is not None:
            sub_meta["event"] = np.asarray(event)[train_idx]
        sub = Problem(
            x=problem.x[train_idx], y=problem.y[train_idx],
            beta_true=problem.beta_true,
            groups=problem.groups, family=problem.family, meta=sub_meta,
        )
        coef_path, int_path = _fit_path(sub, estimator, grid)
        event_test = np.asarray(event)[test_idx] if event is not None else None
        devs = _eval_deviance(
            problem, coef_path, int_path,
            problem.x[test_idx], problem.y[test_idx], event_test,
        )
        fold_devs.append(devs)
        fold_supp.append(int(np.count_nonzero(coef_path[-1])))
    t_cv = time.perf_counter() - t_cv0

    cv_mean = np.mean(fold_devs, axis=0)
    cv_se = np.std(fold_devs, axis=0, ddof=1) / np.sqrt(cv_folds)
    k_min = int(np.argmin(cv_mean))
    # 1-SE rule: largest λ (smallest index, since grid is descending) within 1 SE.
    thresh = cv_mean[k_min] + cv_se[k_min]
    cand = np.where(cv_mean[: k_min + 1] <= thresh)[0]
    k_1se = int(cand.min()) if cand.size else k_min

    # Refit on the full data at λ_min for the headline reporting.
    t_refit0 = time.perf_counter()
    full_coef, full_int = _fit_path(problem, estimator, grid)
    t_refit = time.perf_counter() - t_refit0
    final_support = int(np.count_nonzero(full_coef[k_min]))

    return {
        "dataset":  dataset_id,
        "estimator": estimator,
        "family":   problem.family,
        "n":        int(problem.x.shape[0]),
        "p":        int(problem.x.shape[1]),
        "cv_folds": cv_folds,
        "n_lambdas": n_lambdas,
        "seed":     seed,
        "lambda_min_idx":  k_min,
        "lambda_1se_idx":  k_1se,
        "cv_deviance_mean": cv_mean.tolist(),
        "cv_deviance_se":   cv_se.tolist(),
        "fold_deviance_at_lambda_min": [float(d[k_min]) for d in fold_devs],
        "fold_support_size":           fold_supp,
        "final_support_size_lambda_min": final_support,
        "cv_seconds": t_cv,
        "refit_seconds": t_refit,
        "source": (problem.meta or {}).get("source", "unknown"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset", default=None,
                    help="Restrict to one dataset id.")
    ap.add_argument("--estimator", default=None,
                    help="Restrict to one estimator (within the chosen dataset).")
    ap.add_argument("--use-synthetic", action="store_true",
                    help="Use offline-safe synthetic stand-ins for missing real data.")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-lambdas", type=int, default=50)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    env = capture_env.write(a.out_dir / "env.json")

    n_ok, n_fail = 0, 0
    for ds, family, estimators in CASE_STUDIES:
        if a.dataset and ds != a.dataset:
            continue
        for est in estimators:
            if a.estimator and est != a.estimator:
                continue
            out_path = a.out_dir / f"{ds}__{est}.jsonl"
            if out_path.exists():
                print(f"  skip   {ds}/{est} (cached)")
                continue
            try:
                t0 = time.perf_counter()
                row = run_one(ds, est, cv_folds=a.cv_folds, seed=a.seed,
                              use_synthetic=a.use_synthetic,
                              n_lambdas=a.n_lambdas)
                row["host_id"] = env["host_id"]
                row["git_rev"] = env["git_rev"]
                row["used_synthetic"] = a.use_synthetic
                row["wall_seconds"] = time.perf_counter() - t0
                out_path.write_text(json.dumps(row) + "\n")
                n_ok += 1
                print(f"  ok     {ds}/{est:14s} cv={row['cv_seconds']:.1f}s "
                      f"refit={row['refit_seconds']:.1f}s "
                      f"|Ŝ|={row['final_support_size_lambda_min']} "
                      f"min-dev={min(row['cv_deviance_mean']):.3f}")
            except Exception as e:
                n_fail += 1
                err_path = a.out_dir / f"{ds}__{est}.error.json"
                err_path.write_text(json.dumps({
                    "dataset": ds, "estimator": est,
                    "error": f"{type(e).__name__}: {e}",
                }))
                print(f"  FAIL   {ds}/{est}: {e}")
    print(f"\nrealdata: {n_ok} ok, {n_fail} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
