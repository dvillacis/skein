"""F10 — Threaded-CV vs serial-CV wall-clock.

Targets the M5.x-c paper claim: threaded CV via PyO3 GIL release
delivers 2.3-2.5× speedup over serial CV.

For each (n_jobs, n_folds) pair, fit MCPPathCV on the same synthetic
problem; record wall-clock. Plot speedup vs n_jobs at each n_folds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _run_one(*, n: int, p: int, seed: int, n_folds: int, n_jobs: int,
             n_lambdas: int, tol: float) -> dict:
    from benches.v2.simulators import linear_truth
    from skein_glm.cv import MCPPathCV

    problem = linear_truth.make(n=n, p=p, seed=seed, snr=3.0)

    # 1 warmup + 3 timed trials.
    cv = MCPPathCV(cv=n_folds, n_jobs=n_jobs,
                   n_lambdas=n_lambdas, lambda_min_ratio=1e-2,
                   tol=tol, random_state=seed)
    cv.fit(problem.x, problem.y)
    per_trial = []
    for _ in range(3):
        cv = MCPPathCV(cv=n_folds, n_jobs=n_jobs,
                       n_lambdas=n_lambdas, lambda_min_ratio=1e-2,
                       tol=tol, random_state=seed)
        t0 = time.perf_counter()
        cv.fit(problem.x, problem.y)
        per_trial.append(time.perf_counter() - t0)
    return {
        "n": n, "p": p, "seed": seed, "n_folds": n_folds, "n_jobs": n_jobs,
        "n_lambdas": n_lambdas,
        "fit_time_s_median": float(np.median(per_trial)),
        "fit_time_s_min":    float(min(per_trial)),
        "fit_time_s_max":    float(max(per_trial)),
        "trials": per_trial,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n-lambdas", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    out = a.out_dir / "cv_parallel.jsonl"
    with out.open("w") as f:
        for n_folds in a.folds:
            for n_jobs in a.jobs:
                row = _run_one(
                    n=a.n, p=a.p, seed=a.seed,
                    n_folds=n_folds, n_jobs=n_jobs,
                    n_lambdas=a.n_lambdas, tol=a.tol,
                )
                f.write(json.dumps(row) + "\n")
                print(f"  folds={n_folds:2d} jobs={n_jobs}: "
                      f"{row['fit_time_s_median']:.3f}s "
                      f"(min {row['fit_time_s_min']:.3f})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
