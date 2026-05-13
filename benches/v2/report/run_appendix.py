"""Run the appendix matrix: every public skein estimator at one cell.

This is the populator for T1's count column with live data. Each
appendix cell is at the small size, deep regime, seed 0, skein only
(no comparator), tol relaxed slightly so GLM cells finish quickly
even in maturin dev profile.

Usage:
  python -m benches.v2.report.run_appendix --out-dir benches/v2/results/cells

Each cell still writes a JSONL row that aggregates into a per-scenario
JSON the same way the headline cells do.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from benches.v2.report import capture_env


def _make_problem(datafit: str, n: int, p: int, seed: int,
                  group_size: int = 5):
    """Pick a sensible truth-aware simulator for the family."""
    if datafit == "gaussian":
        from benches.v2.simulators import linear_truth, group_truth
        if group_size:
            return group_truth.make(n=n, p=p, seed=seed, group_size=group_size)
        return linear_truth.make(n=n, p=p, seed=seed)
    if datafit == "logistic":
        from benches.v2.simulators import logistic_truth
        return logistic_truth.make(n=n, p=p, seed=seed)
    if datafit == "poisson":
        from benches.v2.simulators import poisson_truth
        return poisson_truth.make(n=n, p=p, seed=seed)
    if datafit == "cox":
        from benches.v2.simulators import cox_truth
        return cox_truth.make(n=n, p=p, seed=seed)
    raise ValueError(f"unknown datafit: {datafit!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--size", default="small")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--n-lambdas", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="Restrict to these appendix-cell ids.")
    a = ap.parse_args()

    from benches.v2.scenarios.appendix import APPENDIX_CELLS
    from benches.v2.runners import skein_runner

    a.out_dir.mkdir(parents=True, exist_ok=True)
    env = capture_env.capture()
    n_ok, n_fail, n_skip = 0, 0, 0
    for cell_id, datafit, penalty, group_size in APPENDIX_CELLS:
        if a.ids and cell_id not in a.ids:
            continue
        out_jsonl = a.out_dir / f"{cell_id}__{a.size}__deep__seed0__skein.jsonl"
        env_json = out_jsonl.with_suffix(".env.json")
        if out_jsonl.exists():
            n_skip += 1
            continue
        try:
            problem = _make_problem(datafit, a.n, a.p, seed=0,
                                    group_size=group_size)
            grid = np.geomspace(
                float(np.max(np.abs(problem.x.T @ (problem.y - problem.y.mean()))) / problem.x.shape[0]),
                float(np.max(np.abs(problem.x.T @ (problem.y - problem.y.mean()))) / problem.x.shape[0]) * 1e-3,
                a.n_lambdas,
            )
            t0 = time.perf_counter()
            res = skein_runner.fit(problem, penalty=penalty,
                                   lambda_grid=grid, tol=a.tol)
            elapsed = time.perf_counter() - t0
            row = {
                "scenario": cell_id, "size": a.size, "regime": "deep",
                "seed": 0, "package": "skein", "status": "ok",
                "datafit": datafit, "penalty": penalty,
                "n": a.n, "p": a.p, "n_lambdas": a.n_lambdas,
                "tol": a.tol, "trials": [elapsed],
                "fit_time_s": elapsed,
                "fit_time_min_s": elapsed, "fit_time_max_s": elapsed,
                "version": res.version, "n_iter": res.n_iter,
                "final_obj": res.final_obj,
                "active_set_size": int(res.active_set_size),
                "host_id": env["host_id"], "git_rev": env["git_rev"],
                "extra": dict(res.extra), "appendix": True,
            }
            out_jsonl.write_text(json.dumps(row) + "\n")
            env_json.write_text(json.dumps(env, indent=2))
            n_ok += 1
            print(f"  ok    {cell_id} ({datafit}/{penalty}) in {elapsed:.2f}s")
        except Exception as e:
            n_fail += 1
            row = {
                "scenario": cell_id, "size": a.size, "regime": "deep",
                "seed": 0, "package": "skein", "status": "error",
                "datafit": datafit, "penalty": penalty,
                "error": f"{type(e).__name__}: {e}",
                "host_id": env["host_id"],
            }
            out_jsonl.write_text(json.dumps(row) + "\n")
            print(f"  FAIL  {cell_id}: {e}")
    print(f"\nappendix: {n_ok} ok, {n_fail} failed, {n_skip} skipped (already present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
