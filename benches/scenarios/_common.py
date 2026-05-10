"""Shared utilities for bench scenarios.

Each scenario file specifies its problem generator + λ-grid choice +
penalty/family, then delegates the timing loop and the runner-vs-Rscript
dispatch here.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

R_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "r_runner.R"


def lambda_grid(
    x: np.ndarray,
    y: np.ndarray,
    n_lambdas: int,
    lambda_min_ratio: float = 1e-3,
) -> np.ndarray:
    """Standard glmnet recipe: geometric grid from `lambda_max` down to
    `lambda_min_ratio · lambda_max`.

    `lambda_min_ratio` controls how deep the path goes:
      - `1e-3`: deep path (saturated regime — most features active at λ_min)
      - `5e-2` to `1e-1`: sparse regime (path stops near support recovery)
    """
    n = x.shape[0]
    lambda_max = float(np.max(np.abs(x.T @ y)) / n)
    return np.geomspace(lambda_max, lambda_max * lambda_min_ratio, n_lambdas)


def has_rscript() -> bool:
    from shutil import which

    return which("Rscript") is not None


def run_r(
    *,
    package: str,
    penalty: str,
    family: str,
    problem,
    lambda_grid: np.ndarray,
    tol: float,
    extra: dict | None = None,
) -> dict:
    """Shell out to `r_runner.R` for an R-side fit.

    Returns the parsed JSON dict containing `fit_time_s`,
    `coef_path`, `active_set_size`, `version`, etc. Raises
    `NotImplementedError` when `Rscript` is unavailable.
    """
    if not has_rscript():
        raise NotImplementedError("Rscript not on PATH")
    payload: dict = {
        "package": package,
        "penalty": penalty,
        "family": family,
        "X": problem.x.tolist(),
        "y": problem.y.tolist(),
        "lambdas": lambda_grid.tolist(),
        "tol": tol,
    }
    if extra:
        payload.update(extra)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path = tmp / "in.json"
        out_path = tmp / "out.json"
        in_path.write_text(json.dumps(payload))
        subprocess.run(
            ["Rscript", str(R_RUNNER), str(in_path), str(out_path)],
            check=True,
            capture_output=True,
        )
        return json.loads(out_path.read_text())


def time_python_runner(
    *,
    runner,
    problem,
    penalty: str,
    lambda_grid: np.ndarray,
    tol: float,
    trials: int,
    **fit_kwargs,
) -> tuple[list[float], object]:
    """1 warm-up + `trials` timed calls. Returns `(per_trial_s, last_result)`."""
    runner.fit(problem, penalty=penalty, lambda_grid=lambda_grid, tol=tol, **fit_kwargs)
    per_trial: list[float] = []
    last = None
    for _ in range(trials):
        result = runner.fit(
            problem, penalty=penalty, lambda_grid=lambda_grid, tol=tol, **fit_kwargs
        )
        per_trial.append(float(result.fit_time_s))
        last = result
    return per_trial, last


def time_r_runner(
    *,
    package: str,
    penalty: str,
    family: str,
    problem,
    lambda_grid: np.ndarray,
    tol: float,
    trials: int,
    extra: dict | None = None,
) -> tuple[list[float], dict]:
    """Same shape as `time_python_runner` but for the R subprocess path.

    Each warm-up / timed run pays the Rscript startup cost (~200 ms);
    that's representative of how a Python user would call glmnet via
    `reticulate` or a one-shot subprocess, so we keep it in the timing.
    """
    run_r(
        package=package,
        penalty=penalty,
        family=family,
        problem=problem,
        lambda_grid=lambda_grid,
        tol=tol,
        extra=extra,
    )
    per_trial: list[float] = []
    last = None
    for _ in range(trials):
        result = run_r(
            package=package,
            penalty=penalty,
            family=family,
            problem=problem,
            lambda_grid=lambda_grid,
            tol=tol,
            extra=extra,
        )
        per_trial.append(float(result["fit_time_s"]))
        last = result
    return per_trial, last


def summarize(
    *,
    package: str,
    version: str,
    problem,
    n_lambdas: int,
    per_trial: list[float],
    active_set_size: int,
    size: str,
    extra: dict,
    n_iter: int | None = None,
    final_obj: float | None = None,
    n_groups: int | None = None,
) -> dict:
    """Dict shape committed to `benches/results/<scenario>.json`."""
    return {
        "package": package,
        "version": version,
        "n": int(problem.x.shape[0]),
        "p": int(problem.x.shape[1]),
        "n_groups": n_groups,
        "lambda_grid_len": int(n_lambdas),
        "fit_time_s": float(statistics.median(per_trial)),
        "fit_time_min_s": float(min(per_trial)),
        "fit_time_max_s": float(max(per_trial)),
        "trials": per_trial,
        "n_iter": n_iter,
        "final_obj": final_obj,
        "active_set_size": active_set_size,
        "size": size,
        "extra": extra,
    }
