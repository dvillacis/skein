"""Lasso on Gaussian LS: skein vs sklearn / skglm / celer / glmnet.

Each scenario module exposes `run(runner, package, size, tol, n_lambdas, trials)`
returning the dict that gets appended to results/<scenario>.json.

Timing methodology: 1 warm-up call (discarded) + `trials` measured calls
with the same problem and λ-grid. Report median, min, max, and the full
list of per-trial times. Median is the headline number — robust against
the occasional GC/JIT spike.

For R packages, the runner argument is None and we shell out to
benches/runners/r_runner.R via subprocess.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from benches.problems import SIZES, gaussian_lasso

PENALTY = "lasso"
FAMILY = "gaussian"
R_RUNNER = Path(__file__).resolve().parents[1] / "runners" / "r_runner.R"
R_PACKAGES = {"r": "glmnet"}  # which R package this scenario uses by default


def _lambda_grid(x: np.ndarray, y: np.ndarray, n_lambdas: int) -> np.ndarray:
    # Standard glmnet recipe: log-spaced from lambda_max down to ratio*lambda_max.
    n = x.shape[0]
    lambda_max = float(np.max(np.abs(x.T @ y)) / n)
    return np.geomspace(lambda_max, lambda_max * 1e-3, n_lambdas)


def _run_r(problem, lambda_grid: np.ndarray, tol: float, package: str = "glmnet") -> dict:
    if not _has_rscript():
        raise NotImplementedError("Rscript not on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path = tmp / "in.json"
        out_path = tmp / "out.json"
        in_path.write_text(
            json.dumps(
                {
                    "package": package,
                    "penalty": PENALTY,
                    "family": FAMILY,
                    "X": problem.x.tolist(),
                    "y": problem.y.tolist(),
                    "lambdas": lambda_grid.tolist(),
                    "tol": tol,
                }
            )
        )
        subprocess.run(
            ["Rscript", str(R_RUNNER), str(in_path), str(out_path)],
            check=True,
            capture_output=True,
        )
        return json.loads(out_path.read_text())


def _has_rscript() -> bool:
    from shutil import which

    return which("Rscript") is not None


def run(*, runner, package: str, size: str, tol: float, n_lambdas: int, trials: int = 5) -> dict:
    problem = gaussian_lasso(SIZES[size])
    lambda_grid = _lambda_grid(problem.x, problem.y, n_lambdas)

    if package == "r":
        # R subprocess overhead is high; one warm-up + `trials` measured runs
        # of the *same* fit (Rscript invocation) capture variance honestly.
        _run_r(problem, lambda_grid, tol, package="glmnet")  # warm-up
        per_trial = []
        last = None
        for _ in range(trials):
            r_result = _run_r(problem, lambda_grid, tol, package="glmnet")
            per_trial.append(float(r_result["fit_time_s"]))
            last = r_result
        return _summarize(
            package="glmnet",
            version=last.get("version", "unknown"),
            problem=problem,
            n_lambdas=n_lambdas,
            per_trial=per_trial,
            active_set_size=int(last["active_set_size"]),
            size=size,
            extra={"via": "Rscript"},
        )

    # Warm-up
    runner.fit(problem, penalty=PENALTY, lambda_grid=lambda_grid, tol=tol)
    per_trial = []
    last = None
    for _ in range(trials):
        result = runner.fit(problem, penalty=PENALTY, lambda_grid=lambda_grid, tol=tol)
        per_trial.append(float(result.fit_time_s))
        last = result
    return _summarize(
        package=last.package,
        version=last.version,
        problem=problem,
        n_lambdas=n_lambdas,
        per_trial=per_trial,
        active_set_size=int(last.active_set_size),
        size=size,
        extra={k: v for k, v in last.extra.items() if k != "info"},
        n_iter=last.n_iter,
        final_obj=last.final_obj,
    )


def _summarize(
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
) -> dict:
    return {
        "package": package,
        "version": version,
        "n": int(problem.x.shape[0]),
        "p": int(problem.x.shape[1]),
        "n_groups": None,
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
