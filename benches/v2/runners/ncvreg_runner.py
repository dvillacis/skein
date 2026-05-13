"""ncvreg (R) adapter — invoked via Rscript + feather IPC.

Supports MCP/SCAD/Lasso for Gaussian, Logistic, Poisson.
"""
from __future__ import annotations

import shutil

import numpy as np

from benches.problems import Problem
from benches.v2.runners import PenaltyName, RunResult
from benches.v2.runners._r_io import run_r


name = "ncvreg"


def is_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    import subprocess
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             "q(status = as.integer(!all(c('arrow','ncvreg') %in% installed.packages()[,1])))"],
            check=False, capture_output=True, timeout=20,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    gamma: float = 3.0,
    **_: object,
) -> RunResult:
    res = run_r(
        package="ncvreg",
        penalty=penalty,
        family=problem.family,
        x=problem.x, y=problem.y,
        lambdas=np.asarray(lambda_grid),
        tol=tol,
        gamma=gamma if penalty in ("mcp", "scad") else None,
    )
    coef_path = np.asarray(res["coef_path"])
    final_active = int(np.count_nonzero(coef_path[-1]))
    return RunResult(
        package=name,
        version=str(res.get("version", "unknown")),
        fit_time_s=float(res["fit_time_s"]),
        n_iter=int(res["n_iter"]) if res.get("n_iter") is not None else None,
        final_obj=None,
        active_set_size=final_active,
        coef_path=coef_path,
        extra={"via": "Rscript+feather",
               "gamma": gamma if penalty in ("mcp", "scad") else None},
    )
