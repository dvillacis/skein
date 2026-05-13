"""grpreg (R) adapter — group-structured penalties via Rscript + feather IPC.

Supports grLasso / grMCP / grSCAD for Gaussian, Logistic, Poisson.
Groups come from `problem.groups` (length-p int array, 0-based).
"""
from __future__ import annotations

import shutil

import numpy as np

from benches.problems import Problem
from benches.v2.runners import PenaltyName, RunResult
from benches.v2.runners._r_io import run_r


name = "grpreg"


def is_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    import subprocess
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             "q(status = as.integer(!all(c('arrow','grpreg') %in% installed.packages()[,1])))"],
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
    if problem.groups is None:
        raise ValueError("grpreg runner: problem.groups is required for group penalties")
    res = run_r(
        package="grpreg",
        penalty=penalty,
        family=problem.family,
        x=problem.x, y=problem.y,
        lambdas=np.asarray(lambda_grid),
        tol=tol,
        groups=np.asarray(problem.groups, dtype=np.int64),
        gamma=gamma if penalty in ("group_mcp", "group_scad") else None,
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
               "gamma": gamma if penalty in ("group_mcp", "group_scad") else None},
    )
