"""glmnet (R) adapter — invoked via Rscript + feather IPC.

Supports Lasso/EN for Gaussian, Logistic, Poisson. Cox needs a
(time, status) y which is wired separately once the Cox simulator
emits it (Phase C).
"""
from __future__ import annotations

import shutil

import numpy as np

from benches.problems import Problem
from benches.v2.runners import PenaltyName, RunResult
from benches.v2.runners._r_io import run_r


name = "glmnet"


def is_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    # Cheap probe: ask R whether the packages we need are installed.
    import subprocess
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             "q(status = as.integer(!all(c('arrow','glmnet') %in% installed.packages()[,1])))"],
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
    **_: object,
) -> RunResult:
    # Cox in glmnet wants y = Surv(time, status). The Python side passes
    # the times in `y` and the 0/1 event indicator in problem.meta["event"];
    # _r_io packages it as a separate `status.feather` payload that the R
    # runner reads back when family == "cox".
    status = None
    if problem.family == "cox":
        meta_event = problem.meta.get("event") if problem.meta else None
        if meta_event is None:
            raise ValueError(
                "glmnet runner: Cox cell needs problem.meta['event'] "
                "(the simulator must emit a 0/1 status vector alongside y)"
            )
        status = np.asarray(meta_event, dtype=np.int32)
    res = run_r(
        package="glmnet",
        penalty=penalty,
        family=problem.family,
        x=problem.x, y=problem.y,
        lambdas=np.asarray(lambda_grid),
        tol=tol,
        status=status,
    )
    coef_path = np.asarray(res["coef_path"])  # (n_lambdas, p)
    final_active = int(np.count_nonzero(coef_path[-1]))
    return RunResult(
        package=name,
        version=str(res.get("version", "unknown")),
        fit_time_s=float(res["fit_time_s"]),
        n_iter=None,
        final_obj=None,
        active_set_size=final_active,
        coef_path=coef_path,
        extra={"via": "Rscript+feather"},
    )
