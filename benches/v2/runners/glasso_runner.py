"""glasso (R) adapter — invoked via Rscript + feather IPC.

The R `glasso` package operates on a sample covariance matrix S, not the
data matrix X, so we compute S = X^T X / n before handing off. The R
side returns a (n_lambdas × p^2) path of flattened precision matrices;
we reshape and report edge counts to match the sklearn/skein graphical
runners (which set coef_path=None and stash edge stats in `extra`).
"""
from __future__ import annotations

import shutil

import numpy as np

from benches.problems import Problem
from benches.v2.runners import PenaltyName, RunResult
from benches.v2.runners._r_io import run_r


name = "glasso"


def is_available() -> bool:
    if not shutil.which("Rscript"):
        return False
    import subprocess
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             "q(status = as.integer(!all(c('arrow','glasso') %in% installed.packages()[,1])))"],
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
    sim = (problem.meta or {}).get("simulator")
    if sim != "glasso_truth":
        raise NotImplementedError(
            f"glasso runner: only graphical scenarios (simulator='glasso_truth') "
            f"are supported (got simulator={sim!r}, family={problem.family!r})"
        )
    x = np.ascontiguousarray(problem.x, dtype=np.float64)
    n = x.shape[0]
    s = (x.T @ x) / n
    # r_runner.R's fit_glasso checks `family == "gaussian_inv_cov"`, so pass
    # that literal regardless of how the simulator labels Problem.family.
    res = run_r(
        package="glasso",
        penalty=penalty,
        family="gaussian_inv_cov",
        x=s, y=np.zeros(s.shape[0], dtype=np.float64),
        lambdas=np.asarray(lambda_grid),
        tol=tol,
    )
    precision_path = np.asarray(res["coef_path"])  # (n_lambdas, p*p)
    p = s.shape[0]
    last = precision_path[-1].reshape(p, p)
    mask = np.abs(last) > 1e-10
    np.fill_diagonal(mask, False)
    n_edges_final = int(mask.sum() // 2)

    n_edges_per_lambda: list[int] = []
    for row in precision_path:
        m = np.abs(row.reshape(p, p)) > 1e-10
        np.fill_diagonal(m, False)
        n_edges_per_lambda.append(int(m.sum() // 2))
    final_density = n_edges_final / (p * (p - 1) // 2) if p > 1 else 0.0

    return RunResult(
        package=name,
        version=str(res.get("version", "unknown")),
        fit_time_s=float(res["fit_time_s"]),
        n_iter=None,
        final_obj=None,
        active_set_size=n_edges_final,
        coef_path=None,
        extra={
            "via": "Rscript+feather",
            "kind": "glasso",
            "n_edges_per_lambda": n_edges_per_lambda,
            "final_density": final_density,
        },
    )
