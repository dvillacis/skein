"""skglm runner — closest Python competitor (parallel CD, partial group support)."""

from __future__ import annotations

import time

import numpy as np

from benches.problems import Problem
from benches.runners import PenaltyName, RunResult


name = "skglm"


def is_available() -> bool:
    try:
        import skglm  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    import skglm

    return getattr(skglm, "__version__", "unknown")


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    **_: object,
) -> RunResult:
    # skglm exposes Lasso/ElasticNet/MCP estimators and a generic
    # GeneralizedLinearEstimator. The runner builds the right
    # combination per (family, penalty) and times the path fit.
    from skglm.estimators import Lasso, ElasticNet, MCPRegression  # type: ignore

    if problem.family != "gaussian":
        raise NotImplementedError(f"skglm runner: family={problem.family} not yet wired")

    if penalty == "lasso":
        alphas = np.asarray(lambda_grid)
        t0 = time.perf_counter()
        coefs = []
        for lam in alphas:
            est = Lasso(alpha=lam, tol=tol, fit_intercept=True)
            est.fit(problem.x, problem.y)
            coefs.append(est.coef_)
        elapsed = time.perf_counter() - t0
        coef_path = np.stack(coefs, axis=0)
    elif penalty == "elastic_net":
        alphas = np.asarray(lambda_grid)
        t0 = time.perf_counter()
        coefs = []
        for lam in alphas:
            est = ElasticNet(alpha=lam, l1_ratio=0.5, tol=tol, fit_intercept=True)
            est.fit(problem.x, problem.y)
            coefs.append(est.coef_)
        elapsed = time.perf_counter() - t0
        coef_path = np.stack(coefs, axis=0)
    elif penalty == "mcp":
        alphas = np.asarray(lambda_grid)
        t0 = time.perf_counter()
        coefs = []
        for lam in alphas:
            est = MCPRegression(alpha=lam, gamma=3.0, tol=tol, fit_intercept=True)
            est.fit(problem.x, problem.y)
            coefs.append(est.coef_)
        elapsed = time.perf_counter() - t0
        coef_path = np.stack(coefs, axis=0)
    else:
        raise NotImplementedError(f"skglm: penalty={penalty} not yet wired")

    final_active = int(np.count_nonzero(coef_path[-1]))
    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=None,
        final_obj=None,
        active_set_size=final_active,
        coef_path=coef_path,
    )
