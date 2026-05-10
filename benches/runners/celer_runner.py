"""celer runner — fast convex solver via dual extrapolation + screening."""

from __future__ import annotations

import time

import numpy as np

from benches.problems import Problem
from benches.runners import PenaltyName, RunResult


name = "celer"


def is_available() -> bool:
    try:
        import celer  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    import celer

    return getattr(celer, "__version__", "unknown")


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    **_: object,
) -> RunResult:
    from celer import Lasso, ElasticNet  # type: ignore
    from celer import LogisticRegression as CelerLogistic  # type: ignore

    alphas = np.asarray(lambda_grid)

    if problem.family == "gaussian" and penalty == "lasso":
        t0 = time.perf_counter()
        coefs = []
        for lam in alphas:
            est = Lasso(alpha=lam, tol=tol, fit_intercept=True)
            est.fit(problem.x, problem.y)
            coefs.append(est.coef_)
        elapsed = time.perf_counter() - t0
    elif problem.family == "gaussian" and penalty == "elastic_net":
        t0 = time.perf_counter()
        coefs = []
        for lam in alphas:
            est = ElasticNet(alpha=lam, l1_ratio=0.5, tol=tol, fit_intercept=True)
            est.fit(problem.x, problem.y)
            coefs.append(est.coef_)
        elapsed = time.perf_counter() - t0
    elif problem.family == "logistic" and penalty == "lasso":
        n = problem.x.shape[0]
        t0 = time.perf_counter()
        coefs = []
        for lam in alphas:
            est = CelerLogistic(C=1.0 / (lam * n), tol=tol, fit_intercept=True)
            est.fit(problem.x, problem.y)
            coefs.append(est.coef_.ravel())
        elapsed = time.perf_counter() - t0
    else:
        raise NotImplementedError(f"celer: ({problem.family}, {penalty}) not supported")

    coef_path = np.stack(coefs, axis=0)
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
