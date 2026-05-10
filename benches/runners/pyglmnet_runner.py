"""pyglmnet runner — Python-native GLM elastic-net (Gaussian/binomial/poisson)."""

from __future__ import annotations

import time

import numpy as np

from benches.problems import Problem
from benches.runners import PenaltyName, RunResult


name = "pyglmnet"


def is_available() -> bool:
    try:
        import pyglmnet  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    import pyglmnet

    return getattr(pyglmnet, "__version__", "unknown")


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    **_: object,
) -> RunResult:
    from pyglmnet import GLM  # type: ignore

    family_map = {"gaussian": "gaussian", "logistic": "binomial", "poisson": "poisson"}
    if problem.family not in family_map:
        raise NotImplementedError(f"pyglmnet: family={problem.family} not supported")
    if penalty not in ("lasso", "elastic_net"):
        raise NotImplementedError(f"pyglmnet: penalty={penalty} not supported")

    alpha = 1.0 if penalty == "lasso" else 0.5
    reg_lambda = np.asarray(lambda_grid)

    glm = GLM(
        distr=family_map[problem.family],
        alpha=alpha,
        reg_lambda=reg_lambda,
        tol=tol,
        fit_intercept=True,
    )
    t0 = time.perf_counter()
    glm.fit(problem.x, problem.y)
    elapsed = time.perf_counter() - t0

    coef_path = np.stack([fit_.beta_ for fit_ in glm.fit_], axis=0)
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
