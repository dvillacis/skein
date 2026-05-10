"""skein runner — fits via the public sklearn-compatible estimators."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from benches.problems import Problem
from benches.runners import PenaltyName, RunResult


name = "skein"


def is_available() -> bool:
    try:
        import skein_glm  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    try:
        import skein_glm

        return getattr(skein_glm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _build_estimator(
    problem: Problem,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    screening: Literal["off", "strong", "gap_safe"],
):
    # Lazy import so import-time cost doesn't pollute timings.
    from skein_glm import estimators as e

    common = dict(lambdas=np.asarray(lambda_grid), tol=tol, screening=screening)

    if problem.family == "gaussian":
        if penalty == "lasso":
            # Elastic net with alpha=1 is pure L1 (lasso).
            return e.ElasticNetPathRegressor(alpha=1.0, **common)
        if penalty == "elastic_net":
            return e.ElasticNetPathRegressor(alpha=0.5, **common)
        if penalty == "mcp":
            return e.MCPPathRegressor(**common)
        if penalty == "scad":
            return e.SCADPathRegressor(**common)
        if penalty == "group_lasso":
            return e.GroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "group_mcp":
            return e.GroupMCPPathRegressor(groups=problem.groups, **common)
        if penalty == "group_scad":
            return e.GroupSCADPathRegressor(groups=problem.groups, **common)
    raise NotImplementedError(f"skein runner: ({problem.family}, {penalty}) not yet wired")


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    screening: Literal["off", "strong", "gap_safe"] = "strong",
    **_: object,
) -> RunResult:
    est = _build_estimator(problem, penalty, lambda_grid, tol, screening)
    t0 = time.perf_counter()
    est.fit(problem.x, problem.y)
    elapsed = time.perf_counter() - t0

    # Path estimators expose `coefs_` (n_lambdas, p) and `intercepts_` (n_lambdas,).
    coef_path = np.asarray(est.coefs_)
    final_active = int(np.count_nonzero(coef_path[-1]))
    info = getattr(est, "info_", {}) or {}
    intercept_path = np.asarray(getattr(est, "intercepts_", np.zeros(coef_path.shape[0])))

    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=info.get("total_iters"),
        final_obj=info.get("final_obj"),
        active_set_size=final_active,
        coef_path=coef_path,
        intercept_path=intercept_path,
        extra={"screening": screening, "info_keys": sorted(info.keys()) if info else []},
    )
