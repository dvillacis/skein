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
    gamma: float,
):
    # Lazy import so import-time cost doesn't pollute timings.
    from skein_glm import estimators as e

    # Gaussian Path estimators accept `screening` (strong rule + gap-safe);
    # GLM Path estimators don't expose it (they go through prox-Newton).
    common = dict(lambdas=np.asarray(lambda_grid), tol=tol)
    if problem.family == "gaussian":
        common["screening"] = screening

    if problem.family == "gaussian":
        if penalty == "lasso":
            # Elastic net with alpha=1 is pure L1 (lasso).
            return e.ElasticNetPathRegressor(alpha=1.0, **common)
        if penalty == "elastic_net":
            return e.ElasticNetPathRegressor(alpha=0.5, **common)
        if penalty == "mcp":
            return e.MCPPathRegressor(gamma=gamma, **common)
        if penalty == "scad":
            # SCADPathRegressor uses `a` for the SCAD shape parameter; MCP
            # uses `gamma`. The bench scenario speaks γ uniformly, so we
            # translate here.
            return e.SCADPathRegressor(a=gamma, **common)
        if penalty == "group_lasso":
            return e.GroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "group_mcp":
            return e.GroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "group_scad":
            return e.GroupSCADPathRegressor(groups=problem.groups, a=gamma, **common)
        if penalty == "sparse_group_lasso":
            return e.SparseGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "sparse_group_mcp":
            return e.SparseGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_scad":
            return e.SparseGroupSCADPathRegressor(groups=problem.groups, a=gamma, **common)
    if problem.family == "logistic":
        if penalty == "lasso":
            return e.LogisticLassoPathRegressor(**common)
        if penalty == "elastic_net":
            return e.LogisticElasticNetPathRegressor(alpha=0.5, **common)
        if penalty == "mcp":
            return e.LogisticMCPPathRegressor(gamma=gamma, **common)
        if penalty == "scad":
            return e.LogisticSCADPathRegressor(a=gamma, **common)
        if penalty == "group_lasso":
            return e.LogisticGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "group_mcp":
            return e.LogisticGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_lasso":
            return e.LogisticSparseGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "sparse_group_mcp":
            return e.LogisticSparseGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_scad":
            return e.LogisticSparseGroupSCADPathRegressor(groups=problem.groups, a=gamma, **common)
    if problem.family == "poisson":
        if penalty == "lasso":
            return e.PoissonLassoPathRegressor(**common)
        if penalty == "elastic_net":
            return e.PoissonElasticNetPathRegressor(alpha=0.5, **common)
        if penalty == "mcp":
            return e.PoissonMCPPathRegressor(gamma=gamma, **common)
        if penalty == "scad":
            return e.PoissonSCADPathRegressor(a=gamma, **common)
        if penalty == "group_lasso":
            return e.PoissonGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "group_mcp":
            return e.PoissonGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_lasso":
            return e.PoissonSparseGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "sparse_group_mcp":
            return e.PoissonSparseGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_scad":
            return e.PoissonSparseGroupSCADPathRegressor(groups=problem.groups, a=gamma, **common)
    if problem.family == "cox":
        # Cox in skein uses (time, status) via fit(X, time, event=status);
        # the lasso path is exposed through CoxMCPPathRegressor(gamma=∞).
        if penalty == "lasso":
            return e.CoxMCPPathRegressor(gamma=1e6, **common)
        if penalty == "mcp":
            return e.CoxMCPPathRegressor(gamma=gamma, **common)
        if penalty == "scad":
            return e.CoxSCADPathRegressor(a=gamma, **common)
        if penalty == "group_lasso":
            return e.CoxGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "group_mcp":
            return e.CoxGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_lasso":
            return e.CoxSparseGroupLassoPathRegressor(groups=problem.groups, **common)
        if penalty == "sparse_group_mcp":
            return e.CoxSparseGroupMCPPathRegressor(groups=problem.groups, gamma=gamma, **common)
        if penalty == "sparse_group_scad":
            return e.CoxSparseGroupSCADPathRegressor(groups=problem.groups, a=gamma, **common)
    raise NotImplementedError(f"skein runner: ({problem.family}, {penalty}) not yet wired")


def _fit_glasso(
    problem: Problem,
    lambda_grid: np.ndarray,
    tol: float,
) -> RunResult:
    """Fit a path of single-alpha `GraphicalLasso` solves. The bench
    treats the whole grid as one cell so wall-clock is comparable to
    sklearn (which also has no native path solver)."""
    from skein_glm import GraphicalLasso

    return _fit_graphical_path(
        problem, lambda_grid, tol,
        builder=lambda alpha: GraphicalLasso(alpha=float(alpha), tol=tol),
        kind="glasso",
    )


def _fit_glasso_mcp(
    problem: Problem,
    lambda_grid: np.ndarray,
    tol: float,
    gamma: float,
) -> RunResult:
    """Fit a path of single-alpha `GraphicalMCP` solves. Same loop shape
    as `_fit_glasso` (no native path solver in the graphical family) but
    with the nonconvex MCP envelope on the off-diagonal entries of Σ⁻¹.
    """
    from skein_glm import GraphicalMCP

    return _fit_graphical_path(
        problem, lambda_grid, tol,
        builder=lambda alpha: GraphicalMCP(alpha=float(alpha), gamma=gamma, tol=tol),
        kind="glasso_mcp",
    )


def _fit_graphical_path(
    problem: Problem,
    lambda_grid: np.ndarray,
    tol: float,
    *,
    builder,
    kind: str,
) -> RunResult:
    n_edges: list[int] = []
    t0 = time.perf_counter()
    last_precision = None
    for alpha in np.asarray(lambda_grid, dtype=np.float64):
        est = builder(alpha).fit(problem.x)
        precision = est.precision_
        last_precision = precision
        # Count off-diagonal edges (undirected, so /2).
        mask = np.abs(precision) > 1e-10
        np.fill_diagonal(mask, False)
        n_edges.append(int(mask.sum() // 2))
    elapsed = time.perf_counter() - t0
    final_density = (
        n_edges[-1] / (last_precision.shape[0] * (last_precision.shape[0] - 1) // 2)
        if last_precision is not None
        else 0.0
    )
    _ = tol  # tol is forwarded via the builder closure
    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=None,
        final_obj=None,
        active_set_size=n_edges[-1] if n_edges else 0,
        coef_path=None,
        intercept_path=None,
        extra={
            "kind": kind,
            "n_edges_per_lambda": n_edges,
            "final_density": final_density,
        },
    )


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    screening: Literal["off", "strong", "gap_safe"] = "strong",
    gamma: float = 3.0,
    **_: object,
) -> RunResult:
    # Graphical scenarios route on the simulator label rather than the
    # penalty name: v2's `glasso_l1` scenario SPECs penalty="lasso" (the
    # graphical L1 envelope), so dispatching purely on `penalty == "glasso"`
    # silently runs `lasso_path` on the n×p data matrix instead of fitting
    # the precision matrix. See ROADMAP §M14g.1.
    sim = (problem.meta or {}).get("simulator") if problem.meta else None
    if penalty == "glasso" or (sim == "glasso_truth" and penalty == "lasso"):
        return _fit_glasso(problem, lambda_grid, tol)
    if penalty == "glasso_mcp" or (sim == "glasso_truth" and penalty == "mcp"):
        return _fit_glasso_mcp(problem, lambda_grid, tol, gamma)
    est = _build_estimator(problem, penalty, lambda_grid, tol, screening, gamma)
    t0 = time.perf_counter()
    if problem.family == "cox":
        # Cox's fit signature is fit(x, time, event). Event status is
        # stashed in problem.meta by the cox_truth simulator.
        event = problem.meta.get("event") if problem.meta else None
        if event is None:
            raise ValueError("skein runner: Cox cell needs problem.meta['event']")
        est.fit(problem.x, problem.y, np.asarray(event, dtype=np.int64))
    else:
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
        extra={
            "screening": screening,
            "gamma": gamma if penalty in ("mcp", "scad", "group_mcp", "group_scad") else None,
            "info_keys": sorted(info.keys()) if info else [],
        },
    )
