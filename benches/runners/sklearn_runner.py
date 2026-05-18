"""sklearn runner — convex baselines (Lasso, ElasticNet, LogisticRegression)."""

from __future__ import annotations

import time

import numpy as np

from benches.problems import Problem
from benches.runners import PenaltyName, RunResult


name = "sklearn"


def is_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    import sklearn

    return sklearn.__version__


def _fit_glasso(
    problem: Problem,
    lambda_grid: np.ndarray,
    tol: float,
) -> RunResult:
    """sklearn.covariance.GraphicalLasso has no native path solver, so
    loop one alpha at a time — matching the bench's skein glasso path."""
    import warnings

    from sklearn.covariance import GraphicalLasso
    from sklearn.exceptions import ConvergenceWarning

    n_edges: list[int] = []
    t0 = time.perf_counter()
    last_precision = None
    # Suppress ConvergenceWarning: sklearn's glasso emits one per non-
    # converged alpha (common at the dense tail of the grid), which
    # floods the bench log without changing wall-time measurement.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for alpha in np.asarray(lambda_grid, dtype=np.float64):
            # max_iter bumped from sklearn's default 100 to 200 to match
            # skein's outer-iter cap.
            est = GraphicalLasso(alpha=float(alpha), tol=tol, max_iter=200).fit(problem.x)
            precision = est.precision_
            last_precision = precision
            mask = np.abs(precision) > 1e-10
            np.fill_diagonal(mask, False)
            n_edges.append(int(mask.sum() // 2))
    elapsed = time.perf_counter() - t0
    final_density = (
        n_edges[-1] / (last_precision.shape[0] * (last_precision.shape[0] - 1) // 2)
        if last_precision is not None
        else 0.0
    )
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
            "kind": "glasso",
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
    **_: object,
) -> RunResult:
    # See note in skein_runner.fit: v2's glasso_l1 scenario sets
    # penalty="lasso" but the simulator is "glasso_truth", so dispatch on
    # the simulator label to avoid falling through to lasso_path.
    sim = (problem.meta or {}).get("simulator") if problem.meta else None
    if penalty == "glasso" or (sim == "glasso_truth" and penalty == "lasso"):
        return _fit_glasso(problem, lambda_grid, tol)
    from sklearn.linear_model import LogisticRegression, lasso_path, enet_path

    n = problem.x.shape[0]
    # sklearn parameterises by alpha = lambda (with the 1/(2n) loss); we pass the
    # same lambda_grid so the comparison is on a shared grid.
    alphas = np.asarray(lambda_grid)

    t0 = time.perf_counter()
    if problem.family == "gaussian" and penalty == "lasso":
        alphas_out, coef_path, _ = lasso_path(problem.x, problem.y, alphas=alphas, tol=tol)
    elif problem.family == "gaussian" and penalty == "elastic_net":
        alphas_out, coef_path, _ = enet_path(problem.x, problem.y, alphas=alphas, l1_ratio=0.5, tol=tol)
    elif problem.family == "logistic" and penalty == "lasso":
        # sklearn LogisticRegression doesn't expose a path solver; fit per-lambda.
        coefs = []
        for lam in alphas:
            clf = LogisticRegression(
                penalty="l1", solver="saga", C=1.0 / (lam * n), tol=tol, max_iter=10_000, fit_intercept=True
            )
            clf.fit(problem.x, problem.y)
            coefs.append(clf.coef_.ravel())
        coef_path = np.stack(coefs, axis=1)
    else:
        raise NotImplementedError(f"sklearn: ({problem.family}, {penalty}) not supported")
    elapsed = time.perf_counter() - t0

    coef_path = np.asarray(coef_path)  # shape (p, n_lambdas)
    final_active = int(np.count_nonzero(coef_path[:, -1]))
    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=None,
        final_obj=None,
        active_set_size=final_active,
        coef_path=coef_path.T,  # transpose to (n_lambdas, p) to match skein convention
    )
