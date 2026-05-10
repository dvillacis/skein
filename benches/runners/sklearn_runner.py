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


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    **_: object,
) -> RunResult:
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
