"""skglm runner — closest Python competitor (parallel CD, partial group support).

Uses skglm's `Lasso.path` / `ElasticNet.path` internally so warm-starts
along the λ-grid are honest. Calling `Lasso(alpha=λ).fit(X, y)` per λ
(an earlier, easier implementation) handicaps skglm by throwing the
warm β away each time; this runner avoids that and reflects skglm's
actual path-solving capability.
"""

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
    if problem.family != "gaussian":
        raise NotImplementedError(f"skglm runner: family={problem.family} not yet wired")

    alphas = np.asarray(lambda_grid)

    if penalty == "lasso":
        from skglm.estimators import Lasso

        est = Lasso(tol=tol, fit_intercept=True, warm_start=True)
        t0 = time.perf_counter()
        # `Lasso.path` warm-starts down `alphas` internally — the apples-to-
        # apples comparison vs sklearn's `lasso_path` and our solver.
        _alphas_out, coefs, *_rest = est.path(problem.x, problem.y, alphas=alphas)
        elapsed = time.perf_counter() - t0
    elif penalty == "elastic_net":
        from skglm.estimators import ElasticNet

        est = ElasticNet(l1_ratio=0.5, tol=tol, fit_intercept=True, warm_start=True)
        t0 = time.perf_counter()
        _alphas_out, coefs, *_rest = est.path(problem.x, problem.y, alphas=alphas)
        elapsed = time.perf_counter() - t0
    elif penalty == "mcp":
        from skglm.estimators import MCPRegression

        est = MCPRegression(gamma=3.0, tol=tol, fit_intercept=True, warm_start=True)
        t0 = time.perf_counter()
        _alphas_out, coefs, *_rest = est.path(problem.x, problem.y, alphas=alphas)
        elapsed = time.perf_counter() - t0
    else:
        raise NotImplementedError(f"skglm runner: penalty={penalty} not yet wired")

    coefs = np.asarray(coefs)  # shape (n_features, n_alphas)
    coef_path = coefs.T  # → (n_alphas, n_features) to match skein convention
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
