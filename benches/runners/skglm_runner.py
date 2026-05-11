"""skglm runner — closest Python competitor (parallel CD, partial group support).

Uses skglm's `Lasso.path` / `ElasticNet.path` / `MCPRegression.path`
internally so warm-starts along the λ-grid are honest. Calling
`Lasso(alpha=λ).fit(X, y)` per λ (an earlier, easier implementation)
handicaps skglm by throwing the warm β away each time; this runner
avoids that and reflects skglm's actual path-solving capability.

SCAD is the awkward case: skglm ships a `SCAD` penalty class but no
estimator wrapper or `.path()` method. We assemble one manually via
`GeneralizedLinearEstimator(SCAD, Quadratic, AndersonCD)` and iterate
α-by-α with `warm_start=True` on the solver, which preserves the
prior `coef_` between fits — the same shape as the built-in
`.path()` methods.
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
    gamma: float = 3.0,
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

        est = MCPRegression(gamma=gamma, tol=tol, fit_intercept=True, warm_start=True)
        t0 = time.perf_counter()
        _alphas_out, coefs, *_rest = est.path(problem.x, problem.y, alphas=alphas)
        elapsed = time.perf_counter() - t0
    elif penalty == "scad":
        from skglm import GeneralizedLinearEstimator
        from skglm.datafits import Quadratic
        from skglm.penalties import SCAD
        from skglm.solvers import AndersonCD

        scad = SCAD(alpha=float(alphas[0]), gamma=gamma)
        est = GeneralizedLinearEstimator(
            datafit=Quadratic(),
            penalty=scad,
            # AndersonCD is skglm's default for `MCPRegression` too; warm_start
            # preserves est.coef_ across the α-loop below.
            solver=AndersonCD(tol=tol, fit_intercept=True, warm_start=True),
        )
        p = problem.x.shape[1]
        coefs_T = np.empty((len(alphas), p))
        intercepts = np.empty(len(alphas))
        t0 = time.perf_counter()
        for i, lam in enumerate(alphas):
            scad.alpha = float(lam)
            est.fit(problem.x, problem.y)
            coefs_T[i] = est.coef_
            intercepts[i] = est.intercept_
        elapsed = time.perf_counter() - t0
        # Match the (p, n_alphas) layout the lasso/mcp branches produce so the
        # downstream intercept-strip + transpose work uniformly.
        coefs = np.vstack([coefs_T.T, intercepts[np.newaxis, :]])
    else:
        raise NotImplementedError(f"skglm runner: penalty={penalty} not yet wired")

    coefs = np.asarray(coefs)  # (p, n_alphas) without intercept, (p+1, n_alphas) with it.
    p = problem.x.shape[1]
    intercept_path = None
    if coefs.shape[0] == p + 1:
        # skglm appends the intercept as the last row when fit_intercept=True.
        intercept_path = coefs[-1, :].copy()
        coefs = coefs[:p, :]
    coef_path = coefs.T  # → (n_alphas, p) to match skein convention
    final_active = int(np.count_nonzero(coef_path[-1]))
    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=None,
        final_obj=None,
        active_set_size=final_active,
        coef_path=coef_path,
        intercept_path=intercept_path,
    )
