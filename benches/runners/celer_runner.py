"""celer runner — fast convex solver via dual extrapolation + screening.

Uses `celer.homotopy.celer_path` so warm-starts along the λ-grid are
honest (the dropin `celer.Lasso` fit per-λ throws away the previous
solve). This is the same path API celer's own benchmarks use.
"""

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
    from celer.homotopy import celer_path

    alphas = np.asarray(lambda_grid)

    if problem.family == "gaussian" and penalty in ("lasso", "elastic_net"):
        l1_ratio = 1.0 if penalty == "lasso" else 0.5
        t0 = time.perf_counter()
        _alphas, coefs, *_rest = celer_path(
            problem.x,
            problem.y,
            "lasso",
            alphas=alphas,
            tol=tol,
            l1_ratio=l1_ratio,
        )
        elapsed = time.perf_counter() - t0
    elif problem.family == "logistic" and penalty == "lasso":
        # celer_path's logreg loss is the un-normalised sum (no `/n`),
        # so to match skein's `lasso_path`-style λ-scaling we'd need
        # to convert. For now, bench logistic separately.
        raise NotImplementedError("celer runner: logistic not yet wired")
    else:
        raise NotImplementedError(f"celer runner: ({problem.family}, {penalty}) not supported")

    coefs = np.asarray(coefs)  # shape (n_features, n_alphas)
    coef_path = coefs.T  # → (n_alphas, n_features)
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
