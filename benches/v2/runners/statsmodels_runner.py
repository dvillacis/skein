"""statsmodels runner — unpenalized IRLS baseline for Logistic and Poisson.

This is *not* a sparse-modeling competitor; it's a sanity-check
reference for "what does the unpenalized MLE look like on this
problem". Useful in the appendix to anchor the deviance scale —
every penalized fit's deviance should approach this at λ → 0.

Implementation: fit once at the smallest λ in the grid, expand the
result across the whole grid as a constant β (the unpenalized MLE
doesn't depend on λ). This is *not* a fair speed comparison; the
recorded time is one IRLS fit, repeated by the cell driver's trials
loop to surface IRLS variance.
"""
from __future__ import annotations

import time

import numpy as np

from benches.problems import Problem
from benches.v2.runners import PenaltyName, RunResult


name = "statsmodels"


def is_available() -> bool:
    try:
        import statsmodels.api  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    import statsmodels
    return getattr(statsmodels, "__version__", "unknown")


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    **_: object,
) -> RunResult:
    import statsmodels.api as sm

    if problem.family not in ("logistic", "poisson"):
        raise NotImplementedError(
            f"statsmodels runner: family={problem.family} not supported "
            "(use sklearn / skein for Gaussian; lifelines for Cox)"
        )
    family_obj = {"logistic": sm.families.Binomial(),
                  "poisson":  sm.families.Poisson()}[problem.family]

    X = sm.add_constant(problem.x, has_constant="add")
    t0 = time.perf_counter()
    res = sm.GLM(problem.y, X, family=family_obj).fit(maxiter=200, tol=tol)
    elapsed = time.perf_counter() - t0

    coef = np.asarray(res.params)[1:]   # drop intercept
    # Expand to (n_lambdas, p) so downstream agreement code works
    # uniformly — but tag it so figures can suppress it where useful.
    coef_path = np.tile(coef, (len(lambda_grid), 1))
    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=int(res.fit_history.get("iteration", 0))
                 if hasattr(res, "fit_history") else None,
        final_obj=float(res.deviance / 2.0),
        active_set_size=int(np.count_nonzero(coef)),
        coef_path=coef_path,
        extra={"unpenalized_mle": True,
               "deviance": float(res.deviance),
               "note": "λ-independent; replicated across grid for shape parity"},
    )
