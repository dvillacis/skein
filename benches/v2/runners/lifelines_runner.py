"""lifelines runner — Cox PH with L1 penalty as a pure-Python reference.

lifelines' `CoxPHFitter(penalizer=lam, l1_ratio=1.0)` fits a single λ
at a time (no path solver), so we iterate over the grid. This is
intentionally slow compared to glmnet — it's an *independent*
implementation, useful for agreement checks on Cox cells.
"""
from __future__ import annotations

import time

import numpy as np

from benches.problems import Problem
from benches.v2.runners import PenaltyName, RunResult


name = "lifelines"


def is_available() -> bool:
    try:
        import lifelines  # noqa: F401
        import pandas as _pd  # noqa: F401
    except ImportError:
        return False
    return True


def _version() -> str:
    import lifelines
    return getattr(lifelines, "__version__", "unknown")


def fit(
    problem: Problem,
    *,
    penalty: PenaltyName,
    lambda_grid: np.ndarray,
    tol: float,
    **_: object,
) -> RunResult:
    if problem.family != "cox":
        raise NotImplementedError(f"lifelines: family={problem.family} not supported")
    if penalty != "lasso":
        raise NotImplementedError(f"lifelines: penalty={penalty} not supported "
                                  "(only L1 via penalizer+l1_ratio=1.0)")

    import pandas as pd
    from lifelines import CoxPHFitter

    # Convention from benches.v2.simulators.cox_truth: problem.y is
    # the time, problem.meta carries 'event' (0/1) status.
    time_arr = np.asarray(problem.y, dtype=float)
    status = np.asarray(problem.meta.get("event"), dtype=int)
    if status.size != time_arr.size:
        raise ValueError("lifelines runner: missing 'event' status in problem.meta")

    p = problem.x.shape[1]
    feature_cols = [f"x{j}" for j in range(p)]
    df = pd.DataFrame(problem.x, columns=feature_cols)
    df["T"] = time_arr
    df["E"] = status

    coefs = []
    t0 = time.perf_counter()
    for lam in np.asarray(lambda_grid):
        cph = CoxPHFitter(penalizer=float(lam), l1_ratio=1.0)
        cph.fit(df, duration_col="T", event_col="E",
                show_progress=False, fit_options={"step_size": 1.0})
        coefs.append(cph.params_.reindex(feature_cols).fillna(0.0).to_numpy())
    elapsed = time.perf_counter() - t0

    coef_path = np.vstack(coefs)
    return RunResult(
        package=name,
        version=_version(),
        fit_time_s=elapsed,
        n_iter=None,
        final_obj=None,
        active_set_size=int(np.count_nonzero(coef_path[-1])),
        coef_path=coef_path,
        extra={"via": "lifelines.CoxPHFitter"},
    )
