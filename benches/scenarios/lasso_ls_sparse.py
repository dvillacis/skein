"""Lasso on Gaussian LS, *sparse* regime.

  λ_min / λ_max = 5e-2

The path stops near support recovery rather than running into the
saturated tail. This is the regime lasso is actually designed for —
small support relative to `p`, λ chosen via cross-validation in a
range where most features stay zero. It's also where celer/skglm-style
priority working-set CD shines (the WS stays proportional to support
along the entire path), so it's the right scenario to verify that
M10.3's structural changes (col_axpy + F-order DenseMatrix + adaptive
inner tol via prox-gradient distance + KKT-priority WS construction)
deliver on the wallclock front.

Same problem generator, runner ABI, timing methodology, and result
schema as `lasso_ls`. Only `LAMBDA_MIN_RATIO` differs.
"""

from __future__ import annotations


from benches.problems import SIZES, gaussian_lasso

from . import _common

PENALTY = "lasso"
FAMILY = "gaussian"
LAMBDA_MIN_RATIO = 5e-2
R_PACKAGES = {"r": "glmnet"}


def run(*, runner, package: str, size: str, tol: float, n_lambdas: int, trials: int = 5) -> dict:
    problem = gaussian_lasso(SIZES[size])
    grid = _common.lambda_grid(problem.x, problem.y, n_lambdas, LAMBDA_MIN_RATIO)

    if package == "r":
        per_trial, last = _common.time_r_runner(
            package="glmnet",
            penalty=PENALTY,
            family=FAMILY,
            problem=problem,
            lambda_grid=grid,
            tol=tol,
            trials=trials,
        )
        return _common.summarize(
            package="glmnet",
            version=last.get("version", "unknown"),
            problem=problem,
            n_lambdas=n_lambdas,
            per_trial=per_trial,
            active_set_size=int(last["active_set_size"]),
            size=size,
            extra={"via": "Rscript", "lambda_min_ratio": LAMBDA_MIN_RATIO},
        )

    per_trial, last = _common.time_python_runner(
        runner=runner,
        problem=problem,
        penalty=PENALTY,
        lambda_grid=grid,
        tol=tol,
        trials=trials,
    )
    return _common.summarize(
        package=last.package,
        version=last.version,
        problem=problem,
        n_lambdas=n_lambdas,
        per_trial=per_trial,
        active_set_size=int(last.active_set_size),
        size=size,
        extra={
            **{k: v for k, v in last.extra.items() if k != "info"},
            "lambda_min_ratio": LAMBDA_MIN_RATIO,
        },
        n_iter=last.n_iter,
        final_obj=last.final_obj,
    )
