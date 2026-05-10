"""Lasso on Gaussian LS, *deep-path* regime.

  λ_min / λ_max = 1e-3

This pushes the path deep enough that the active set saturates near
λ_min — typical of "I want the entire regularisation path including
the over-fit tail" usage. For the active-set-stays-small regime, see
`lasso_ls_sparse`.

Each scenario module exposes `run(runner, package, size, tol, n_lambdas,
trials)` returning the dict that gets appended to
`results/<scenario>.json`.

Timing methodology: 1 warm-up call (discarded) + `trials` measured
calls with the same problem and λ-grid. Median is the headline number.
"""

from __future__ import annotations


from benches.problems import SIZES, gaussian_lasso

from . import _common

PENALTY = "lasso"
FAMILY = "gaussian"
LAMBDA_MIN_RATIO = 1e-3
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
