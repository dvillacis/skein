"""MCP on Gaussian LS, *deep-path* regime.

  λ_min / λ_max = 1e-3
  γ = 3.0  (the ncvreg / skglm / skein default; pinned for fairness)

This is the nonconvex counterpart of `lasso_ls`. The deep path drives
β past the MCP knee (|β_j| > γ·λ) into the unpenalised tail, which is
where MCP differentiates itself from lasso — heavy shrinkage near the
origin, unbiased estimates of the truly large coefficients. The path
also makes the LLA-vs-pure-CD difference visible: tighter inner solves
matter more when each outer LLA iterate moves through the nonconvex
region.

Comparators are intentionally narrow (skein, skglm, R/ncvreg): glmnet
is convex-only, sklearn / celer / pyglmnet have no MCP. For the
active-set-stays-small regime see `mcp_ls_sparse`.

Timing methodology, runner ABI, and result schema match `lasso_ls`.
"""

from __future__ import annotations


from benches.problems import SIZES, gaussian_lasso

from . import _common

PENALTY = "mcp"
FAMILY = "gaussian"
LAMBDA_MIN_RATIO = 1e-3
GAMMA = 3.0
R_PACKAGES = {"r": "ncvreg"}


def run(*, runner, package: str, size: str, tol: float, n_lambdas: int, trials: int = 5) -> dict:
    problem = gaussian_lasso(SIZES[size])
    grid = _common.lambda_grid(problem.x, problem.y, n_lambdas, LAMBDA_MIN_RATIO)

    if package == "r":
        per_trial, last = _common.time_r_runner(
            package="ncvreg",
            penalty=PENALTY,
            family=FAMILY,
            problem=problem,
            lambda_grid=grid,
            tol=tol,
            trials=trials,
            extra={"gamma": GAMMA},
        )
        return _common.summarize(
            package="ncvreg",
            version=last.get("version", "unknown"),
            problem=problem,
            n_lambdas=n_lambdas,
            per_trial=per_trial,
            active_set_size=int(last["active_set_size"]),
            size=size,
            extra={
                "via": "Rscript",
                "lambda_min_ratio": LAMBDA_MIN_RATIO,
                "gamma": GAMMA,
            },
        )

    per_trial, last = _common.time_python_runner(
        runner=runner,
        problem=problem,
        penalty=PENALTY,
        lambda_grid=grid,
        tol=tol,
        trials=trials,
        gamma=GAMMA,
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
            "gamma": GAMMA,
        },
        n_iter=last.n_iter,
        final_obj=last.final_obj,
    )
