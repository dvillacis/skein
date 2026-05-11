"""MCP on Gaussian LS, *sparse* regime.

  λ_min / λ_max = 5e-2
  γ = 3.0

Same problem family as `mcp_ls` but the path stops near support
recovery rather than running into the saturated tail. This is the
regime MCP is actually designed for — pick λ via CV in a range where
most features stay zero, and γ controls how quickly the penalty fades
to zero on the truly nonzero coefficients.

Runner ABI, timing methodology, and result schema match `lasso_ls`.
Only `LAMBDA_MIN_RATIO` differs from `mcp_ls`.
"""

from __future__ import annotations


from benches.problems import SIZES, gaussian_lasso

from . import _common

PENALTY = "mcp"
FAMILY = "gaussian"
LAMBDA_MIN_RATIO = 5e-2
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
