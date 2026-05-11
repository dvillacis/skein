"""SCAD on Gaussian LS, *deep-path* regime.

  λ_min / λ_max = 1e-3
  γ = 3.7  (the ncvreg / skein SCAD default; pinned for fairness)

SCAD is MCP's older sibling — same nonconvex flavour (quadratic spline,
unbiased on truly large coefficients) but a different penalty shape
and a conventional γ of 3.7 instead of 3.0.

Comparators: skein, skglm, and R/ncvreg. skglm does not ship a SCAD
estimator wrapper, but `skglm.penalties.SCAD` is exposed and the
runner assembles a warm-started α-loop through
`GeneralizedLinearEstimator` so the comparison stays apples-to-apples.

Runner ABI, timing methodology, and result schema match `mcp_ls`.
"""

from __future__ import annotations


from benches.problems import SIZES, gaussian_lasso

from . import _common

PENALTY = "scad"
FAMILY = "gaussian"
LAMBDA_MIN_RATIO = 1e-3
GAMMA = 3.7
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
