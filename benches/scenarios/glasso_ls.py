"""Graphical Lasso (L1 on Σ⁻¹) scenario.

Glasso scales like O(p³) per outer iteration, so the v1 dimension table
(`small / medium / large = n=1k/10k/50k, p=100/1k/5k`) does not transfer:
p=5k means a 5000×5000 working covariance and each iteration is ≈10¹¹
flops. Glasso-specific dimensions:

  small  : p = 20,  n = 200    (10 sec budget)
  medium : p = 100, n = 1000   (60 sec budget)
  large  : p = 200, n = 2000   (a few minutes)

The Ω simulator is shared with the v2 suite (`benches.v2.simulators.glasso_truth`),
banded topology with bandwidth 2.

skein and sklearn both lack a native path solver for glasso, so the
runners loop single-alpha fits along the grid. Wall-clock is the sum of
per-alpha solves, comparable apples-to-apples.
"""

from __future__ import annotations


import numpy as np

from benches.v2.simulators import glasso_truth

from . import _common

PENALTY = "glasso"
FAMILY = "gaussian"

# Glasso-specific sizes (p, n) — keyed by the same size names as SIZES
# but with much smaller p because of the O(p³) inner cost.
GLASSO_SIZES: dict[str, tuple[int, int]] = {
    "small": (20, 200),
    "medium": (100, 1_000),
    "large": (200, 2_000),
}


def _alpha_grid(x: np.ndarray, n_alphas: int = 20) -> np.ndarray:
    """Geometric grid from `alpha_max` (max off-diagonal |S_ij|) down to
    `alpha_max * 1e-2`. Stops sooner than the LS path because solving
    glasso at very small α is slow and rarely interesting (graph fills
    in). 20 alphas keeps the bench cell tractable at p=200."""
    n = x.shape[0]
    s = (x.T @ x) / n
    off_diag = s - np.diag(np.diag(s))
    alpha_max = float(np.abs(off_diag).max())
    return np.geomspace(alpha_max, alpha_max * 1e-2, n_alphas)


def run(*, runner, package: str, size: str, tol: float, n_lambdas: int, trials: int = 5) -> dict:
    if package == "r":
        # R `glasso` is supported by the v2 runner via Arrow IPC; the v1
        # r_runner.R does not include a glasso branch. Skip cleanly.
        raise NotImplementedError("v1 r_runner.R has no glasso branch (see benches/v2)")

    if package not in ("skein", "sklearn"):
        # skglm / celer / pyglmnet have no graphical-model fitters.
        raise NotImplementedError(f"{package}: glasso not supported")

    p, n = GLASSO_SIZES[size]
    problem = glasso_truth.make(n=n, p=p, seed=1, topology="banded")
    grid = _alpha_grid(problem.x, n_alphas=20)

    per_trial, last = _common.time_python_runner(
        runner=runner,
        problem=problem,
        penalty=PENALTY,
        lambda_grid=grid,
        tol=tol,
        trials=trials,
    )
    extra = {
        k: v for k, v in last.extra.items() if k != "info"
    }
    extra["alpha_min_ratio"] = 1e-2
    extra["topology"] = "banded"
    extra["n_edges_true"] = int(problem.meta["n_edges_true"])
    return _common.summarize(
        package=last.package,
        version=last.version,
        problem=problem,
        n_lambdas=int(grid.size),
        per_trial=per_trial,
        active_set_size=int(last.active_set_size),
        size=size,
        extra=extra,
        n_iter=last.n_iter,
        final_obj=last.final_obj,
    )
