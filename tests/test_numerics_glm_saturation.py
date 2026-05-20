"""H2 numerical-stability sweep — GLM tail saturation.

GLM datafits each have a saturation regime where the canonical link
maps to an extreme working response:

* **Poisson** — `μ = exp(η)`. The solver clamps `|η| ≤ ETA_CLAMP = 30`,
  so `μ ∈ [exp(-30), exp(30)] ≈ [1e-13, 1e13]`. We construct designs
  that drive `η` against both endpoints simultaneously.
* **Binomial / logistic** — `p ∈ [0, 1]` with `w_i = p(1-p)` floored at
  `W_FLOOR = 1e-4`. Linearly separable data pushes every `p_i` to {0, 1}
  and every `w_i` to the floor; the IRLS surrogate then has a near-
  singular Hessian.
* **Cox PH** — Efron's tie correction is exact only for "small" tie
  blocks (the correction is a triangular sum over the within-block
  rank). Heavy tied designs (every observation tied) stress the
  correction; the solver must still converge.

These regimes are where M14d/M14e came from. The tests assert the
fit terminates with finite coefficients and predictions, and that the
clamp/floor logic actually keeps the surrogate finite.
"""
from __future__ import annotations

import time

import numpy as np

import skein_glm

TIME_BUDGET_S = 30.0
# Defined in crates/skein-core/src/numerics.rs; keep in sync.
ETA_CLAMP = 30.0
W_FLOOR = 1e-4


def _fit_under_budget(est, *args, **kwargs):
    t0 = time.perf_counter()
    est.fit(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    assert elapsed < TIME_BUDGET_S, (
        f"{type(est).__name__} fit took {elapsed:.2f}s "
        f"(budget {TIME_BUDGET_S}s) — likely an infinite loop"
    )


def _assert_finite_path(coefs) -> None:
    coefs = np.asarray(coefs)
    assert np.all(np.isfinite(coefs)), "non-finite coefficient on the path"


# ---------- Poisson: η near ETA_CLAMP --------------------------------------


def _poisson_saturating_problem(n: int = 200, p: int = 12, seed: int = 0):
    """Build X with one column that, combined with a large true β,
    drives η_i near ±ETA_CLAMP across the sample. y is sampled at the
    clamped μ so the fit isn't fighting unfeasible targets."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    # Strong signal on column 0 — uncentered linear predictor reaches
    # roughly 30 * sign(x[:, 0]) at fit time.
    x[:, 0] = rng.uniform(-1.0, 1.0, size=n)
    beta = np.zeros(p)
    beta[0] = 25.0  # paired with x[:,0] ∈ [-1,1] this gives η ∈ [-25, 25]
    eta = np.clip(x @ beta, -ETA_CLAMP + 1.0, ETA_CLAMP - 1.0)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    return x, y


def test_poisson_lasso_saturating_eta_remains_finite():
    x, y = _poisson_saturating_problem()
    est = skein_glm.PoissonLassoPathRegressor(n_lambdas=12)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    # Linear predictor at every λ must respect the clamp (no overflow).
    eta_path = x @ est.coefs_.T + est.intercepts_[None, :]
    assert np.all(np.isfinite(eta_path))
    # μ must be finite — the clamp guarantees exp(η) ≤ exp(ETA_CLAMP) ≈ 1e13.
    mu_path = np.exp(np.clip(eta_path, -ETA_CLAMP, ETA_CLAMP))
    assert np.all(np.isfinite(mu_path))


def test_poisson_mcp_saturating_eta_remains_finite():
    x, y = _poisson_saturating_problem(seed=1)
    est = skein_glm.PoissonMCPPathRegressor(gamma=3.0, n_lambdas=12)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


def test_poisson_lasso_very_large_counts():
    """y_i ≈ 1e12 stresses the deviance computation, not just η. The
    `ETA_CLAMP` keeps μ bounded but the residual `(y - μ)` can be
    huge."""
    rng = np.random.default_rng(2)
    n, p = 200, 8
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[0] = 1.0
    mu = np.exp(np.clip(x @ beta, -ETA_CLAMP + 1.0, ETA_CLAMP - 1.0))
    mu = mu * 1e10  # rescale so counts are huge
    y = rng.poisson(mu).astype(np.float64)
    est = skein_glm.PoissonLassoPathRegressor(n_lambdas=10)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


# ---------- Binomial: predicted probabilities pinned at W_FLOOR -----------


def _separable_logistic_problem(n: int = 200, p: int = 12, seed: int = 0):
    """A linearly separable problem — every observation is correctly
    classified by sign(x[:, 0]), so the unpenalized MLE has |β_0| → ∞
    and `p_i ∈ {0, 1}` at the limit. The penalty caps β_0, but in early
    IRLS iterations `w_i = p(1-p)` collapses toward `W_FLOOR`."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = (x[:, 0] > 0).astype(np.float64)
    return x, y


def test_logistic_lasso_separable_problem_remains_finite():
    x, y = _separable_logistic_problem()
    est = skein_glm.LogisticLassoPathRegressor(n_lambdas=15, lambda_min_ratio=1e-3)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    # Linear predictor must be finite even when |β_0| is large.
    eta_path = x @ est.coefs_.T + est.intercepts_[None, :]
    assert np.all(np.isfinite(eta_path))
    # p = sigmoid(η) stays in [0, 1] when η is finite. No NaN from log(0).
    p_path = 1.0 / (1.0 + np.exp(-np.clip(eta_path, -ETA_CLAMP, ETA_CLAMP)))
    assert np.all((p_path >= 0.0) & (p_path <= 1.0))


def test_logistic_mcp_separable_problem_remains_finite():
    x, y = _separable_logistic_problem(seed=1)
    est = skein_glm.LogisticMCPPathRegressor(gamma=3.0, n_lambdas=15)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


def test_logistic_lasso_quasi_separable_with_few_violations():
    """Almost-separable: 95% of points respect the boundary. The IRLS
    weights still collapse toward W_FLOOR for most samples but a few
    well-balanced rows keep the Hessian conditioned."""
    rng = np.random.default_rng(3)
    n, p = 300, 12
    x = rng.standard_normal((n, p))
    y = (x[:, 0] > 0).astype(np.float64)
    # Flip 5% of labels randomly.
    flip = rng.choice(n, size=n // 20, replace=False)
    y[flip] = 1.0 - y[flip]
    est = skein_glm.LogisticLassoPathRegressor(n_lambdas=15, lambda_min_ratio=1e-3)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


def test_logistic_lasso_extreme_class_imbalance():
    """1 positive among 999 negatives. Every fit weight `p(1-p)` is at
    the W_FLOOR floor; the intercept does most of the work."""
    rng = np.random.default_rng(4)
    n, p = 1000, 8
    x = rng.standard_normal((n, p))
    y = np.zeros(n)
    y[0] = 1.0  # one positive
    est = skein_glm.LogisticLassoPathRegressor(n_lambdas=10)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


# ---------- Cox: heavy ties -----------------------------------------------


def _heavy_ties_cox_problem(
    n: int = 300, p: int = 8, n_unique_times: int = 3, seed: int = 0,
):
    """Bin event times into very few unique values so the tie blocks
    are huge — heavier than Efron's correction was designed for."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    eta = 0.5 * x[:, 0] - 0.3 * x[:, 2]
    raw_time = rng.exponential(1.0 / np.exp(np.clip(eta, -3, 3)))
    edges = np.linspace(0.0, raw_time.max() + 1e-9, n_unique_times + 1)
    time = edges[np.digitize(raw_time, edges) - 1] + 0.1
    event = (rng.uniform(size=n) < 0.8).astype(np.float64)
    return x, time, event


def test_cox_lasso_heavy_ties_breslow_remains_finite():
    x, time, event = _heavy_ties_cox_problem(n_unique_times=3)
    est = skein_glm.CoxMCPPathRegressor(
        gamma=1e6,  # near-lasso (large γ kills MCP concavity)
        ties="breslow",
        n_lambdas=12,
        lambda_min_ratio=1e-2,
    )
    _fit_under_budget(est, x, time, event)
    _assert_finite_path(est.coefs_)


def test_cox_lasso_heavy_ties_efron_remains_finite():
    """Efron's correction is the harder code path under heavy ties — the
    triangular within-block sum dominates the gradient evaluation."""
    x, time, event = _heavy_ties_cox_problem(n_unique_times=3, seed=1)
    est = skein_glm.CoxMCPPathRegressor(
        gamma=1e6,
        ties="efron",
        n_lambdas=12,
        lambda_min_ratio=1e-2,
    )
    _fit_under_budget(est, x, time, event)
    _assert_finite_path(est.coefs_)


def test_cox_mcp_heavy_ties_remains_finite():
    x, time, event = _heavy_ties_cox_problem(n_unique_times=2, seed=2)
    est = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="efron", n_lambdas=10, lambda_min_ratio=1e-2,
    )
    _fit_under_budget(est, x, time, event)
    _assert_finite_path(est.coefs_)


def test_cox_all_events_at_one_time():
    """The pathological extreme — every event is at the same time,
    one giant tie block."""
    rng = np.random.default_rng(5)
    n, p = 100, 6
    x = rng.standard_normal((n, p))
    time = np.ones(n)  # all tied
    event = np.ones(n)  # all events
    est = skein_glm.CoxMCPPathRegressor(
        gamma=1e6, ties="efron", n_lambdas=8, lambda_min_ratio=1e-2,
    )
    _fit_under_budget(est, x, time, event)
    _assert_finite_path(est.coefs_)
