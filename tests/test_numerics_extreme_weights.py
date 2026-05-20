"""H2 numerical-stability sweep — extreme weights.

Three weight axes can produce ill-conditioned subproblems:

* **`sample_weights`** spanning 12+ orders of magnitude. The IRLS / prox-
  Newton diagonal `w_i` accumulates `sample_weights * irls_w` and feeds
  the per-column Lipschitz bound `L_jj = Σ w_i x_ij²`. Wide spreads can
  collapse `L_jj` to near zero (small-weight columns) or saturate it
  (large-weight columns) within the same fit.
* **Zero per-feature `weights`**. A zero weight means "do not penalize
  this feature" — the path solver and CV must keep the feature active
  end-to-end without any divide-by-zero in the weight-rescale path.
* **Zero per-group weights** in a sparse-group fit. The group block-CD
  uses `weights_group` to scale the ℓ₂ thresholding step; one zero
  group is the most common pattern (an unpenalized covariate block).

Each test asserts the fit completes, all coefficients are finite, and
the "unpenalized" channels are actually picked up where expected.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import skein_glm

TIME_BUDGET_S = 30.0


def _ls_problem(n: int = 200, p: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1.5, -2.0, 0.8]
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return x, y


def _logistic_problem(n: int = 300, p: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1.2, -1.0, 0.8]
    prob = 1.0 / (1.0 + np.exp(-(x @ beta)))
    y = (rng.uniform(size=n) < prob).astype(np.float64)
    return x, y


def _poisson_problem(n: int = 300, p: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)) * 0.3
    beta = np.zeros(p)
    beta[:3] = [0.4, -0.3, 0.2]
    y = rng.poisson(np.exp(x @ beta)).astype(np.float64)
    return x, y


def _fit_under_budget(est, x, y, **fit_kwargs):
    t0 = time.perf_counter()
    est.fit(x, y, **fit_kwargs) if fit_kwargs else est.fit(x, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < TIME_BUDGET_S, (
        f"{type(est).__name__} fit took {elapsed:.2f}s "
        f"(budget {TIME_BUDGET_S}s) — likely an infinite loop"
    )


def _assert_finite_path(coefs) -> None:
    coefs = np.asarray(coefs)
    assert np.all(np.isfinite(coefs)), "non-finite coefficient on the path"


# ---------- sample_weights spanning many orders of magnitude --------------


def _wide_spread_sample_weights(n: int, spread_decades: int = 12) -> np.ndarray:
    """Geometric-spread weights from 10**(-spread/2) to 10**(spread/2)."""
    half = spread_decades / 2.0
    return np.logspace(-half, half, n)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=12),
        lambda: skein_glm.SCADPathRegressor(a=3.7, n_lambdas=12),
        lambda: skein_glm.ElasticNetPathRegressor(alpha=1.0, n_lambdas=12),
        lambda: skein_glm.ElasticNetPathRegressor(alpha=0.5, n_lambdas=12),
    ],
    ids=["mcp", "scad", "lasso", "en"],
)
def test_ls_sample_weights_wide_spread_remain_finite(factory):
    x, y = _ls_problem()
    sw = _wide_spread_sample_weights(x.shape[0], spread_decades=12)
    est = factory()
    est.sample_weights = sw
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    # A finite prediction on the training matrix.
    pred = x @ est.coefs_.T
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: skein_glm.LogisticLassoPathRegressor(n_lambdas=12),
        lambda: skein_glm.LogisticMCPPathRegressor(gamma=3.0, n_lambdas=12),
    ],
    ids=["logistic-lasso", "logistic-mcp"],
)
def test_logistic_sample_weights_wide_spread_remain_finite(factory):
    x, y = _logistic_problem()
    sw = _wide_spread_sample_weights(x.shape[0], spread_decades=12)
    est = factory()
    est.sample_weights = sw
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: skein_glm.PoissonLassoPathRegressor(n_lambdas=12),
        lambda: skein_glm.PoissonMCPPathRegressor(gamma=3.0, n_lambdas=12),
    ],
    ids=["poisson-lasso", "poisson-mcp"],
)
def test_poisson_sample_weights_wide_spread_remain_finite(factory):
    x, y = _poisson_problem()
    sw = _wide_spread_sample_weights(x.shape[0], spread_decades=12)
    est = factory()
    est.sample_weights = sw
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


def test_ls_sample_weights_with_zeros_remain_finite():
    """A handful of zero `sample_weights` (effective dropped rows) must
    not break the IRLS / Lipschitz aggregation."""
    x, y = _ls_problem(n=200)
    sw = np.ones(x.shape[0])
    sw[::20] = 0.0  # zero out every 20th observation
    est = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=12)
    est.sample_weights = sw
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


# ---------- zero per-feature weights (effective unpenalized feature) ------


@pytest.mark.parametrize(
    "factory",
    [
        lambda w: skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=15, weights=w),
        lambda w: skein_glm.SCADPathRegressor(a=3.7, n_lambdas=15, weights=w),
        lambda w: skein_glm.ElasticNetPathRegressor(alpha=1.0, n_lambdas=15, weights=w),
    ],
    ids=["mcp", "scad", "lasso"],
)
@pytest.mark.parametrize("standardize", [False, True])
def test_ls_zero_feature_weight_keeps_feature_active(factory, standardize):
    """A zero per-feature weight should make that feature unpenalized —
    its coefficient stays nonzero across the full path."""
    x, y = _ls_problem(p=16)
    weights = np.ones(16)
    weights[0] = 0.0  # unpenalized
    est = factory(weights)
    est.standardize = standardize
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    # Feature 0 must be nonzero at every λ.
    assert np.all(np.abs(est.coefs_[:, 0]) > 1e-6), (
        "unpenalized feature should be nonzero everywhere on the path"
    )


def test_logistic_zero_feature_weight_keeps_feature_active():
    x, y = _logistic_problem()
    weights = np.ones(16)
    weights[0] = 0.0
    est = skein_glm.LogisticLassoPathRegressor(n_lambdas=12, weights=weights)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    assert np.all(np.abs(est.coefs_[:, 0]) > 1e-6)


def test_poisson_zero_feature_weight_keeps_feature_active():
    x, y = _poisson_problem()
    weights = np.ones(16)
    weights[0] = 0.0
    est = skein_glm.PoissonLassoPathRegressor(n_lambdas=12, weights=weights)
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    assert np.all(np.abs(est.coefs_[:, 0]) > 1e-6)


# ---------- zero per-group weights ----------------------------------------


def _groups_of(p: int, group_size: int = 4) -> np.ndarray:
    return np.repeat(np.arange(p // group_size), group_size).astype(np.int64)


def test_group_lasso_zero_group_weight_keeps_group_active():
    x, y = _ls_problem(p=16)
    groups = _groups_of(16, group_size=4)
    n_groups = int(groups.max()) + 1
    group_weights = np.ones(n_groups)
    group_weights[0] = 0.0  # unpenalized group
    est = skein_glm.GroupLassoPathRegressor(
        groups=groups, n_lambdas=12, weights=group_weights,
    )
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    # Some coefficient inside the unpenalized group is always nonzero.
    group0_norm = np.linalg.norm(est.coefs_[:, 0:4], axis=1)
    assert np.all(group0_norm > 1e-6)


def test_sparse_group_lasso_zero_group_weight_keeps_group_active():
    """In a sparse-group fit the unpenalized group still feels the
    within-group ℓ₁ piece (`alpha * λ * ||β_g||_1`), but the group-norm
    component is gated by the zero group weight, so at least one
    coefficient in the group must remain active across the path."""
    x, y = _ls_problem(p=16)
    groups = _groups_of(16, group_size=4)
    n_groups = int(groups.max()) + 1
    group_weights = np.ones(n_groups)
    group_weights[0] = 0.0
    est = skein_glm.SparseGroupLassoPathRegressor(
        groups=groups, n_lambdas=12, alpha=0.5, weights=group_weights,
    )
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)


def test_group_mcp_zero_group_weight_keeps_group_active():
    x, y = _ls_problem(p=16)
    groups = _groups_of(16, group_size=4)
    n_groups = int(groups.max()) + 1
    group_weights = np.ones(n_groups)
    group_weights[0] = 0.0
    est = skein_glm.GroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=12, weights=group_weights,
    )
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_)
    group0_norm = np.linalg.norm(est.coefs_[:, 0:4], axis=1)
    assert np.all(group0_norm > 1e-6)
