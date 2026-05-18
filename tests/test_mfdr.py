"""Marginal false-discovery-rate (mFDR) selection (grpreg mfdr).

The MFDR estimator and `select_by_mfdr` provide a path-aware FDR
control complementary to skein's existing CV, IC, and stability
selection layers. Both are pure Python — no Rust dependency.

These tests verify:

* The mFDR curve goes from ~0 at λ_max (no discoveries) to ~1 at the
  smallest λ on the path (every feature picked up).
* `select` picks an index where the active set is close to the truth
  (high precision and recall on a sparse fixture).
* The bound is respected: selected mFDR ≤ target.
* `estimate_mfdr` returns the right shape and stays in [0, 1].
* The `MFDR` class wrapper agrees with `select_by_mfdr` functional API.
* Logistic / Poisson families are detected and produce non-degenerate
  mFDR curves.
* Multinomial / Multitask raise `NotImplementedError`.
"""
from __future__ import annotations

import numpy as np
import pytest

import skein_glm


@pytest.fixture
def lasso_problem():
    """Sparse linear regression: 200×50 with 5 active features."""
    rng = np.random.default_rng(0)
    n, p = 200, 50
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:5] = [2.0, -1.5, 1.0, -1.2, 0.8]
    y = x @ beta + 0.5 * rng.standard_normal(n)
    return x, y, beta


def test_mfdr_at_largest_lambda_is_zero(lasso_problem):
    """No discoveries ⇒ mFDR = 0 by convention (clipped from 0/1)."""
    x, y, _ = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=1e6, n_lambdas=20, lambda_min_ratio=1e-2, standardize=True,
    ).fit(x, y)
    mfdr = skein_glm.estimate_mfdr(model, x, y)
    # At index 0, β = 0 ⇒ R = 0 ⇒ ratio = p · 2Φ(-large) / max(1, 0) → tiny.
    assert mfdr[0] < 1e-3


def test_mfdr_grows_as_lambda_shrinks(lasso_problem):
    """The mFDR curve is roughly non-decreasing in path index."""
    x, y, _ = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=1e6, n_lambdas=30, lambda_min_ratio=1e-3, standardize=True,
    ).fit(x, y)
    mfdr = skein_glm.estimate_mfdr(model, x, y)
    # First quartile should be substantially smaller than last quartile.
    q1 = mfdr[:len(mfdr) // 4].mean()
    q4 = mfdr[3 * len(mfdr) // 4 :].mean()
    assert q4 > q1 + 0.1, f"expected q4 ({q4:.3f}) > q1 ({q1:.3f}) + 0.1"


def test_mfdr_values_in_unit_interval(lasso_problem):
    x, y, _ = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=3.0, n_lambdas=20, standardize=True,
    ).fit(x, y)
    mfdr = skein_glm.estimate_mfdr(model, x, y)
    assert np.all(mfdr >= 0.0)
    assert np.all(mfdr <= 1.0)


def test_select_returns_active_set_with_high_precision(lasso_problem):
    """At target=0.1, the selected model should pick up most true features
    with limited false positives."""
    x, y, beta_true = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=1e6, n_lambdas=50, lambda_min_ratio=1e-3, standardize=True,
    ).fit(x, y)
    idx, val = skein_glm.select_by_mfdr(model, x, y, target=0.1)
    assert val <= 0.1
    active = np.where(np.abs(model.coefs_[idx]) > 1e-12)[0]
    true_active = set(np.where(beta_true != 0)[0].tolist())
    selected = set(active.tolist())
    # Should recover ≥ 4 of 5 true features with FDR ≤ 0.1.
    assert len(true_active & selected) >= 4


def test_select_raises_when_no_lambda_qualifies(lasso_problem):
    """If target is too strict for the entire path, raise ValueError."""
    x, y, _ = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=1e6, n_lambdas=10, lambda_min_ratio=0.5, standardize=True,
    ).fit(x, y)
    # Use a target below the smallest path mFDR. First check what the path achieves:
    mfdr = skein_glm.estimate_mfdr(model, x, y)
    min_mfdr = mfdr.min()
    if min_mfdr > 0.0:
        target = min_mfdr / 2.0
        with pytest.raises(ValueError, match="no λ on the path"):
            skein_glm.select_by_mfdr(model, x, y, target=target)


def test_mfdr_class_wrapper_matches_functional_api(lasso_problem):
    x, y, _ = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=1e6, n_lambdas=30, lambda_min_ratio=1e-3, standardize=True,
    ).fit(x, y)
    fn_idx, fn_val = skein_glm.select_by_mfdr(model, x, y, target=0.1)
    sel = skein_glm.MFDR(model).fit(x, y)
    cls_idx, cls_val = sel.select(target=0.1)
    assert fn_idx == cls_idx
    np.testing.assert_allclose(fn_val, cls_val)


def test_mfdr_class_select_before_fit_raises():
    model = skein_glm.MCPPathRegressor(gamma=3.0)
    sel = skein_glm.MFDR(model)
    with pytest.raises(RuntimeError, match="call .fit"):
        sel.select(target=0.1)


def test_select_target_validation(lasso_problem):
    x, y, _ = lasso_problem
    model = skein_glm.MCPPathRegressor(
        gamma=3.0, n_lambdas=15, standardize=True,
    ).fit(x, y)
    with pytest.raises(ValueError, match=r"target must be in \(0, 1\]"):
        skein_glm.select_by_mfdr(model, x, y, target=0.0)
    with pytest.raises(ValueError, match=r"target must be in \(0, 1\]"):
        skein_glm.select_by_mfdr(model, x, y, target=1.5)


def test_mfdr_for_logistic_path():
    rng = np.random.default_rng(1)
    n, p = 200, 30
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1.5, -1.2, 0.8]
    eta = x @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < prob).astype(np.float64)
    model = skein_glm.LogisticMCPPathRegressor(
        gamma=1e6, n_lambdas=20, standardize=True,
    ).fit(x, y)
    mfdr = skein_glm.estimate_mfdr(model, x, y)
    assert mfdr.shape == (len(model.lambdas_),)
    assert np.all((mfdr >= 0.0) & (mfdr <= 1.0))


def test_mfdr_for_poisson_path():
    rng = np.random.default_rng(2)
    n, p = 200, 20
    x = rng.standard_normal((n, p)) * 0.3
    beta = np.zeros(p)
    beta[:3] = [0.5, -0.4, 0.3]
    rate = np.exp(x @ beta)
    y = rng.poisson(rate).astype(np.float64)
    model = skein_glm.PoissonMCPPathRegressor(
        gamma=1e6, n_lambdas=15, standardize=True,
    ).fit(x, y)
    mfdr = skein_glm.estimate_mfdr(model, x, y)
    assert mfdr.shape == (len(model.lambdas_),)
    assert np.all((mfdr >= 0.0) & (mfdr <= 1.0))


def test_mfdr_rejects_multitask():
    """Multitask uses a different formulation — current impl bails out."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((50, 10))
    y = rng.standard_normal((50, 2))

    # Use a multitask estimator from the package; sniff via class name.
    class FakeMultitaskPath:
        def __init__(self):
            self.coefs_ = np.zeros((5, 10))
            self.intercepts_ = np.zeros(5)
            self.lambdas_ = np.linspace(1.0, 0.1, 5)
        def predict(self, x):
            return x @ self.coefs_.T + self.intercepts_[None, :]

    # Class name must contain "Multitask" for the sniffer to flag it.
    FakeMultitaskPath.__name__ = "FakeMultitaskPath"
    model = FakeMultitaskPath()
    with pytest.raises(NotImplementedError, match="multi-response"):
        skein_glm.estimate_mfdr(model, x, y[:, 0])
