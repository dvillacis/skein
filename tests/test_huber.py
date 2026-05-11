"""Tests for Huber-robust regression (M3.7)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

skein_glm = pytest.importorskip("skein_glm")


def _outlier_problem(seed: int = 0, n: int = 200, p: int = 10):
    """Sparse-truth LS problem with a small fraction of large outliers.

    Without robustification, ordinary LS gets badly pulled by the
    outliers; Huber should recover something very close to the true β
    even when ~5 % of `y` is contaminated.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 2.0
    true_beta[1] = -1.5
    true_beta[2] = 1.0
    signal = x @ true_beta
    y = signal + 0.1 * rng.standard_normal(n)
    # Inject ~5% outliers at ±5σ
    n_outliers = max(1, n // 20)
    idx = rng.choice(n, n_outliers, replace=False)
    y[idx] += 5.0 * rng.choice([-1.0, 1.0], n_outliers)
    return x, y, true_beta


# --- Path / shape ------------------------------------------------------

def test_huber_mcp_path_shape():
    x, y, _ = _outlier_problem(seed=1)
    m = skein_glm.HuberMCPPathRegressor(
        delta=1.345, gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2,
    ).fit(x, y)
    assert m.coefs_.shape == (20, x.shape[1])
    assert m.intercepts_.shape == (20,)
    assert m.lambdas_.shape == (20,)
    # λ should be strictly decreasing.
    assert np.all(np.diff(m.lambdas_) < 0)


def test_huber_scad_path_shape():
    x, y, _ = _outlier_problem(seed=2)
    m = skein_glm.HuberSCADPathRegressor(
        delta=1.345, a=3.7, n_lambdas=15, lambda_min_ratio=1e-2,
    ).fit(x, y)
    assert m.coefs_.shape == (15, x.shape[1])
    assert m.intercepts_.shape == (15,)


# --- Signal recovery ---------------------------------------------------

def test_huber_mcp_recovers_signal_with_outliers():
    x, y, true_beta = _outlier_problem(seed=3)
    m = skein_glm.HuberMCPPathRegressor(
        delta=1.345, gamma=3.0, n_lambdas=30, lambda_min_ratio=1e-3,
    ).fit(x, y)
    # At the smallest λ the non-zero truth should be recovered with sign.
    last = m.coefs_[-1]
    for k in [0, 1, 2]:
        assert np.sign(last[k]) == np.sign(true_beta[k]), (
            f"feature {k} sign mismatch: β = {last[k]}, truth = {true_beta[k]}"
        )
    # Magnitudes should be close to the truth.
    np.testing.assert_allclose(last[:3], true_beta[:3], atol=0.15)


def test_huber_outperforms_least_squares_under_contamination():
    """The whole point of Huber: on data with outliers, the estimated
    coefficients should be closer to the truth than plain LS.
    """
    x, y, true_beta = _outlier_problem(seed=4, n=400)
    m_huber = skein_glm.HuberMCPRegressor(
        lambda_=0.005, delta=1.345, gamma=1e6,  # effectively lasso inner
    ).fit(x, y)
    m_ls = skein_glm.MCPRegressor(
        lambda_=0.005, gamma=1e6,
    ).fit(x, y)
    err_huber = float(np.linalg.norm(m_huber.coef_ - true_beta))
    err_ls = float(np.linalg.norm(m_ls.coef_ - true_beta))
    assert err_huber < err_ls, (
        f"Huber should beat LS under contamination: "
        f"err_huber={err_huber:.4f}, err_ls={err_ls:.4f}"
    )


# --- Reduction to LS ---------------------------------------------------

def test_huber_large_delta_matches_least_squares():
    """δ ≫ max|r| ⇒ Huber ≡ LS; the regularized fits should coincide."""
    rng = np.random.default_rng(5)
    n, p = 150, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p); true_beta[:3] = [1.0, -0.7, 0.4]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)  # *No* outliers
    m_huber = skein_glm.HuberMCPPathRegressor(
        delta=100.0, gamma=1e6, n_lambdas=10, lambda_min_ratio=1e-2,
    ).fit(x, y)
    m_ls = skein_glm.MCPPathRegressor(
        gamma=1e6, lambdas=m_huber.lambdas_,
    ).fit(x, y)
    # Same λ-grid + clean data + large δ ⇒ near-identical β up to
    # prox-Newton outer-loop tolerance.
    np.testing.assert_allclose(m_huber.coefs_, m_ls.coefs_, atol=1e-4)


# --- Lambda_max -------------------------------------------------------

def test_huber_at_lambda_max_returns_zero_on_clean_data():
    """At the auto-derived `λ_max`, every coefficient should be zero —
    *if* the surrogate is stationary across prox-Newton iterates. That
    holds for LS (`surrogate ≡ data`) and for Huber when no residual
    crosses `δ` (weights stay uniformly 1). With outliers the surrogate
    changes between outer iterations and a feature can escape the
    boundary at γ < ∞, so this property is checked on clean data only.
    """
    rng = np.random.default_rng(6)
    n, p = 200, 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p); true_beta[:3] = [1.0, -0.7, 0.4]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)  # no outliers
    # δ much larger than any residual ⇒ uniform IRLS weights w_i = 1.
    m = skein_glm.HuberMCPPathRegressor(
        delta=100.0, gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-3,
    ).fit(x, y)
    np.testing.assert_allclose(m.coefs_[0], 0.0, atol=1e-6)


# --- Validation -------------------------------------------------------

def test_huber_rejects_nonfinite_y():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((20, 3))
    y = np.array([1.0, 2.0, np.nan] + [0.0] * 17)
    with pytest.raises(ValueError, match=r"finite y"):
        skein_glm.HuberMCPRegressor(lambda_=0.1, delta=1.0).fit(x, y)


def test_huber_rejects_nonpositive_delta():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal(20)
    with pytest.raises(ValueError, match=r"delta"):
        skein_glm.HuberMCPRegressor(lambda_=0.1, delta=0.0).fit(x, y)
    with pytest.raises(ValueError, match=r"delta"):
        skein_glm.HuberMCPRegressor(lambda_=0.1, delta=-1.0).fit(x, y)


def test_huber_sparse_x_not_supported_yet():
    """Sparse path is deferred; the dense-only check should raise a
    helpful NotImplementedError rather than crashing inside PyO3."""
    rng = np.random.default_rng(9)
    x = sparse.random(50, 5, density=0.3, random_state=rng, format="csc")
    y = rng.standard_normal(50)
    with pytest.raises(NotImplementedError, match=r"Sparse Huber"):
        skein_glm.HuberMCPRegressor(lambda_=0.1, delta=1.0).fit(x, y)


# --- predict / decision_function --------------------------------------

def test_huber_predict_matches_decision_function():
    x, y, _ = _outlier_problem(seed=10, n=100)
    m = skein_glm.HuberMCPRegressor(lambda_=0.05, delta=1.0).fit(x, y)
    yhat_decision = m.decision_function(x)
    yhat_predict = m.predict(x)
    np.testing.assert_allclose(yhat_predict, yhat_decision)
    # Path predict shape: (n_samples, n_lambdas).
    mp = skein_glm.HuberMCPPathRegressor(
        delta=1.0, n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, y)
    yhat_path = mp.predict(x)
    assert yhat_path.shape == (x.shape[0], 8)
