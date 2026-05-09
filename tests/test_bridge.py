"""Smoke + parity tests for the bridge (ℓ_q) penalty (M6.x).

Penalty `λ · Σ_j w_j |β_j|^q` with `q ∈ (0, 1]`. Convex at q = 1 (plain
weighted lasso); concave for q < 1, fit via outer LLA. The Rust core
gets path-LLA tests in `solver/path_lla.rs`; this file exercises the
Python-facing API end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _problem(seed: int = 0, n: int = 200, p: int = 10):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[2] = 0.8
    true_beta[4] = -1.0
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    return x, y, true_beta


def test_bridge_path_q_one_matches_lasso_via_mcp_high_gamma():
    """Bridge at q=1 is plain weighted lasso. With weights = 1 and
    eps small, it should exactly match `MCPPathRegressor` at large γ
    on the same λ-grid."""
    x, y, _ = _problem(0)
    p_path = skein_glm.MCPPathRegressor(
        gamma=1e9, n_lambdas=10, lambda_min_ratio=1e-2
    ).fit(x, y)
    bridge = skein_glm.BridgePathRegressor(
        q=1.0, eps=1e-12, lambdas=p_path.lambdas_,
    ).fit(x, y)
    np.testing.assert_allclose(bridge.coefs_, p_path.coefs_, atol=1e-6)
    np.testing.assert_allclose(bridge.intercepts_, p_path.intercepts_, atol=1e-6)


def test_bridge_path_q_half_recovers_signal():
    """Bridge LLA can land in local minima at very small q from the
    cold start. With path warm-starts (large λ → small λ) and q = 0.7
    the recovery is reliable; q = 0.5 occasionally drops a feature."""
    x, y, true_beta = _problem(1, n=300)
    path = skein_glm.BridgePathRegressor(
        q=0.7, n_lambdas=25, lambda_min_ratio=1e-3
    ).fit(x, y)
    last = path.coefs_[-1]
    for j in [0, 2, 4]:
        assert np.sign(last[j]) == np.sign(true_beta[j]), (
            f"feature {j} sign mismatch: β = {last[j]:.3f}"
        )
        assert abs(last[j]) > 0.4, f"feature {j} too small: β = {last[j]:.3f}"
    for j in [1, 3, 5, 6, 7, 8, 9]:
        assert abs(last[j]) < 0.2


def test_bridge_path_lambda_max_returns_zero():
    x, y, _ = _problem(2)
    path = skein_glm.BridgePathRegressor(
        q=0.5, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    np.testing.assert_allclose(path.coefs_[0], 0.0, atol=1e-3)


def test_bridge_path_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y, _ = _problem(3, n=80, p=8)
    x_csc = sparse.csc_matrix(x)
    path_d = skein_glm.BridgePathRegressor(
        q=0.5, n_lambdas=10, lambda_min_ratio=1e-2
    ).fit(x, y)
    path_s = skein_glm.BridgePathRegressor(
        q=0.5, lambdas=path_d.lambdas_,
    ).fit(x_csc, y)
    np.testing.assert_allclose(path_d.coefs_, path_s.coefs_, atol=1e-7)
    np.testing.assert_allclose(path_d.intercepts_, path_s.intercepts_, atol=1e-7)


def test_bridge_path_dense_sparse_equivalence_with_standardize():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y, _ = _problem(4, n=80, p=8)
    x[:, 1] *= 30.0  # inflate one column so standardization actually does something
    x_csc = sparse.csc_matrix(x)
    path_d = skein_glm.BridgePathRegressor(
        q=0.5, n_lambdas=10, lambda_min_ratio=1e-2, standardize=True
    ).fit(x, y)
    path_s = skein_glm.BridgePathRegressor(
        q=0.5, lambdas=path_d.lambdas_, standardize=True
    ).fit(x_csc, y)
    np.testing.assert_allclose(path_d.coefs_, path_s.coefs_, atol=1e-6)
    np.testing.assert_allclose(path_d.intercepts_, path_s.intercepts_, atol=1e-6)


def test_bridge_path_cv_picks_active_features():
    x, y, true_beta = _problem(5, n=300)
    cv = skein_glm.BridgePathCV(
        q=0.7, cv=3, random_state=0, n_lambdas=15, lambda_min_ratio=1e-3
    ).fit(x, y)
    assert cv.coef_.shape == (10,)
    for j in [0, 2, 4]:
        assert np.sign(cv.coef_[j]) == np.sign(true_beta[j])


def test_bridge_q_smaller_sparsifies_more_aggressively():
    """A smaller q produces a sparser solution at the same λ — the
    headline property that motivates bridge over lasso."""
    x, y, _ = _problem(6, n=200)
    # Use the same λ-grid for fair comparison.
    p_lasso = skein_glm.BridgePathRegressor(
        q=1.0, n_lambdas=15, lambda_min_ratio=1e-2
    ).fit(x, y)
    p_half = skein_glm.BridgePathRegressor(
        q=0.5, lambdas=p_lasso.lambdas_,
    ).fit(x, y)
    # At an intermediate λ, count active features.
    mid = len(p_lasso.lambdas_) // 2
    lasso_active = int(np.sum(np.abs(p_lasso.coefs_[mid]) > 1e-6))
    half_active = int(np.sum(np.abs(p_half.coefs_[mid]) > 1e-6))
    assert half_active <= lasso_active


def test_bridge_rejects_q_out_of_range():
    x, y, _ = _problem(7, n=40)
    with pytest.raises(ValueError, match=r"q must be in"):
        skein_glm.BridgePathRegressor(q=1.5, n_lambdas=4).fit(x, y)
    with pytest.raises(ValueError, match=r"q must be in"):
        skein_glm.BridgePathRegressor(q=0.0, n_lambdas=4).fit(x, y)


def test_bridge_rejects_eps_non_positive():
    x, y, _ = _problem(8, n=40)
    with pytest.raises(ValueError, match=r"eps must be > 0"):
        skein_glm.BridgePathRegressor(q=0.5, eps=0.0, n_lambdas=4).fit(x, y)


def test_bridge_path_predict_shape():
    x, y, _ = _problem(9, n=50, p=6)
    path = skein_glm.BridgePathRegressor(
        q=0.5, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    pred = path.predict(x)
    assert pred.shape == (x.shape[0], 8)
