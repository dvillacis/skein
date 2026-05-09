"""Tests for adaptive {Lasso, MCP, SCAD} estimators (M6.x)."""

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


def test_adaptive_lasso_recovers_signal():
    x, y, true_beta = _problem(0)
    m = skein_glm.AdaptiveLassoPathRegressor(
        n_lambdas=15, lambda_min_ratio=1e-2
    ).fit(x, y)
    last = m.coefs_[-1]
    for j in [0, 2, 4]:
        assert np.sign(last[j]) == np.sign(true_beta[j])
        assert abs(last[j] - true_beta[j]) < 0.2
    for j in [1, 3, 5, 6, 7, 8, 9]:
        assert abs(last[j]) < 0.1


def test_adaptive_estimators_expose_pilot_artifacts():
    x, y, _ = _problem(1)
    m = skein_glm.AdaptiveMCPPathRegressor(
        gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    # `coef_pilot_` and `weights_` are populated for inspection.
    assert m.coef_pilot_.shape == (10,)
    assert m.weights_.shape == (10,)
    # Inactive features should have huge adaptive weights (pilot β ≈ 0).
    assert m.weights_.max() > 1e3
    # Active features should have small weights (pilot β meaningful).
    active = np.argsort(np.abs(m.coef_pilot_))[-3:]
    assert m.weights_[active].max() < 10.0


def test_adaptive_lasso_pilot_position_last_uses_smallest_lambda():
    x, y, _ = _problem(2)
    mid = skein_glm.AdaptiveLassoPathRegressor(
        pilot_position="mid", n_pilot_lambdas=10, n_lambdas=8,
        lambda_min_ratio=1e-2,
    ).fit(x, y)
    last = skein_glm.AdaptiveLassoPathRegressor(
        pilot_position="last", n_pilot_lambdas=10, n_lambdas=8,
        lambda_min_ratio=1e-2,
    ).fit(x, y)
    # 'last' uses smaller λ ⇒ larger pilot magnitudes ⇒ smaller weights.
    assert np.linalg.norm(last.coef_pilot_) > np.linalg.norm(mid.coef_pilot_)


def test_adaptive_mcp_path_cv_picks_active_features():
    x, y, true_beta = _problem(3, n=300)
    cv = skein_glm.AdaptiveMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=15, lambda_min_ratio=1e-3
    ).fit(x, y)
    assert cv.coef_.shape == (10,)
    for j in [0, 2, 4]:
        assert np.sign(cv.coef_[j]) == np.sign(true_beta[j])
    # CV exposes the pilot artifacts too.
    assert cv.weights_.shape == (10,)


def test_adaptive_scad_pred_shape():
    x, y, _ = _problem(4, n=80, p=6)
    m = skein_glm.AdaptiveSCADPathRegressor(
        a=3.7, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    pred = m.predict(x)
    assert pred.shape == (x.shape[0], 8)


def test_adaptive_dense_sparse_equivalence_lasso():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y, _ = _problem(5, n=80, p=8)
    x_csc = sparse.csc_matrix(x)
    m_d = skein_glm.AdaptiveLassoPathRegressor(
        n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, y)
    m_s = skein_glm.AdaptiveLassoPathRegressor(
        lambdas=m_d.lambdas_,
    ).fit(x_csc, y)
    np.testing.assert_allclose(m_d.coefs_, m_s.coefs_, atol=1e-4)
    np.testing.assert_allclose(m_d.weights_, m_s.weights_, atol=1e-4)


def test_adaptive_pilot_position_validation():
    x, y, _ = _problem(6, n=40)
    with pytest.raises(ValueError, match="pilot_position"):
        skein_glm.AdaptiveLassoPathRegressor(
            pilot_position="invalid", n_lambdas=4
        ).fit(x, y)
    with pytest.raises(ValueError, match="out of range"):
        skein_glm.AdaptiveLassoPathRegressor(
            pilot_position=99, n_pilot_lambdas=10, n_lambdas=4,
        ).fit(x, y)


def test_adaptive_eta_validation():
    x, y, _ = _problem(7, n=40)
    with pytest.raises(ValueError, match="eta must be > 0"):
        skein_glm.AdaptiveLassoPathRegressor(eta=-1.0, n_lambdas=4).fit(x, y)
    with pytest.raises(ValueError, match="eps_pilot must be > 0"):
        skein_glm.AdaptiveLassoPathRegressor(
            eta=1.0, eps_pilot=0.0, n_lambdas=4
        ).fit(x, y)


def test_adaptive_higher_eta_sparsifies_more():
    """Higher η pushes inactive-feature weights up faster, so the
    final fit zeroes more features at the same λ."""
    x, y, _ = _problem(8, n=200)
    mid_lam = None
    nactives = []
    for eta in [0.5, 1.0, 2.0]:
        m = skein_glm.AdaptiveLassoPathRegressor(
            eta=eta, n_lambdas=10, lambda_min_ratio=1e-2
        ).fit(x, y)
        if mid_lam is None:
            mid_lam = len(m.lambdas_) // 2
        nactives.append(int(np.sum(np.abs(m.coefs_[mid_lam]) > 1e-6)))
    # Monotone non-increasing in eta.
    assert nactives[0] >= nactives[1] >= nactives[2]


def test_adaptive_lasso_predict_after_cv():
    x, y, _ = _problem(9, n=80)
    cv = skein_glm.AdaptiveLassoPathCV(
        cv=3, random_state=0, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    pred = cv.predict(x)
    assert pred.shape == (x.shape[0],)


# =========================================================================
# Adaptive group estimators
# =========================================================================


def _group_problem(seed: int = 0, n: int = 200):
    """5 groups of 2 features each. Groups 0 and 2 are active."""
    rng = np.random.default_rng(seed)
    p = 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -0.8
    true_beta[4] = 1.0
    true_beta[5] = -0.6
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    return x, y, groups, true_beta


def test_adaptive_group_lasso_recovers_active_groups():
    x, y, groups, _ = _group_problem(0)
    m = skein_glm.AdaptiveGroupLassoPathRegressor(
        groups=groups, n_lambdas=15, lambda_min_ratio=1e-2
    ).fit(x, y)
    last = m.coefs_[-1]
    # Groups 0, 2 active; 1, 3, 4 zero.
    for j in [0, 1, 4, 5]:  # active
        assert abs(last[j]) > 0.4
    for j in [2, 3, 6, 7, 8, 9]:  # noise
        assert abs(last[j]) < 0.1
    # Per-group weights: 5 entries, 2 small (active groups), 3 huge.
    assert m.weights_.shape == (5,)
    assert np.sum(m.weights_ > 1e3) >= 3  # noise groups blown up


def test_adaptive_group_mcp_path_cv_picks_active_groups():
    x, y, groups, _ = _group_problem(1, n=300)
    cv = skein_glm.AdaptiveGroupMCPPathCV(
        groups=groups, gamma=3.0, cv=3, random_state=0, n_lambdas=12, lambda_min_ratio=1e-3
    ).fit(x, y)
    assert cv.coef_.shape == (10,)
    for j in [0, 1, 4, 5]:
        assert abs(cv.coef_[j]) > 0.3


def test_adaptive_group_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y, groups, _ = _group_problem(2, n=80)
    x_csc = sparse.csc_matrix(x)
    m_d = skein_glm.AdaptiveGroupLassoPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, y)
    m_s = skein_glm.AdaptiveGroupLassoPathRegressor(
        groups=groups, lambdas=m_d.lambdas_,
    ).fit(x_csc, y)
    np.testing.assert_allclose(m_d.coefs_, m_s.coefs_, atol=1e-4)
    np.testing.assert_allclose(m_d.weights_, m_s.weights_, atol=1e-4)


def test_adaptive_group_predict_shape():
    x, y, groups, _ = _group_problem(3, n=60)
    m = skein_glm.AdaptiveGroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    pred = m.predict(x)
    assert pred.shape == (x.shape[0], 8)


def test_adaptive_group_validation():
    x, y, groups, _ = _group_problem(4, n=40)
    with pytest.raises(ValueError, match=r"eta must be > 0"):
        skein_glm.AdaptiveGroupLassoPathRegressor(
            groups=groups, eta=-1.0, n_lambdas=4
        ).fit(x, y)
    with pytest.raises(ValueError, match=r"eps_pilot must be > 0"):
        skein_glm.AdaptiveGroupLassoPathRegressor(
            groups=groups, eps_pilot=0.0, n_lambdas=4
        ).fit(x, y)
