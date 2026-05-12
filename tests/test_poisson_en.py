"""Tests for the first-class Poisson elastic-net + lasso primitives.

Mirrors `tests/test_logistic_en.py`. Retires the prior
`PoissonMCPRegressor(gamma=1e9)` convention used by
`AdaptivePoissonLasso*`.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from skein_glm import (
    PoissonElasticNetPathCV,
    PoissonElasticNetPathRegressor,
    PoissonElasticNetRegressor,
    PoissonLassoPathCV,
    PoissonLassoPathRegressor,
    PoissonLassoRegressor,
    PoissonMCPRegressor,
)


def _make_poisson_problem(n=200, p=20, s=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = rng.uniform(0.3, 0.6, size=s) * rng.choice([-1.0, 1.0], size=s)
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(float)
    return X, y, beta


def test_en_single_lambda_shape() -> None:
    X, y, _ = _make_poisson_problem()
    est = PoissonElasticNetRegressor(lambda_=0.05, alpha=0.5).fit(X, y)
    assert est.coef_.shape == (20,)
    assert isinstance(est.intercept_, float)
    assert est.n_features_in_ == 20


def test_lasso_single_lambda_facade() -> None:
    X, y, _ = _make_poisson_problem()
    a = PoissonLassoRegressor(lambda_=0.05).fit(X, y)
    b = PoissonElasticNetRegressor(lambda_=0.05, alpha=1.0).fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-12)
    np.testing.assert_allclose(a.intercept_, b.intercept_, atol=1e-12)


def test_path_shape() -> None:
    X, y, _ = _make_poisson_problem()
    est = PoissonElasticNetPathRegressor(alpha=0.5, n_lambdas=20).fit(X, y)
    assert est.coefs_.shape == (20, 20)
    assert np.all(est.coefs_[0] == 0.0)


def test_lasso_path_facade() -> None:
    X, y, _ = _make_poisson_problem()
    a = PoissonLassoPathRegressor(n_lambdas=15).fit(X, y)
    b = PoissonElasticNetPathRegressor(alpha=1.0, n_lambdas=15).fit(X, y)
    np.testing.assert_allclose(a.coefs_, b.coefs_, atol=1e-12)


def test_lasso_matches_mcp_at_large_gamma_on_support() -> None:
    X, y, _ = _make_poisson_problem(n=300, p=20, s=3, seed=10)
    lasso = PoissonLassoRegressor(
        lambda_=0.05, max_outer=30, outer_tol=1e-10, tol=1e-10,
    ).fit(X, y)
    mcp = PoissonMCPRegressor(
        lambda_=0.05, gamma=1e9, max_outer=30, outer_tol=1e-10, tol=1e-10,
    ).fit(X, y)
    active_new = np.abs(lasso.coef_) > 1e-6
    active_old = np.abs(mcp.coef_) > 1e-6
    np.testing.assert_array_equal(active_new, active_old)


def test_lasso_recovers_sparse_signal() -> None:
    X, y, beta = _make_poisson_problem(n=500, p=20, s=3, seed=20)
    est = PoissonLassoPathRegressor(n_lambdas=30).fit(X, y)
    true_support = np.abs(beta) > 1e-12
    matches = [
        np.array_equal(np.abs(est.coefs_[k]) > 1e-4, true_support)
        for k in range(est.coefs_.shape[0])
    ]
    assert any(matches), "no λ on the path recovered the true support"


def test_alpha_zero_is_ridge_no_sparsity() -> None:
    X, y, _ = _make_poisson_problem(n=300, p=15, s=4, seed=30)
    ridge = PoissonElasticNetRegressor(
        lambda_=0.05, alpha=0.0, max_outer=30, outer_tol=1e-10, tol=1e-10,
    ).fit(X, y)
    assert int(np.sum(np.abs(ridge.coef_) < 1e-8)) <= 1


def test_alpha_sparsity_ordering() -> None:
    X, y, _ = _make_poisson_problem(n=200, p=30, s=3, seed=40)
    lam = 0.08
    ridge = PoissonElasticNetRegressor(lambda_=lam, alpha=0.0).fit(X, y)
    en = PoissonElasticNetRegressor(lambda_=lam, alpha=0.5).fit(X, y)
    lasso = PoissonLassoRegressor(lambda_=lam).fit(X, y)
    active_r = int(np.sum(np.abs(ridge.coef_) > 1e-6))
    active_en = int(np.sum(np.abs(en.coef_) > 1e-6))
    active_l = int(np.sum(np.abs(lasso.coef_) > 1e-6))
    assert active_l <= active_en <= active_r


def test_sparse_input_parity() -> None:
    X, y, _ = _make_poisson_problem(n=150, p=15, s=3, seed=50)
    Xs = sparse.csc_matrix(X)
    dense = PoissonLassoRegressor(lambda_=0.05).fit(X, y)
    sp = PoissonLassoRegressor(lambda_=0.05).fit(Xs, y)
    np.testing.assert_allclose(dense.coef_, sp.coef_, atol=1e-7)


def test_offset_changes_fit() -> None:
    """Passing a nontrivial Poisson offset (log-exposure) changes the
    fitted coefficients — same plumbing as the MCP/SCAD Poisson siblings."""
    X, y, _ = _make_poisson_problem(n=200, p=15, s=3, seed=60)
    rng = np.random.default_rng(60)
    offset = rng.uniform(-0.5, 0.5, size=200)
    a = PoissonElasticNetPathRegressor(alpha=0.5, n_lambdas=10).fit(X, y)
    b = PoissonElasticNetPathRegressor(
        alpha=0.5, n_lambdas=10, offset=offset,
    ).fit(X, y)
    assert not np.allclose(a.coefs_, b.coefs_)


def test_predict_returns_mu() -> None:
    """predict on Poisson returns μ = exp(η) per the inherited
    _PoissonRegressorBase semantics."""
    X, y, _ = _make_poisson_problem(n=200, p=15, s=3, seed=70)
    est = PoissonLassoRegressor(lambda_=0.05).fit(X, y)
    mu_hat = est.predict(X)
    assert np.all(mu_hat > 0)
    assert mu_hat.shape == (200,)


def test_path_cv_picks_a_lambda() -> None:
    X, y, _ = _make_poisson_problem(n=200, p=15, s=3, seed=80)
    cv = PoissonLassoPathCV(n_lambdas=15, cv=3, random_state=0).fit(X, y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (15,)


def test_en_path_cv() -> None:
    X, y, _ = _make_poisson_problem(n=200, p=15, s=3, seed=81)
    cv = PoissonElasticNetPathCV(alpha=0.5, n_lambdas=15, cv=3, random_state=0).fit(X, y)
    assert cv.lambda_best_ > 0


def test_rejects_alpha_out_of_range() -> None:
    X, y, _ = _make_poisson_problem(n=50, p=10, s=2)
    with pytest.raises(ValueError, match="alpha"):
        PoissonElasticNetRegressor(lambda_=0.05, alpha=1.5).fit(X, y)


def test_rejects_negative_y() -> None:
    X, y, _ = _make_poisson_problem(n=50, p=10, s=2)
    y_bad = y.copy()
    y_bad[0] = -1.0
    with pytest.raises(ValueError):
        PoissonLassoRegressor(lambda_=0.05).fit(X, y_bad)
