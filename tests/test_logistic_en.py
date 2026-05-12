"""Tests for the first-class logistic elastic-net + lasso primitives.

These replace the prior `LogisticMCPRegressor(gamma=1e9)` convention. The
new primitives call prox-Newton with the convex `ElasticNet` penalty
directly — no LLA outer loop — so the result is a proper convex solve
rather than a numerical approximation.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from skein_glm import (
    LogisticElasticNetPathCV,
    LogisticElasticNetPathRegressor,
    LogisticElasticNetRegressor,
    LogisticLassoPathCV,
    LogisticLassoPathRegressor,
    LogisticLassoRegressor,
    LogisticMCPRegressor,
)


def _make_logistic_problem(n=200, p=30, s=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = rng.uniform(0.8, 1.5, size=s) * rng.choice([-1.0, 1.0], size=s)
    prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = (rng.uniform(size=n) < prob).astype(float)
    return X, y, beta


# --- shape and convergence ------------------------------------------


def test_en_single_lambda_shape_and_active() -> None:
    X, y, _ = _make_logistic_problem()
    est = LogisticElasticNetRegressor(lambda_=0.05, alpha=0.5).fit(X, y)
    assert est.coef_.shape == (30,)
    assert isinstance(est.intercept_, float)
    assert est.n_features_in_ == 30


def test_lasso_single_lambda_alpha_pinned_to_one() -> None:
    """LogisticLassoRegressor is a facade around EN(alpha=1.0)."""
    X, y, _ = _make_logistic_problem()
    a = LogisticLassoRegressor(lambda_=0.05).fit(X, y)
    b = LogisticElasticNetRegressor(lambda_=0.05, alpha=1.0).fit(X, y)
    np.testing.assert_allclose(a.coef_, b.coef_, atol=1e-12)
    np.testing.assert_allclose(a.intercept_, b.intercept_, atol=1e-12)


def test_path_shape() -> None:
    X, y, _ = _make_logistic_problem()
    est = LogisticElasticNetPathRegressor(alpha=0.5, n_lambdas=20).fit(X, y)
    assert est.coefs_.shape == (20, 30)
    assert est.intercepts_.shape == (20,)
    assert est.lambdas_.shape == (20,)
    # λ_max should give the all-zero solution.
    assert np.all(est.coefs_[0] == 0.0)


def test_lasso_path_facade() -> None:
    """LogisticLassoPathRegressor ≡ LogisticElasticNetPathRegressor(alpha=1)."""
    X, y, _ = _make_logistic_problem()
    a = LogisticLassoPathRegressor(n_lambdas=15, lambda_min_ratio=1e-3).fit(X, y)
    b = LogisticElasticNetPathRegressor(
        alpha=1.0, n_lambdas=15, lambda_min_ratio=1e-3,
    ).fit(X, y)
    np.testing.assert_allclose(a.coefs_, b.coefs_, atol=1e-12)
    np.testing.assert_allclose(a.lambdas_, b.lambdas_, atol=1e-12)


# --- correctness vs MCP-at-large-γ at the same λ --------------------


def test_lasso_matches_mcp_at_large_gamma_on_support() -> None:
    """The new convex lasso primitive should give the same active set
    as `LogisticMCPRegressor(gamma=1e9)`. Coefficients differ by a few
    percent (the MCP-at-γ=1e9 was an approximation), but support is
    identical and the new fit is closer to sklearn / glmnet."""
    X, y, _ = _make_logistic_problem(n=200, p=30, s=3, seed=10)
    lasso = LogisticLassoRegressor(
        lambda_=0.05, max_outer=30, outer_tol=1e-10, tol=1e-10,
    ).fit(X, y)
    mcp_approx = LogisticMCPRegressor(
        lambda_=0.05, gamma=1e9, max_outer=30, outer_tol=1e-10, tol=1e-10,
    ).fit(X, y)
    active_new = np.abs(lasso.coef_) > 1e-6
    active_old = np.abs(mcp_approx.coef_) > 1e-6
    np.testing.assert_array_equal(active_new, active_old)


def test_lasso_recovers_sparse_signal() -> None:
    X, y, beta = _make_logistic_problem(n=400, p=20, s=3, seed=20)
    est = LogisticLassoPathRegressor(n_lambdas=30).fit(X, y)
    # Some λ along the path should give the right support.
    true_support = np.abs(beta) > 1e-12
    matches = [
        np.array_equal(np.abs(est.coefs_[k]) > 1e-6, true_support)
        for k in range(est.coefs_.shape[0])
    ]
    assert any(matches), "no λ on the path recovered the true support"


# --- alpha=0 limit: ridge ------------------------------------------


def test_alpha_zero_is_ridge_no_sparsity() -> None:
    """alpha=0 → ridge: shrinks but does not introduce exact zeros at
    a moderate λ."""
    X, y, _ = _make_logistic_problem(n=300, p=20, s=5, seed=30)
    ridge = LogisticElasticNetRegressor(
        lambda_=0.05, alpha=0.0, max_outer=30, outer_tol=1e-10, tol=1e-10,
    ).fit(X, y)
    assert int(np.sum(np.abs(ridge.coef_) < 1e-8)) <= 1


def test_alpha_intermediate_sparser_than_ridge_less_sparse_than_lasso() -> None:
    """At a fixed λ: ridge (α=0) has the most active features; lasso
    (α=1) the fewest; α=0.5 lies between."""
    X, y, _ = _make_logistic_problem(n=200, p=40, s=3, seed=40)
    lam = 0.08
    ridge = LogisticElasticNetRegressor(lambda_=lam, alpha=0.0).fit(X, y)
    en = LogisticElasticNetRegressor(lambda_=lam, alpha=0.5).fit(X, y)
    lasso = LogisticLassoRegressor(lambda_=lam).fit(X, y)
    active_r = int(np.sum(np.abs(ridge.coef_) > 1e-6))
    active_en = int(np.sum(np.abs(en.coef_) > 1e-6))
    active_l = int(np.sum(np.abs(lasso.coef_) > 1e-6))
    assert active_l <= active_en <= active_r


# --- input dispatch ------------------------------------------------


def test_sparse_input_parity() -> None:
    """csc_matrix and ndarray inputs produce identical coefs."""
    X, y, _ = _make_logistic_problem(n=150, p=20, s=3, seed=50)
    Xs = sparse.csc_matrix(X)
    dense = LogisticLassoRegressor(lambda_=0.05).fit(X, y)
    sp = LogisticLassoRegressor(lambda_=0.05).fit(Xs, y)
    np.testing.assert_allclose(dense.coef_, sp.coef_, atol=1e-7)


def test_sparse_path_parity() -> None:
    X, y, _ = _make_logistic_problem(n=150, p=20, s=3, seed=51)
    Xs = sparse.csc_matrix(X)
    dense = LogisticElasticNetPathRegressor(alpha=0.5, n_lambdas=10).fit(X, y)
    sp = LogisticElasticNetPathRegressor(alpha=0.5, n_lambdas=10).fit(Xs, y)
    np.testing.assert_allclose(dense.coefs_, sp.coefs_, atol=1e-7)


# --- sklearn-style predict semantics inherited ---------------------


def test_predict_proba_and_class_label() -> None:
    X, y, _ = _make_logistic_problem(n=200, p=15, s=3, seed=60)
    est = LogisticLassoRegressor(lambda_=0.05).fit(X, y)
    proba = est.predict_proba(X)
    # 1D since single λ; values in [0, 1].
    assert proba.ndim == 1
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    classes = est.predict(X)
    assert set(np.unique(classes)).issubset({0.0, 1.0, 0, 1})


def test_path_predict_proba_2d() -> None:
    X, y, _ = _make_logistic_problem(n=200, p=15, s=3, seed=61)
    est = LogisticLassoPathRegressor(n_lambdas=12).fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (200, 12)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


# --- sample weights -------------------------------------------------


def test_sample_weights_change_fit() -> None:
    """Passing nontrivial sample weights produces a different fit than
    uniform — the underlying prox-Newton honors them."""
    X, y, _ = _make_logistic_problem(n=200, p=15, s=3, seed=70)
    sw = np.ones_like(y)
    sw[:50] = 4.0  # heavy-weight the first quarter
    a = LogisticElasticNetPathRegressor(alpha=0.5, n_lambdas=10).fit(X, y)
    b = LogisticElasticNetPathRegressor(
        alpha=0.5, n_lambdas=10, sample_weights=sw,
    ).fit(X, y)
    # Should not be identical at all λs.
    assert not np.allclose(a.coefs_, b.coefs_)


# --- CV wrappers ---------------------------------------------------


def test_lasso_path_cv_picks_a_lambda() -> None:
    X, y, _ = _make_logistic_problem(n=200, p=15, s=3, seed=80)
    cv = LogisticLassoPathCV(n_lambdas=15, cv=3, random_state=0).fit(X, y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (15,)
    assert hasattr(cv, "cv_mean_scores_")


def test_en_path_cv_picks_a_lambda() -> None:
    X, y, _ = _make_logistic_problem(n=200, p=15, s=3, seed=81)
    cv = LogisticElasticNetPathCV(
        alpha=0.5, n_lambdas=15, cv=3, random_state=0,
    ).fit(X, y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (15,)


# --- input validation ----------------------------------------------


def test_rejects_alpha_out_of_range() -> None:
    X, y, _ = _make_logistic_problem(n=50, p=10, s=2)
    with pytest.raises(ValueError, match="alpha"):
        LogisticElasticNetRegressor(lambda_=0.05, alpha=1.5).fit(X, y)
    with pytest.raises(ValueError, match="alpha"):
        LogisticElasticNetRegressor(lambda_=0.05, alpha=-0.1).fit(X, y)


def test_rejects_non_binary_y() -> None:
    X, y, _ = _make_logistic_problem(n=50, p=10, s=2)
    y_bad = y.copy()
    y_bad[0] = 2.0
    with pytest.raises(ValueError):
        LogisticLassoRegressor(lambda_=0.05).fit(X, y_bad)
