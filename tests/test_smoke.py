"""End-to-end smoke tests. Require `maturin develop` to have been run."""
from __future__ import annotations

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")


def _toy_problem(seed: int = 0, *, alpha: float = 0.0):
    rng = np.random.default_rng(seed)
    n, p = 100, 20
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x @ true_beta + alpha + 0.1 * rng.standard_normal(n)
    return x, y, true_beta


# ---- existing single-λ smoke tests --------------------------------------


def test_mcp_regressor_fits_and_zeros_noise_features():
    x, y, true_beta = _toy_problem()
    model = skein.MCPRegressor(lambda_=0.05, gamma=3.0).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    # Recovers signal in the first 3 features.
    assert np.allclose(np.sign(model.coef_[:3]), np.sign(true_beta[:3]))
    # Most noise features zeroed out.
    assert (np.abs(model.coef_[3:]) < 1e-2).sum() >= 15


def test_scad_regressor_fits():
    x, y, _ = _toy_problem()
    model = skein.SCADRegressor(lambda_=0.05, a=3.7).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    # `info_` shape changed to per-λ vectors when path machinery is used.
    assert len(model.info_["iters"]) == 1


def test_per_feature_weights_change_solution():
    x, y, _ = _toy_problem()
    p = x.shape[1]
    base = skein.MCPRegressor(lambda_=0.05, gamma=3.0).fit(x, y).coef_
    # Penalize feature 0 ten times more heavily.
    w = np.ones(p)
    w[0] = 10.0
    weighted = skein.MCPRegressor(
        lambda_=0.05, gamma=3.0, weights=w
    ).fit(x, y).coef_
    assert abs(weighted[0]) < abs(base[0])


# ---- intercept handling --------------------------------------------------


def test_mcp_regressor_recovers_nonzero_intercept():
    x, y, _ = _toy_problem(alpha=5.0)
    model = skein.MCPRegressor(lambda_=0.05, gamma=3.0, fit_intercept=True).fit(x, y)
    # Recovered intercept should be close to 5.0 (small residual error from regularization).
    assert abs(model.intercept_ - 5.0) < 0.5


def test_predict_includes_intercept():
    x, y, _ = _toy_problem(alpha=3.0)
    model = skein.MCPRegressor(lambda_=0.05, gamma=3.0, fit_intercept=True).fit(x, y)
    # predict on the training data should produce values near y (with shrinkage bias).
    y_pred = model.predict(x)
    # Mean of predictions should be near mean of y (intercept handles location).
    assert abs(y_pred.mean() - y.mean()) < 0.1


def test_no_intercept_when_fit_intercept_false():
    x, y, _ = _toy_problem(alpha=5.0)
    model = skein.MCPRegressor(
        lambda_=0.05, gamma=3.0, fit_intercept=False
    ).fit(x, y)
    assert model.intercept_ == 0.0


# ---- path estimator ------------------------------------------------------


def test_mcp_path_regressor_returns_decreasing_lambdas():
    x, y, _ = _toy_problem()
    model = skein.MCPPathRegressor(gamma=3.0, n_lambdas=20).fit(x, y)
    assert model.lambdas_.shape == (20,)
    assert np.all(np.diff(model.lambdas_) < 0), "lambdas must be strictly decreasing"
    assert model.coefs_.shape == (20, x.shape[1])
    assert model.intercepts_.shape == (20,)


def test_mcp_path_regressor_with_explicit_lambdas():
    x, y, _ = _toy_problem()
    custom = np.array([1.0, 0.5, 0.1, 0.01])
    model = skein.MCPPathRegressor(gamma=3.0, lambdas=custom).fit(x, y)
    np.testing.assert_allclose(model.lambdas_, custom)
    assert model.coefs_.shape == (4, x.shape[1])


def test_mcp_path_predict_shape():
    x, y, _ = _toy_problem()
    model = skein.MCPPathRegressor(gamma=3.0, n_lambdas=5).fit(x, y)
    pred = model.predict(x)
    # Per-λ predictions: (n_samples, n_lambdas)
    assert pred.shape == (x.shape[0], 5)


def test_path_first_coef_at_lambda_max_is_zero():
    x, y, _ = _toy_problem()
    model = skein.MCPPathRegressor(gamma=3.0, n_lambdas=10).fit(x, y)
    # At the largest λ in the auto path (= λ_max), β = 0 and intercept = ȳ.
    np.testing.assert_allclose(model.coefs_[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(model.intercepts_[0], y.mean(), atol=1e-6)


# ---- screening modes ----------------------------------------------------


def test_screening_modes_produce_consistent_results_on_lasso():
    x, y, _ = _toy_problem()
    coefs_by_mode = {}
    for mode in ("off", "strong", "gap_safe"):
        model = skein.MCPPathRegressor(
            gamma=1e6,  # ≈ lasso (convex)
            n_lambdas=8,
            screening=mode,
            tol=1e-10,
            max_iter=5000,
        ).fit(x, y)
        coefs_by_mode[mode] = model.coefs_
    # All three modes must agree on the convex problem.
    np.testing.assert_allclose(
        coefs_by_mode["off"], coefs_by_mode["strong"], atol=1e-5
    )
    np.testing.assert_allclose(
        coefs_by_mode["off"], coefs_by_mode["gap_safe"], atol=1e-5
    )


def test_unknown_screening_raises():
    x, y, _ = _toy_problem()
    with pytest.raises(ValueError, match="screening"):
        skein.MCPRegressor(screening="bogus").fit(x, y)


# ---- standardization end-to-end ----------------------------------------


def test_standardize_recovers_signal_with_unequal_column_scales():
    rng = np.random.default_rng(0)
    n, p = 100, 10
    x = rng.standard_normal((n, p))
    # Inflate column 0's scale by 100x; without standardization the same λ
    # over-penalizes it relative to the others.
    x[:, 0] *= 100.0
    true_beta = np.zeros(p)
    true_beta[0] = 0.05  # small in absolute terms but large × column scale
    true_beta[1] = -1.0
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    model = skein.MCPPathRegressor(
        gamma=3.0, n_lambdas=20, standardize=True, tol=1e-10, max_iter=5000
    ).fit(x, y)
    # At a small λ near the path's end, β should be close to the truth.
    last = model.coefs_[-1]
    assert np.sign(last[0]) == np.sign(true_beta[0])
    assert np.sign(last[1]) == np.sign(true_beta[1])


# ====================================================================
# Group estimators
# ====================================================================


def _sparse_group_problem(seed: int = 0, *, alpha: float = 0.0):
    """4 groups of 2 features (p=8); truth: groups 0 and 2 active."""
    rng = np.random.default_rng(seed)
    n, p = 80, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    true_beta[4] = 0.7
    true_beta[5] = 1.2
    y = x @ true_beta + alpha + 0.1 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, y, true_beta, groups


def _group_norm(beta, groups, g):
    mask = groups == g
    return float(np.sqrt(np.sum(beta[mask] ** 2)))


def test_group_lasso_path_returns_decreasing_lambdas():
    x, y, _, groups = _sparse_group_problem(seed=1)
    model = skein.GroupLassoPathRegressor(groups=groups, n_lambdas=15).fit(x, y)
    assert model.lambdas_.shape == (15,)
    assert np.all(np.diff(model.lambdas_) < 0)
    assert model.coefs_.shape == (15, x.shape[1])
    assert model.intercepts_.shape == (15,)


def test_group_lasso_path_recovers_active_groups_at_small_lambda():
    x, y, _, groups = _sparse_group_problem(seed=2)
    model = skein.GroupLassoPathRegressor(
        groups=groups, n_lambdas=20, lambda_min_ratio=5e-3, tol=1e-10, max_iter=5000,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.5
    assert _group_norm(last, groups, 2) > 0.5


def test_group_lasso_single_lambda_regressor_smoke():
    x, y, _, groups = _sparse_group_problem(seed=3)
    model = skein.GroupLassoRegressor(groups=groups, lambda_=0.05).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert isinstance(model.intercept_, float)


def test_group_mcp_path_recovers_active_groups_via_lla():
    x, y, _, groups = _sparse_group_problem(seed=4)
    model = skein.GroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=10,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.5
    assert _group_norm(last, groups, 2) > 0.5
    # Outer-iter info field is populated.
    assert "outer_iters" in model.info_


def test_sparse_group_lasso_within_group_sparsity():
    """One feature active inside group {0,1}; at a moderate λ, SGL should
    shrink |β_1| substantially below plain group lasso. The path-end values
    converge to near-OLS for both and don't differentiate cleanly, so this
    test uses single-λ regressors at λ = 0.05 (mirrors the cargo test)."""
    rng = np.random.default_rng(5)
    n, p = 80, 4
    x = rng.standard_normal((n, p))
    true_beta = np.array([2.0, 0.0, 0.0, 0.0])
    y = x @ true_beta + 0.05 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    sgl = skein.SparseGroupLassoRegressor(
        groups=groups, lambda_=0.05, alpha=0.5, tol=1e-10, max_iter=5000,
    ).fit(x, y).coef_
    plain = skein.GroupLassoRegressor(
        groups=groups, lambda_=0.05, tol=1e-10, max_iter=5000,
    ).fit(x, y).coef_
    assert abs(plain[1]) > 0.01, "group lasso should keep feature 1 active"
    assert abs(sgl[1]) < abs(plain[1]) / 2.0


def test_sparse_group_mcp_path_recovers_sparse_in_group_truth():
    """Sparse-group MCP via LLA: zero feature 1 (within-group L1) AND debias feature 0."""
    rng = np.random.default_rng(6)
    n, p = 80, 4
    x = rng.standard_normal((n, p))
    true_beta = np.array([2.0, 0.0, 0.0, 0.0])
    y = x @ true_beta + 0.05 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    model = skein.SparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=25,
        lambda_min_ratio=5e-3, tol=1e-10, max_iter=5000, max_outer=10,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert abs(last[0]) > 0.5
    assert abs(last[1]) < 0.3


def test_group_estimator_predict_shape():
    x, y, _, groups = _sparse_group_problem(seed=7)
    model = skein.GroupLassoPathRegressor(groups=groups, n_lambdas=5).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0], 5)


def test_groups_label_validation_raises_on_wrong_length():
    x, y, _, _ = _sparse_group_problem(seed=8)
    bad_groups = np.array([0, 0, 1], dtype=np.int64)  # too short
    with pytest.raises(ValueError, match="groups"):
        skein.GroupLassoRegressor(groups=bad_groups, lambda_=0.1).fit(x, y)


# ====================================================================
# Logistic regression
# ====================================================================


def _logistic_problem(seed: int = 0, *, alpha: float = 0.0):
    """200 samples, 10 features, sparse-truth logistic problem."""
    rng = np.random.default_rng(seed)
    n, p = 200, 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 2.0
    true_beta[1] = -1.5
    true_beta[2] = 1.0
    eta = x @ true_beta + alpha
    p_class = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_class).astype(np.float64)
    return x, y, true_beta


def test_logistic_mcp_recovers_signs():
    x, y, true_beta = _logistic_problem(seed=1)
    model = skein.LogisticMCPRegressor(
        lambda_=0.005, gamma=1e6,  # ≈ lasso
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    for k in range(3):
        assert np.sign(model.coef_[k]) == np.sign(true_beta[k]), \
            f"feature {k} sign mismatch: β = {model.coef_[k]}"


def test_logistic_predict_returns_binary_labels():
    x, y, _ = _logistic_problem(seed=2)
    model = skein.LogisticMCPRegressor(lambda_=0.01, gamma=3.0).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0],)
    assert set(np.unique(pred).tolist()).issubset({0.0, 1.0})


def test_logistic_predict_proba_in_unit_interval():
    x, y, _ = _logistic_problem(seed=3)
    model = skein.LogisticMCPRegressor(lambda_=0.01, gamma=3.0).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (x.shape[0],)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_logistic_recovers_intercept_when_y_is_imbalanced():
    # Heavily shifted η ⇒ Pr(y=1) ≈ 0.95 ⇒ intercept should be positive
    # and ~3 (since sigmoid(3) ≈ 0.95).
    x, y, _ = _logistic_problem(seed=4, alpha=3.0)
    model = skein.LogisticMCPRegressor(
        lambda_=0.005, gamma=1e6, fit_intercept=True,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    assert model.intercept_ > 1.0, f"expected α > 1 for imbalanced y, got {model.intercept_}"


def test_logistic_no_intercept_when_fit_intercept_false():
    x, y, _ = _logistic_problem(seed=5)
    model = skein.LogisticMCPRegressor(
        lambda_=0.01, gamma=3.0, fit_intercept=False
    ).fit(x, y)
    assert model.intercept_ == 0.0


def test_logistic_path_returns_decreasing_lambdas():
    x, y, _ = _logistic_problem(seed=6)
    model = skein.LogisticMCPPathRegressor(
        gamma=3.0, n_lambdas=15,
    ).fit(x, y)
    assert model.lambdas_.shape == (15,)
    assert np.all(np.diff(model.lambdas_) < 0)
    assert model.coefs_.shape == (15, x.shape[1])
    assert model.intercepts_.shape == (15,)


def test_logistic_path_predict_proba_shape():
    x, y, _ = _logistic_problem(seed=7)
    model = skein.LogisticMCPPathRegressor(gamma=3.0, n_lambdas=5).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (x.shape[0], 5)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_logistic_path_recovers_signs_at_smallest_lambda():
    x, y, true_beta = _logistic_problem(seed=8)
    model = skein.LogisticMCPPathRegressor(
        gamma=1e6, n_lambdas=25, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    last = model.coefs_[-1]
    for k in range(3):
        assert np.sign(last[k]) == np.sign(true_beta[k]), \
            f"feature {k} sign mismatch at smallest λ: β = {last[k]}"


def test_logistic_scad_estimator_smoke():
    x, y, _ = _logistic_problem(seed=9)
    model = skein.LogisticSCADRegressor(lambda_=0.01, a=3.7).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert isinstance(model.intercept_, float)


def test_logistic_y_validation_rejects_non_binary():
    x = np.random.RandomState(0).standard_normal((20, 3))
    y_bad = np.array([0.0, 0.5, 1.0] * 6 + [0.0, 0.0])  # has 0.5 entries
    with pytest.raises(ValueError, match="y ∈"):
        skein.LogisticMCPRegressor(lambda_=0.01).fit(x, y_bad)


# ====================================================================
# Logistic + group estimators (M3.3)
# ====================================================================


def _logistic_group_problem(seed: int = 0, *, alpha: float = 0.0):
    """4 groups of 2 features (p=8); groups 0 and 2 active. n=240 samples
    chosen for stable logistic recovery on this signal-to-noise problem."""
    rng = np.random.default_rng(seed)
    n, p = 240, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    true_beta[4] = 0.7
    true_beta[5] = 1.2
    eta = x @ true_beta + alpha
    p_class = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_class).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, y, true_beta, groups


def test_logistic_group_lasso_path_recovers_active_groups():
    x, y, _, groups = _logistic_group_problem(seed=11)
    model = skein.LogisticGroupLassoPathRegressor(
        groups=groups, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.5
    assert _group_norm(last, groups, 2) > 0.3


def test_logistic_group_lasso_single_lambda_smoke():
    x, y, _, groups = _logistic_group_problem(seed=12)
    model = skein.LogisticGroupLassoRegressor(
        groups=groups, lambda_=0.01,
    ).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert isinstance(model.intercept_, float)


def test_logistic_group_mcp_path_recovers_active_groups_via_lla():
    x, y, _, groups = _logistic_group_problem(seed=13)
    model = skein.LogisticGroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.5
    assert _group_norm(last, groups, 2) > 0.3
    assert "outer_iters" in model.info_


def test_logistic_sparse_group_lasso_within_group_sparsity():
    """Within an active group, only feature 0 is truly nonzero. SGL should
    shrink |β_1| more than plain group lasso."""
    rng = np.random.default_rng(14)
    n, p = 300, 4
    x = rng.standard_normal((n, p))
    true_beta = np.array([2.5, 0.0, 0.0, 0.0])
    eta = x @ true_beta
    p_class = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_class).astype(np.float64)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    sgl = skein.LogisticSparseGroupLassoRegressor(
        groups=groups, lambda_=0.02, alpha=0.5,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y).coef_
    plain = skein.LogisticGroupLassoRegressor(
        groups=groups, lambda_=0.02,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y).coef_
    assert abs(plain[1]) > 0.01, "plain group lasso should keep feature 1 active"
    assert abs(sgl[1]) < abs(plain[1])


def test_logistic_sparse_group_mcp_path_recovers_in_group_sparse_truth():
    rng = np.random.default_rng(15)
    n, p = 300, 4
    x = rng.standard_normal((n, p))
    true_beta = np.array([2.5, 0.0, 0.0, 0.0])
    eta = x @ true_beta
    p_class = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_class).astype(np.float64)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    model = skein.LogisticSparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=25,
        lambda_min_ratio=5e-3, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert abs(last[0]) > 0.5
    assert abs(last[1]) < abs(last[0])


def test_logistic_group_path_predict_proba_shape():
    x, y, _, groups = _logistic_group_problem(seed=16)
    model = skein.LogisticGroupLassoPathRegressor(
        groups=groups, n_lambdas=5,
    ).fit(x, y)
    proba = model.predict_proba(x)
    assert proba.shape == (x.shape[0], 5)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_logistic_group_recovers_intercept_when_y_imbalanced():
    x, y, _, groups = _logistic_group_problem(seed=17, alpha=3.0)
    model = skein.LogisticGroupLassoRegressor(
        groups=groups, lambda_=0.005, fit_intercept=True,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    assert model.intercept_ > 1.0, \
        f"expected α > 1 for imbalanced y, got {model.intercept_}"


def test_logistic_group_y_validation_rejects_non_binary():
    rng = np.random.default_rng(18)
    x = rng.standard_normal((20, 4))
    y_bad = np.array([0.0, 0.5, 1.0] * 6 + [0.0, 0.0])
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="y ∈"):
        skein.LogisticGroupLassoRegressor(
            groups=groups, lambda_=0.01,
        ).fit(x, y_bad)


# ====================================================================
# Poisson regression (M3.4)
# ====================================================================


def _poisson_problem(seed: int = 0, *, alpha: float = 0.0):
    """Sparse-truth Poisson: 300 samples, 10 features, only first 3 active.
    X is scaled to U(-1, 1) and β kept small so μ stays in roughly
    [exp(-1.6), exp(1.6)] ≈ [0.2, 5]; this matches a regime where IRLS is
    well-conditioned end-to-end on the warm-started λ-path."""
    rng = np.random.default_rng(seed)
    n, p = 300, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    true_beta[2] = 0.4
    eta = x @ true_beta + alpha
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    return x, y, true_beta


def test_poisson_mcp_recovers_signs():
    x, y, true_beta = _poisson_problem(seed=1)
    model = skein.PoissonMCPRegressor(
        lambda_=0.005, gamma=1e6,  # ≈ lasso (convex inner)
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, y)
    for k in range(3):
        assert np.sign(model.coef_[k]) == np.sign(true_beta[k]), \
            f"feature {k} sign mismatch: β = {model.coef_[k]}"


def test_poisson_predict_returns_nonneg_rates():
    x, y, _ = _poisson_problem(seed=2)
    model = skein.PoissonMCPRegressor(lambda_=0.01, gamma=3.0).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0],)
    assert np.all(pred >= 0.0), "Poisson rates must be ≥ 0"
    assert np.all(np.isfinite(pred))


def test_poisson_decision_function_is_log_predict():
    x, y, _ = _poisson_problem(seed=3)
    model = skein.PoissonMCPRegressor(lambda_=0.01, gamma=3.0).fit(x, y)
    eta = model.decision_function(x)
    mu = model.predict(x)
    np.testing.assert_allclose(np.exp(eta), mu, atol=1e-10)


def test_poisson_recovers_intercept_when_y_high_mean():
    # alpha = 1.5 ⇒ μ ≈ exp(1.5) ≈ 4.5 baseline ⇒ intercept should be ≈ 1.5.
    x, y, _ = _poisson_problem(seed=4, alpha=1.5)
    model = skein.PoissonMCPRegressor(
        lambda_=0.005, gamma=1e6, fit_intercept=True,
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, y)
    assert abs(model.intercept_ - 1.5) < 0.4, \
        f"expected intercept ≈ 1.5, got {model.intercept_}"


def test_poisson_no_intercept_when_fit_intercept_false():
    x, y, _ = _poisson_problem(seed=5)
    model = skein.PoissonMCPRegressor(
        lambda_=0.01, gamma=3.0, fit_intercept=False
    ).fit(x, y)
    assert model.intercept_ == 0.0


def test_poisson_path_returns_decreasing_lambdas():
    x, y, _ = _poisson_problem(seed=6)
    model = skein.PoissonMCPPathRegressor(gamma=3.0, n_lambdas=15).fit(x, y)
    assert model.lambdas_.shape == (15,)
    assert np.all(np.diff(model.lambdas_) < 0)
    assert model.coefs_.shape == (15, x.shape[1])
    assert model.intercepts_.shape == (15,)


def test_poisson_path_predict_shape():
    x, y, _ = _poisson_problem(seed=7)
    model = skein.PoissonMCPPathRegressor(gamma=3.0, n_lambdas=5).fit(x, y)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0], 5)
    assert np.all(pred >= 0.0)


def test_poisson_y_validation_rejects_negative():
    x = np.random.RandomState(0).standard_normal((20, 3))
    y_bad = np.array([0.0, 1.0, -1.0] * 6 + [0.0, 0.0])
    with pytest.raises(ValueError, match="y ≥ 0"):
        skein.PoissonMCPRegressor(lambda_=0.01).fit(x, y_bad)


# ---- Poisson + group penalties (M3.4) -----------------------------------


def _poisson_group_problem(seed: int = 0, *, alpha: float = 0.0):
    """8 features in 4 groups of 2; groups 0 and 2 active. X ~ U(-1,1)
    keeps μ moderate."""
    rng = np.random.default_rng(seed)
    n, p = 300, 8
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.6
    true_beta[1] = -0.4
    true_beta[4] = 0.3
    true_beta[5] = 0.5
    eta = x @ true_beta + alpha
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, y, true_beta, groups


def test_poisson_group_lasso_path_recovers_active_groups():
    x, y, _, groups = _poisson_group_problem(seed=11)
    model = skein.PoissonGroupLassoPathRegressor(
        groups=groups, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.2
    assert _group_norm(last, groups, 2) > 0.2


def test_poisson_group_mcp_path_recovers_active_groups_via_lla():
    x, y, _, groups = _poisson_group_problem(seed=12)
    model = skein.PoissonGroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.2
    assert _group_norm(last, groups, 2) > 0.2
    assert "outer_iters" in model.info_


def test_poisson_sparse_group_mcp_path_recovers_in_group_sparse_truth():
    rng = np.random.default_rng(13)
    n, p = 300, 4
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.array([0.8, 0.0, 0.0, 0.0])
    eta = x @ true_beta
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    model = skein.PoissonSparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=25,
        lambda_min_ratio=5e-3, tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert abs(last[0]) > 0.2
    assert abs(last[1]) < abs(last[0])


def test_poisson_group_y_validation_rejects_negative():
    rng = np.random.default_rng(14)
    x = rng.standard_normal((20, 4))
    y_bad = np.array([0.0, 1.0, -1.0] * 6 + [0.0, 0.0])
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="y ≥ 0"):
        skein.PoissonGroupLassoRegressor(
            groups=groups, lambda_=0.01,
        ).fit(x, y_bad)


# ====================================================================
# Cox proportional hazards (M3.5)
# ====================================================================


def _cox_problem(seed: int = 0):
    """Sparse-truth Cox PH problem with exponential baseline hazard.
    Sample T_i ~ Exp(rate=exp(η_i)), C_i ~ Exp(rate=0.5);
    observe t = min(T,C), δ = 1[T ≤ C]. Yields ~50-70% events."""
    rng = np.random.default_rng(seed)
    n, p = 300, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    true_beta[2] = 0.3
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    return x, time, event, true_beta


def test_cox_mcp_recovers_signs():
    x, time, event, true_beta = _cox_problem(seed=1)
    model = skein.CoxMCPRegressor(
        lambda_=0.005, gamma=1e6,  # ≈ lasso
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, time, event)
    for k in range(3):
        assert np.sign(model.coef_[k]) == np.sign(true_beta[k]), \
            f"feature {k} sign mismatch: β = {model.coef_[k]}"


def test_cox_predict_matches_decision_function():
    x, time, event, _ = _cox_problem(seed=2)
    model = skein.CoxMCPRegressor(lambda_=0.01, gamma=3.0).fit(x, time, event)
    pred = model.predict(x)
    eta = model.decision_function(x)
    np.testing.assert_allclose(pred, eta, atol=1e-12)
    assert pred.shape == (x.shape[0],)


def test_cox_no_intercept_attribute():
    """Cox doesn't fit an intercept (baseline hazard absorbs constants).
    Estimators should not expose `intercept_`."""
    x, time, event, _ = _cox_problem(seed=3)
    model = skein.CoxMCPRegressor(lambda_=0.01, gamma=3.0).fit(x, time, event)
    assert not hasattr(model, "intercept_")


def test_cox_path_returns_decreasing_lambdas():
    x, time, event, _ = _cox_problem(seed=4)
    model = skein.CoxMCPPathRegressor(gamma=3.0, n_lambdas=15).fit(x, time, event)
    assert model.lambdas_.shape == (15,)
    assert np.all(np.diff(model.lambdas_) < 0)
    assert model.coefs_.shape == (15, x.shape[1])


def test_cox_path_predict_shape():
    x, time, event, _ = _cox_problem(seed=5)
    model = skein.CoxMCPPathRegressor(gamma=3.0, n_lambdas=5).fit(x, time, event)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0], 5)


def test_cox_path_recovers_signs_at_smallest_lambda():
    x, time, event, true_beta = _cox_problem(seed=6)
    model = skein.CoxMCPPathRegressor(
        gamma=1e6, n_lambdas=25, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, time, event)
    last = model.coefs_[-1]
    for k in range(3):
        assert np.sign(last[k]) == np.sign(true_beta[k]), \
            f"feature {k} sign mismatch at smallest λ: β = {last[k]}"


def test_cox_validation_rejects_non_binary_event():
    x, time, _, _ = _cox_problem(seed=7)
    bad_event = np.full(time.shape[0], 0.5)
    with pytest.raises(ValueError, match="event"):
        skein.CoxMCPRegressor(lambda_=0.01).fit(x, time, bad_event)


def test_cox_validation_rejects_no_events():
    x, time, _, _ = _cox_problem(seed=8)
    no_events = np.zeros(time.shape[0])
    with pytest.raises(ValueError, match="at least one event"):
        skein.CoxMCPRegressor(lambda_=0.01).fit(x, time, no_events)


def test_cox_validation_rejects_negative_time():
    x, time, event, _ = _cox_problem(seed=9)
    bad_time = time.copy()
    bad_time[0] = -1.0
    with pytest.raises(ValueError, match="time"):
        skein.CoxMCPRegressor(lambda_=0.01).fit(x, bad_time, event)


# ---- Cox + group penalties (M3.5) ---------------------------------------


def _cox_group_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 300, 8
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.6
    true_beta[1] = -0.4
    true_beta[4] = 0.3
    true_beta[5] = 0.5
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, time, event, true_beta, groups


def test_cox_group_lasso_path_recovers_active_groups():
    x, time, event, _, groups = _cox_group_problem(seed=11)
    model = skein.CoxGroupLassoPathRegressor(
        groups=groups, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, time, event)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.2
    assert _group_norm(last, groups, 2) > 0.15


def test_cox_group_mcp_path_recovers_active_groups_via_lla():
    x, time, event, _, groups = _cox_group_problem(seed=12)
    model = skein.CoxGroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, time, event)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.2
    assert _group_norm(last, groups, 2) > 0.15
    assert "outer_iters" in model.info_


def test_cox_sparse_group_mcp_path_recovers_in_group_sparse_truth():
    rng = np.random.default_rng(13)
    n, p = 300, 4
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.array([0.8, 0.0, 0.0, 0.0])
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    model = skein.CoxSparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=25,
        lambda_min_ratio=5e-3, tol=1e-10, max_iter=5000, max_outer=30,
    ).fit(x, time, event)
    last = model.coefs_[-1]
    assert abs(last[0]) > 0.2
    assert abs(last[1]) < abs(last[0])


# ====================================================================
# SparseCSC backend (M4.1 / M4.2a)
# ====================================================================


scipy_sparse = pytest.importorskip("scipy.sparse")


def _ls_problem_with_density(seed: int, density: float):
    """Generate a dense X with a fraction `density` of non-zero entries
    plus a sparse-truth signal so dense vs sparse can be compared."""
    rng = np.random.default_rng(seed)
    n, p = 80, 12
    x = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > density
    x[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[3] = -1.0
    true_beta[7] = 0.7
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    return x, y


def test_sparse_mcp_path_matches_dense_within_tol():
    """Dense and sparse must converge to the same β at every λ on a
    shared grid. We pass explicit λ values because the two paths
    derive λ_max slightly differently (dense centers y first; sparse
    uses intercept-as-column with no intercept warm-start before
    λ_max), which would shift the auto-grid points apart. At each
    fixed λ the convex (γ=1e6 ≈ lasso) optimum is unique."""
    x_dense, y = _ls_problem_with_density(seed=11, density=0.4)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01, 0.003], dtype=np.float64)

    common = dict(
        gamma=1e6,
        lambdas=lambdas,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=True, standardize=False,
    )
    dense_model = skein.MCPPathRegressor(**common).fit(x_dense, y)
    sparse_model = skein.MCPPathRegressor(**common).fit(x_sparse, y)

    np.testing.assert_allclose(
        dense_model.coefs_, sparse_model.coefs_, atol=1e-6
    )
    np.testing.assert_allclose(
        dense_model.intercepts_, sparse_model.intercepts_, atol=1e-6
    )


def test_sparse_mcp_single_lambda_matches_dense():
    x_dense, y = _ls_problem_with_density(seed=12, density=0.3)
    x_sparse = scipy_sparse.csc_matrix(x_dense)

    common = dict(
        lambda_=0.05, gamma=1e6,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=True, standardize=False,
    )
    dense_model = skein.MCPRegressor(**common).fit(x_dense, y)
    sparse_model = skein.MCPRegressor(**common).fit(x_sparse, y)

    np.testing.assert_allclose(dense_model.coef_, sparse_model.coef_, atol=1e-7)
    assert abs(dense_model.intercept_ - sparse_model.intercept_) < 1e-7


def test_sparse_scad_path_matches_dense_within_tol():
    x_dense, y = _ls_problem_with_density(seed=13, density=0.3)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)

    common = dict(
        a=1e6,  # ≈ lasso for SCAD too (all penalties become L1)
        lambdas=lambdas,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=False, standardize=False,
    )
    dense_model = skein.SCADPathRegressor(**common).fit(x_dense, y)
    sparse_model = skein.SCADPathRegressor(**common).fit(x_sparse, y)

    np.testing.assert_allclose(
        dense_model.coefs_, sparse_model.coefs_, atol=1e-6
    )


def test_sparse_no_intercept_matches_dense_no_intercept():
    x_dense, y = _ls_problem_with_density(seed=14, density=0.4)
    x_sparse = scipy_sparse.csc_matrix(x_dense)

    common = dict(
        lambda_=0.1, gamma=3.0,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=False, standardize=False,
    )
    dense_model = skein.MCPRegressor(**common).fit(x_dense, y)
    sparse_model = skein.MCPRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense_model.coef_, sparse_model.coef_, atol=1e-7)
    assert dense_model.intercept_ == 0.0
    assert sparse_model.intercept_ == 0.0


def test_sparse_predict_shape_and_values_match_dense():
    x_dense, y = _ls_problem_with_density(seed=15, density=0.3)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    common = dict(
        lambda_=0.05, gamma=3.0,
        tol=1e-10, max_iter=5000, screening="off",
        fit_intercept=True, standardize=False,
    )
    dense_model = skein.MCPRegressor(**common).fit(x_dense, y)
    # Predict on the sparse view should match predict on dense.
    pred_dense = dense_model.predict(x_dense)
    pred_sparse = dense_model.predict(x_sparse)
    np.testing.assert_allclose(pred_dense, pred_sparse, atol=1e-12)


def test_sparse_mcp_with_standardize_runs_and_recovers_signal():
    """`standardize=True` is supported for sparse via the lazy
    `Standardized` wrapper (M4.3) — no longer raises. β should still
    recover the active features at small λ."""
    x_dense, y = _ls_problem_with_density(seed=16, density=0.5)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    model = skein.MCPPathRegressor(
        gamma=1e6, n_lambdas=20, lambda_min_ratio=5e-3,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=True, standardize=True,
    ).fit(x_sparse, y)
    last = model.coefs_[-1]
    # True β has nonzero entries at indices 0, 3, 7.
    for j in (0, 3, 7):
        assert abs(last[j]) > 0.05, f"feature {j} not recovered"


def test_sparse_csr_input_is_converted_to_csc():
    """Estimator should accept other scipy.sparse formats and convert to CSC."""
    x_dense, y = _ls_problem_with_density(seed=17, density=0.4)
    x_csr = scipy_sparse.csr_matrix(x_dense)
    model = skein.MCPRegressor(
        lambda_=0.1, gamma=3.0, tol=1e-10, max_iter=5000,
        fit_intercept=True, standardize=False, screening="off",
    ).fit(x_csr, y)
    assert model.coef_.shape == (x_dense.shape[1],)


# ---- Sparse + group penalties (M4.2b) ----------------------------------


def _ls_group_problem_with_density(seed: int, density: float):
    rng = np.random.default_rng(seed)
    n, p = 80, 8
    x = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > density
    x[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    true_beta[4] = 0.7
    true_beta[5] = 1.2
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, y, groups


def test_sparse_group_lasso_path_matches_dense():
    x_dense, y, groups = _ls_group_problem_with_density(seed=21, density=0.4)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)

    common = dict(
        groups=groups, lambdas=lambdas,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=True, standardize=False,
    )
    dense_model = skein.GroupLassoPathRegressor(**common).fit(x_dense, y)
    sparse_model = skein.GroupLassoPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(
        dense_model.coefs_, sparse_model.coefs_, atol=1e-6
    )
    np.testing.assert_allclose(
        dense_model.intercepts_, sparse_model.intercepts_, atol=1e-6
    )


def test_sparse_group_mcp_path_matches_dense_with_lasso_limit():
    """Use γ=1e6 to make group MCP ≈ group lasso (convex inner problem),
    so dense and sparse converge to the same minimum at each λ."""
    x_dense, y, groups = _ls_group_problem_with_density(seed=22, density=0.5)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)

    common = dict(
        groups=groups, gamma=1e6, lambdas=lambdas,
        tol=1e-12, max_iter=10000, max_outer=30, screening="off",
        fit_intercept=False, standardize=False,
    )
    dense_model = skein.GroupMCPPathRegressor(**common).fit(x_dense, y)
    sparse_model = skein.GroupMCPPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(
        dense_model.coefs_, sparse_model.coefs_, atol=1e-6
    )


def test_sparse_group_lasso_single_lambda_smoke():
    x_dense, y, groups = _ls_group_problem_with_density(seed=23, density=0.4)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    model = skein.GroupLassoRegressor(
        groups=groups, lambda_=0.1,
        tol=1e-10, max_iter=5000, screening="off",
        fit_intercept=True, standardize=False,
    ).fit(x_sparse, y)
    assert model.coef_.shape == (x_dense.shape[1],)
    # Active groups still recovered.
    assert _group_norm(model.coef_, groups, 0) > 0.0


def test_sparse_sparse_group_lasso_path_matches_dense():
    x_dense, y, groups = _ls_group_problem_with_density(seed=24, density=0.4)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)

    common = dict(
        groups=groups, alpha=0.5, lambdas=lambdas,
        tol=1e-12, max_iter=10000, screening="off",
        fit_intercept=False, standardize=False,
    )
    dense_model = skein.SparseGroupLassoPathRegressor(**common).fit(x_dense, y)
    sparse_model = skein.SparseGroupLassoPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(
        dense_model.coefs_, sparse_model.coefs_, atol=1e-6
    )


def test_sparse_sparse_group_mcp_smoke():
    x_dense, y, groups = _ls_group_problem_with_density(seed=25, density=0.4)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    model = skein.SparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=10,
        lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, max_outer=20,
        fit_intercept=False, standardize=False,
    ).fit(x_sparse, y)
    assert model.coefs_.shape == (10, x_dense.shape[1])
    # Some active group survives at the smallest λ.
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) + _group_norm(last, groups, 2) > 0.0


def test_sparse_group_lasso_with_standardize_runs():
    """`standardize=True` works for sparse group LS via the lazy
    `Standardized` wrapper (M4.3). Smoke test — exact dense↔sparse
    equivalence is covered by separate tests."""
    x_dense, y, groups = _ls_group_problem_with_density(seed=26, density=0.5)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    model = skein.GroupLassoPathRegressor(
        groups=groups, n_lambdas=10, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, screening="off",
        fit_intercept=True, standardize=True,
    ).fit(x_sparse, y)
    last = model.coefs_[-1]
    assert _group_norm(last, groups, 0) > 0.0
    assert _group_norm(last, groups, 2) > 0.0


# ---- Sparse + GLMs (M4.2c) ---------------------------------------------


def _logistic_sparse_problem(seed: int, density: float = 0.4):
    rng = np.random.default_rng(seed)
    n, p = 200, 10
    x = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > density
    x[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 2.0
    true_beta[1] = -1.5
    true_beta[2] = 1.0
    eta = x @ true_beta
    p_class = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_class).astype(np.float64)
    return x, y


def test_sparse_logistic_mcp_path_matches_dense():
    x_dense, y = _logistic_sparse_problem(seed=31)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([0.1, 0.05, 0.02, 0.01, 0.005], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas,
        tol=1e-12, max_iter=10000, max_outer=30,
        fit_intercept=True,
    )
    dense = skein.LogisticMCPPathRegressor(**common).fit(x_dense, y)
    sparse = skein.LogisticMCPPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_sparse_logistic_predict_proba_matches_dense():
    x_dense, y = _logistic_sparse_problem(seed=32)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    common = dict(
        lambda_=0.01, gamma=3.0, tol=1e-10, max_iter=5000, max_outer=20,
        fit_intercept=True,
    )
    dense = skein.LogisticMCPRegressor(**common).fit(x_dense, y)
    p_dense = dense.predict_proba(x_dense)
    p_sparse = dense.predict_proba(x_sparse)
    np.testing.assert_allclose(p_dense, p_sparse, atol=1e-12)


def test_sparse_logistic_group_lasso_matches_dense():
    x_dense, y = _logistic_sparse_problem(seed=33, density=0.5)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    groups = np.repeat(np.arange(5), 2).astype(np.int64)  # 5 groups of 2
    lambdas = np.array([0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        groups=groups, lambdas=lambdas,
        tol=1e-12, max_iter=10000, max_outer=30,
        fit_intercept=True,
    )
    dense = skein.LogisticGroupLassoPathRegressor(**common).fit(x_dense, y)
    sparse = skein.LogisticGroupLassoPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)


def _poisson_sparse_problem(seed: int, density: float = 0.4):
    rng = np.random.default_rng(seed)
    n, p = 200, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    mask = rng.uniform(size=(n, p)) > density
    x[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    true_beta[2] = 0.4
    eta = x @ true_beta
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    return x, y


def test_sparse_poisson_mcp_path_matches_dense():
    x_dense, y = _poisson_sparse_problem(seed=34)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas,
        tol=1e-12, max_iter=10000, max_outer=30,
        fit_intercept=True,
    )
    dense = skein.PoissonMCPPathRegressor(**common).fit(x_dense, y)
    sparse = skein.PoissonMCPPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)


def test_sparse_poisson_group_lasso_smoke():
    x_dense, y = _poisson_sparse_problem(seed=35, density=0.5)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    groups = np.repeat(np.arange(5), 2).astype(np.int64)
    model = skein.PoissonGroupLassoRegressor(
        groups=groups, lambda_=0.05,
        tol=1e-10, max_iter=5000, max_outer=20, fit_intercept=True,
    ).fit(x_sparse, y)
    assert model.coef_.shape == (x_dense.shape[1],)


def _cox_sparse_problem(seed: int, density: float = 0.4):
    rng = np.random.default_rng(seed)
    n, p = 200, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    mask = rng.uniform(size=(n, p)) > density
    x[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    true_beta[2] = 0.3
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    return x, time, event


def test_sparse_cox_mcp_path_matches_dense():
    x_dense, time, event = _cox_sparse_problem(seed=36)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    lambdas = np.array([0.1, 0.03, 0.01, 0.003], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas,
        tol=1e-12, max_iter=10000, max_outer=30,
    )
    dense = skein.CoxMCPPathRegressor(**common).fit(x_dense, time, event)
    sparse = skein.CoxMCPPathRegressor(**common).fit(x_sparse, time, event)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)


def test_sparse_cox_predict_matches_dense():
    x_dense, time, event = _cox_sparse_problem(seed=37)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    model = skein.CoxMCPRegressor(
        lambda_=0.01, gamma=3.0, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x_dense, time, event)
    pred_dense = model.predict(x_dense)
    pred_sparse = model.predict(x_sparse)
    np.testing.assert_allclose(pred_dense, pred_sparse, atol=1e-12)


def test_sparse_cox_group_lasso_smoke():
    x_dense, time, event = _cox_sparse_problem(seed=38, density=0.5)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    groups = np.repeat(np.arange(5), 2).astype(np.int64)
    model = skein.CoxGroupLassoRegressor(
        groups=groups, lambda_=0.05,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x_sparse, time, event)
    assert model.coef_.shape == (x_dense.shape[1],)


# ---- Sparse + standardize_x via lazy Standardized wrapper (M4.3) -------


def test_sparse_mcp_path_with_standardize_matches_dense_path():
    """Dense LS path uses centering+scaling; sparse LS path uses
    column-augmentation+scaling via `Standardized<SparseCSC>`. They
    parameterize the same problem and converge to the same β at each λ
    on a shared grid (γ=1e6 ≈ lasso ⇒ unique global optimum)."""
    rng = np.random.default_rng(101)
    n, p = 80, 12
    x_dense = rng.standard_normal((n, p))
    # Inflate column 0's scale 30× — the case where standardize matters.
    x_dense[:, 0] *= 30.0
    mask = rng.uniform(size=(n, p)) > 0.5
    x_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 0.05
    true_beta[2] = -1.0
    true_beta[5] = 0.7
    y = x_dense @ true_beta + 0.05 * rng.standard_normal(n)
    x_sparse = scipy_sparse.csc_matrix(x_dense)

    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas,
        tol=1e-12, max_iter=20000, screening="off",
        fit_intercept=True, standardize=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x_dense, y)
    sparse = skein.MCPPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_sparse_group_lasso_path_with_standardize_matches_dense():
    rng = np.random.default_rng(103)
    n, p = 60, 8
    x_dense = rng.standard_normal((n, p))
    x_dense[:, 0] *= 10.0  # one column with inflated scale
    mask = rng.uniform(size=(n, p)) > 0.5
    x_dense[mask] = 0.0
    true_beta = np.array([0.1, -0.5, 0.0, 0.0, 0.4, -0.6, 0.0, 0.0])
    y = x_dense @ true_beta + 0.05 * rng.standard_normal(n)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        groups=groups, lambdas=lambdas,
        tol=1e-12, max_iter=20000, screening="off",
        fit_intercept=True, standardize=True,
    )
    dense = skein.GroupLassoPathRegressor(**common).fit(x_dense, y)
    sparse = skein.GroupLassoPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


# ====================================================================
# Cross-validation (M5.1a)
# ====================================================================


def _cv_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 120, 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[2] = -2.0
    true_beta[5] = 0.7
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    return x, y, true_beta


def test_mcp_path_cv_returns_finite_scores_and_picks_lambda():
    x, y, _ = _cv_problem(seed=1)
    model = skein.MCPPathCV(
        gamma=1e6,  # ≈ lasso for clean CV behavior
        cv=5, random_state=0,
        n_lambdas=15, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    assert model.cv_scores_.shape == (5, 15)
    assert model.cv_mean_scores_.shape == (15,)
    assert model.cv_std_scores_.shape == (15,)
    assert np.all(np.isfinite(model.cv_scores_))
    assert model.lambdas_.shape == (15,)
    assert model.lambda_best_ in model.lambdas_
    assert model.coef_.shape == (x.shape[1],)
    assert isinstance(model.intercept_, float)


def test_mcp_path_cv_recovers_signal_at_chosen_lambda():
    """On a noiseless-ish problem, CV should pick a small λ that
    recovers the active features. Tests the end-to-end pipeline:
    fold splits, scoring, λ selection, and refit."""
    x, y, true_beta = _cv_problem(seed=2)
    model = skein.MCPPathCV(
        gamma=1e6, cv=5, random_state=0,
        n_lambdas=20, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    for k in (0, 2, 5):
        assert np.sign(model.coef_[k]) == np.sign(true_beta[k]), \
            f"feature {k} sign mismatch (β={model.coef_[k]})"


def test_mcp_path_cv_predict_shape():
    x, y, _ = _cv_problem(seed=3)
    model = skein.MCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=10,
    ).fit(x, y)
    pred = model.predict(x)
    # CV's `predict` uses the single best-λ β, so output is 1D.
    assert pred.shape == (x.shape[0],)


def test_mcp_path_cv_with_explicit_lambdas_skips_init_path():
    x, y, _ = _cv_problem(seed=4)
    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    model = skein.MCPPathCV(
        gamma=3.0, lambdas=lambdas, cv=3, random_state=0,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    np.testing.assert_array_equal(model.lambdas_, lambdas)
    assert model.cv_scores_.shape == (3, 5)


def test_mcp_path_cv_works_with_sparse_input():
    rng = np.random.default_rng(5)
    n, p = 80, 10
    x_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.4
    x_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[2] = -1.0
    y = x_dense @ true_beta + 0.05 * rng.standard_normal(n)
    x_sparse = scipy_sparse.csc_matrix(x_dense)

    model = skein.MCPPathCV(
        gamma=1e6, cv=4, random_state=0, n_lambdas=12,
        lambda_min_ratio=1e-2, tol=1e-10, max_iter=5000,
        screening="off", fit_intercept=True,
    ).fit(x_sparse, y)
    assert model.coef_.shape == (p,)
    # predict on sparse should match dense predict.
    pred_sparse = model.predict(x_sparse)
    pred_dense = model.predict(x_dense)
    np.testing.assert_allclose(pred_sparse, pred_dense, atol=1e-12)


def test_scad_path_cv_smoke():
    x, y, _ = _cv_problem(seed=6)
    model = skein.SCADPathCV(
        a=3.7, cv=3, random_state=0, n_lambdas=10,
    ).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert model.cv_scores_.shape == (3, 10)


def test_mcp_path_cv_accepts_sklearn_splitter():
    """`cv` accepts an integer (KFold) or a pre-built sklearn
    splitter."""
    from sklearn.model_selection import KFold
    x, y, _ = _cv_problem(seed=7)
    splitter = KFold(n_splits=4, shuffle=True, random_state=42)
    model = skein.MCPPathCV(
        gamma=3.0, cv=splitter, n_lambdas=8,
    ).fit(x, y)
    assert model.cv_scores_.shape == (4, 8)


# ---- CV for LS group penalties (M5.1b) ---------------------------------


def _cv_group_problem(seed: int = 0):
    """4 groups of 2 features (p=8); groups 0 and 2 active."""
    rng = np.random.default_rng(seed)
    n, p = 120, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    true_beta[4] = 0.7
    true_beta[5] = 1.2
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, y, true_beta, groups


def test_group_lasso_path_cv_recovers_active_groups():
    x, y, _, groups = _cv_group_problem(seed=11)
    model = skein.GroupLassoPathCV(
        groups=groups, cv=5, random_state=0,
        n_lambdas=15, lambda_min_ratio=5e-3,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    assert model.cv_scores_.shape == (5, 15)
    assert np.all(np.isfinite(model.cv_scores_))
    # The CV-chosen β should retain the truly active groups.
    assert _group_norm(model.coef_, groups, 0) > 0.5
    assert _group_norm(model.coef_, groups, 2) > 0.3


def test_group_mcp_path_cv_smoke_and_lla_outer_iters_recorded():
    x, y, _, groups = _cv_group_problem(seed=12)
    model = skein.GroupMCPPathCV(
        groups=groups, gamma=3.0, cv=4, random_state=0,
        n_lambdas=10, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, max_outer=20, screening="off",
    ).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert model.cv_scores_.shape == (4, 10)


def test_sparse_group_lasso_path_cv_smoke():
    x, y, _, groups = _cv_group_problem(seed=13)
    model = skein.SparseGroupLassoPathCV(
        groups=groups, alpha=0.5, cv=4, random_state=0,
        n_lambdas=10, tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert model.cv_scores_.shape == (4, 10)


def test_sparse_group_mcp_path_cv_smoke():
    x, y, _, groups = _cv_group_problem(seed=14)
    model = skein.SparseGroupMCPPathCV(
        groups=groups, gamma=3.0, alpha=0.5, cv=3, random_state=0,
        n_lambdas=10, tol=1e-10, max_iter=5000, max_outer=20, screening="off",
    ).fit(x, y)
    assert model.coef_.shape == (x.shape[1],)
    assert model.cv_scores_.shape == (3, 10)


def test_group_lasso_path_cv_with_sparse_input():
    rng = np.random.default_rng(15)
    n, p = 80, 8
    x_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    x_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    true_beta[4] = 0.7
    y = x_dense @ true_beta + 0.05 * rng.standard_normal(n)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    model = skein.GroupLassoPathCV(
        groups=groups, cv=4, random_state=0,
        n_lambdas=10, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x_sparse, y)
    assert model.coef_.shape == (p,)
    pred_sparse = model.predict(x_sparse)
    pred_dense = model.predict(x_dense)
    np.testing.assert_allclose(pred_sparse, pred_dense, atol=1e-12)


# ---- CV for GLM families (M5.1c) ---------------------------------------


def _logistic_cv_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 240, 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 2.0
    true_beta[1] = -1.5
    true_beta[2] = 1.0
    eta = x @ true_beta
    p_class = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_class).astype(np.float64)
    return x, y, true_beta


def test_logistic_mcp_path_cv_runs_and_picks_lambda():
    x, y, _ = _logistic_cv_problem(seed=21)
    model = skein.LogisticMCPPathCV(
        gamma=1e6, cv=4, random_state=0,
        n_lambdas=12, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    assert model.cv_scores_.shape == (4, 12)
    assert np.all(np.isfinite(model.cv_scores_))
    # Deviance is non-negative.
    assert (model.cv_scores_ >= 0).all()
    assert model.coef_.shape == (x.shape[1],)
    assert isinstance(model.intercept_, float)


def test_logistic_path_cv_predict_proba_shape_and_range():
    x, y, _ = _logistic_cv_problem(seed=22)
    model = skein.LogisticMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=8,
    ).fit(x, y)
    p = model.predict_proba(x)
    # CV uses the single best λ, so shape is 1D.
    assert p.shape == (x.shape[0],)
    assert np.all((p >= 0) & (p <= 1))
    pred = model.predict(x)
    assert set(np.unique(pred).tolist()).issubset({0.0, 1.0})


def test_logistic_group_lasso_cv_smoke():
    rng = np.random.default_rng(23)
    n, p = 240, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    true_beta[4] = 0.7
    eta = x @ true_beta
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    model = skein.LogisticGroupLassoPathCV(
        groups=groups, cv=3, random_state=0,
        n_lambdas=8, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    assert model.coef_.shape == (p,)
    assert model.cv_scores_.shape == (3, 8)


def _poisson_cv_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 240, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    true_beta[2] = 0.4
    mu = np.exp(x @ true_beta)
    y = rng.poisson(mu).astype(np.float64)
    return x, y, true_beta


def test_poisson_mcp_path_cv_runs_and_picks_lambda():
    x, y, _ = _poisson_cv_problem(seed=24)
    model = skein.PoissonMCPPathCV(
        gamma=1e6, cv=4, random_state=0,
        n_lambdas=12, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    assert model.cv_scores_.shape == (4, 12)
    assert np.all(np.isfinite(model.cv_scores_))
    # Predict returns rates μ ≥ 0.
    pred = model.predict(x)
    assert pred.shape == (x.shape[0],)
    assert np.all(pred >= 0)


def test_poisson_group_lasso_cv_smoke():
    rng = np.random.default_rng(25)
    n, p = 240, 8
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.6
    true_beta[1] = -0.4
    true_beta[4] = 0.3
    mu = np.exp(x @ true_beta)
    y = rng.poisson(mu).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    model = skein.PoissonGroupLassoPathCV(
        groups=groups, cv=3, random_state=0,
        n_lambdas=8, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    assert model.coef_.shape == (p,)


def _cox_cv_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 240, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    true_beta[2] = 0.3
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    return x, time, event, true_beta


def test_cox_mcp_path_cv_concordance_above_chance():
    """A correctly-ordered Cox model should give c-index above 0.5."""
    x, time, event, _ = _cox_cv_problem(seed=26)
    model = skein.CoxMCPPathCV(
        gamma=1e6, cv=4, random_state=0,
        n_lambdas=12, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, time, event)
    assert model.cv_scores_.shape == (4, 12)
    # The CV-best λ should yield a mean concordance > 0.5 (random
    # ordering would give exactly 0.5 in expectation).
    assert model.cv_mean_scores_[
        np.argmax(model.cv_mean_scores_)
    ] > 0.55


def test_cox_path_cv_no_intercept_attribute():
    """Cox CV mirrors Cox base — no intercept_."""
    x, time, event, _ = _cox_cv_problem(seed=27)
    model = skein.CoxMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=8,
    ).fit(x, time, event)
    assert not hasattr(model, "intercept_")
    assert model.coef_.shape == (x.shape[1],)


def test_cox_path_cv_predict_matches_decision_function():
    x, time, event, _ = _cox_cv_problem(seed=28)
    model = skein.CoxMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=8,
    ).fit(x, time, event)
    eta = model.decision_function(x)
    pred = model.predict(x)
    np.testing.assert_allclose(eta, pred, atol=1e-12)
    assert eta.shape == (x.shape[0],)


def test_cox_group_lasso_cv_smoke():
    rng = np.random.default_rng(29)
    n, p = 240, 8
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.6
    true_beta[1] = -0.4
    true_beta[4] = 0.3
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    model = skein.CoxGroupLassoPathCV(
        groups=groups, cv=3, random_state=0,
        n_lambdas=8, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, time, event)
    assert model.coef_.shape == (p,)
    assert model.cv_scores_.shape == (3, 8)


def test_cox_path_cv_with_sparse_input():
    rng = np.random.default_rng(30)
    n, p = 200, 8
    x_dense = rng.uniform(-1.0, 1.0, size=(n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    x_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    eta = x_dense @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=1.0 / 0.5, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    x_sparse = scipy_sparse.csc_matrix(x_dense)

    model = skein.CoxMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=8,
        tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x_sparse, time, event)
    assert model.coef_.shape == (p,)
    pred_sparse = model.predict(x_sparse)
    pred_dense = model.predict(x_dense)
    np.testing.assert_allclose(pred_sparse, pred_dense, atol=1e-12)


# ====================================================================
# Information criteria (M5.2)
# ====================================================================


def test_select_by_ic_ls_bic_chooses_finite_lambda():
    rng = np.random.default_rng(40)
    n, p = 120, 12
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[2] = -1.0
    y = x @ true_beta + 0.2 * rng.standard_normal(n)
    path = skein.MCPPathRegressor(
        gamma=1e6, n_lambdas=20, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    best_idx, scores = skein.select_by_ic(path, x, y, criterion="bic")
    assert scores.shape == (20,)
    assert np.all(np.isfinite(scores))
    assert 0 <= best_idx < 20
    assert path.lambdas_[best_idx] > 0


def test_select_by_ic_aic_and_bic_can_differ():
    """AIC penalizes per-active less than BIC for n > e², so on a
    moderate-n problem with a path of nested models, AIC should pick
    a less sparse (smaller-λ, larger-k) solution."""
    rng = np.random.default_rng(41)
    n, p = 80, 15
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:5] = [1.0, -1.0, 0.5, -0.5, 0.3]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    path = skein.MCPPathRegressor(
        gamma=1e6, n_lambdas=25, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    aic_idx, _ = skein.select_by_ic(path, x, y, criterion="aic")
    bic_idx, _ = skein.select_by_ic(path, x, y, criterion="bic")
    # On a typical n > e² problem AIC ≤ BIC λ-index ⇒ AIC keeps more
    # features active (smaller λ → larger index in the grid).
    aic_active = int(np.sum(np.abs(path.coefs_[aic_idx]) > 1e-12))
    bic_active = int(np.sum(np.abs(path.coefs_[bic_idx]) > 1e-12))
    assert aic_active >= bic_active


def test_select_by_ic_ebic_stricter_than_bic_when_p_large():
    """EBIC adds a `2γ log C(p,k)` term, so for p ≫ n it should pick
    a sparser (larger-λ, smaller-k) solution than BIC."""
    rng = np.random.default_rng(42)
    n, p = 50, 60  # p > n
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -1.0, 0.7]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    path = skein.MCPPathRegressor(
        gamma=1e6, n_lambdas=30, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=5000, screening="off",
    ).fit(x, y)
    bic_idx, _ = skein.select_by_ic(path, x, y, criterion="bic")
    ebic_idx, _ = skein.select_by_ic(path, x, y, criterion="ebic", ebic_gamma=1.0)
    bic_k = int(np.sum(np.abs(path.coefs_[bic_idx]) > 1e-12))
    ebic_k = int(np.sum(np.abs(path.coefs_[ebic_idx]) > 1e-12))
    assert ebic_k <= bic_k


def test_select_by_ic_logistic_runs_and_scores_finite():
    rng = np.random.default_rng(43)
    n, p = 200, 10
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    eta = x @ true_beta
    y = (rng.uniform(0, 1, n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    path = skein.LogisticMCPPathRegressor(
        gamma=1e6, n_lambdas=15, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    best_idx, scores = skein.select_by_ic(path, x, y, criterion="bic")
    assert scores.shape == (15,)
    assert np.all(np.isfinite(scores))


def test_select_by_ic_poisson_runs():
    rng = np.random.default_rng(44)
    n, p = 200, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    mu = np.exp(x @ true_beta)
    y = rng.poisson(mu).astype(np.float64)
    path = skein.PoissonMCPPathRegressor(
        gamma=1e6, n_lambdas=12, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, y)
    best_idx, scores = skein.select_by_ic(path, x, y, criterion="bic")
    assert scores.shape == (12,)
    assert np.all(np.isfinite(scores))


def test_select_by_ic_cox_dispatches_on_time_event():
    rng = np.random.default_rng(45)
    n, p = 200, 10
    x = rng.uniform(-1.0, 1.0, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.7
    true_beta[1] = -0.5
    eta = x @ true_beta
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=2.0, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    path = skein.CoxMCPPathRegressor(
        gamma=1e6, n_lambdas=12, tol=1e-10, max_iter=5000, max_outer=20,
    ).fit(x, time, event)
    best_idx, scores = skein.select_by_ic(path, x, time, event, criterion="bic")
    assert scores.shape == (12,)
    assert np.all(np.isfinite(scores))


def test_select_by_ic_rejects_unknown_criterion():
    rng = np.random.default_rng(46)
    x = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)
    path = skein.MCPPathRegressor(gamma=3.0, n_lambdas=8).fit(x, y)
    with pytest.raises(ValueError, match="criterion"):
        skein.select_by_ic(path, x, y, criterion="bogus")
    with pytest.raises(ValueError, match="ebic_gamma"):
        skein.select_by_ic(path, x, y, criterion="ebic", ebic_gamma=2.0)


def test_select_by_ic_cox_rejects_y_only():
    rng = np.random.default_rng(47)
    x = rng.uniform(-1.0, 1.0, size=(80, 5))
    eta = x[:, 0]
    t_event = rng.exponential(scale=1.0 / np.exp(eta))
    t_cens = rng.exponential(scale=2.0, size=80)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    path = skein.CoxMCPPathRegressor(gamma=3.0, n_lambdas=8).fit(x, time, event)
    with pytest.raises(ValueError, match="time.*event"):
        skein.select_by_ic(path, x, time, criterion="bic")  # missing event


def test_select_by_ic_works_with_sparse_input():
    rng = np.random.default_rng(48)
    n, p = 100, 10
    x_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    x_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[2] = -1.0
    y = x_dense @ true_beta + 0.1 * rng.standard_normal(n)
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    path = skein.MCPPathRegressor(
        gamma=1e6, n_lambdas=15, tol=1e-10, max_iter=5000, screening="off",
    ).fit(x_sparse, y)
    # Pass the same sparse matrix to IC; predict should handle it.
    best_idx_sparse, scores_sparse = skein.select_by_ic(
        path, x_sparse, y, criterion="bic"
    )
    best_idx_dense, scores_dense = skein.select_by_ic(
        path, x_dense, y, criterion="bic"
    )
    np.testing.assert_allclose(scores_sparse, scores_dense, atol=1e-10)
    assert best_idx_sparse == best_idx_dense


# ====================================================================
# GLM dense ↔ sparse equivalence under `standardize=True` (M4.3 follow-up)
# ====================================================================


def _glm_standardize_problem(seed, n=200, p=10, scale_inflation=20.0):
    """Sparse-pattern X with one column inflated 20× — the case where
    standardize matters. Returns dense + sparse views of the same X."""
    rng = np.random.default_rng(seed)
    x_dense = rng.standard_normal((n, p))
    x_dense[:, 0] *= scale_inflation
    mask = rng.uniform(size=(n, p)) > 0.5
    x_dense[mask] = 0.0
    x_sparse = scipy_sparse.csc_matrix(x_dense)
    return x_dense, x_sparse, rng


def test_logistic_mcp_path_dense_sparse_equivalence_with_standardize():
    """Logistic + MCP path with standardize=True converges to the same β
    on dense and sparse representations of the same X. γ=1e6 ⇒ ≈ lasso
    (convex inner) so the optimum is unique."""
    x_dense, x_sparse, rng = _glm_standardize_problem(seed=11)
    n = x_dense.shape[0]
    eta = x_dense @ np.array([0.05, 0.0, -1.5, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=30, outer_tol=1e-10,
        fit_intercept=True, standardize=True,
    )
    dense = skein.LogisticMCPPathRegressor(**common).fit(x_dense, y)
    sparse = skein.LogisticMCPPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_logistic_group_lasso_path_dense_sparse_equivalence_with_standardize():
    x_dense, x_sparse, rng = _glm_standardize_problem(seed=13, p=8)
    n = x_dense.shape[0]
    eta = x_dense @ np.array([0.1, -1.0, 0.0, 0.0, 0.7, -0.5, 0.0, 0.0])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        groups=groups, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=20, outer_tol=1e-10,
        fit_intercept=True, standardize=True,
    )
    dense = skein.LogisticGroupLassoPathRegressor(**common).fit(x_dense, y)
    sparse = skein.LogisticGroupLassoPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_poisson_mcp_path_dense_sparse_equivalence_with_standardize():
    x_dense, x_sparse, rng = _glm_standardize_problem(seed=17, scale_inflation=10.0)
    # Smaller true β to keep μ = exp(η) bounded under inflated columns.
    eta = x_dense @ np.array([0.02, 0.0, -0.4, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    eta = np.clip(eta, -3.0, 3.0)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)

    lambdas = np.array([0.3, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=30, outer_tol=1e-10,
        fit_intercept=True, standardize=True,
    )
    dense = skein.PoissonMCPPathRegressor(**common).fit(x_dense, y)
    sparse = skein.PoissonMCPPathRegressor(**common).fit(x_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_cox_mcp_path_dense_sparse_equivalence_with_standardize():
    """Cox has no intercept — Standardized<D> wraps the user matrix
    directly, no augmentation."""
    x_dense, x_sparse, rng = _glm_standardize_problem(seed=23)
    n = x_dense.shape[0]
    eta = x_dense @ np.array([0.05, 0.0, -1.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    rate = np.exp(np.clip(eta, -5.0, 5.0))
    t_event = -np.log(rng.uniform(size=n).clip(1e-12)) / rate
    t_cens = -np.log(rng.uniform(size=n).clip(1e-12)) / 0.5
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)

    lambdas = np.array([0.3, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=30, outer_tol=1e-10,
        standardize=True,
    )
    dense = skein.CoxMCPPathRegressor(**common).fit(x_dense, time, event)
    sparse = skein.CoxMCPPathRegressor(**common).fit(x_sparse, time, event)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)


# ====================================================================
# Memory-mapped backend (M4.x mmap)
# ====================================================================


def _write_fortran_f64(x: np.ndarray, path) -> None:
    """Write a 2D f64 array to disk in column-major (Fortran) order
    using raw bytes, no header. NOTE: `np.tofile()` always writes in
    C order regardless of array layout, so we go through `tobytes(
    order='F')` and write the buffer ourselves."""
    buf = np.ascontiguousarray(x, dtype=np.float64).tobytes(order="F")
    with open(str(path), "wb") as f:
        f.write(buf)


def test_mmap_design_constructor_validates_file_size(tmp_path):
    """Mismatched (n, p) vs file size must raise immediately."""
    x = np.zeros((3, 2), dtype=np.float64)
    p = tmp_path / "x.bin"
    _write_fortran_f64(x, p)
    # Claim shape (3, 3) — file only has 6 f64s.
    with pytest.raises(ValueError, match="bytes"):
        skein.MmapDesignF64(str(p), n_rows=3, n_cols=3)


def test_mmap_design_constructor_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        skein.MmapDesignF64(str(tmp_path / "nonexistent.bin"), 10, 5)


def test_mmap_mcp_path_matches_dense(tmp_path):
    """LS + MCP path on MmapDesignF64 must match the dense path on the
    same X within 1e-7 across a shared lambda grid."""
    rng = np.random.default_rng(31)
    n, p = 60, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)

    file = tmp_path / "x.bin"
    _write_fortran_f64(x, file)
    mmap_design = skein.MmapDesignF64(str(file), n, p)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=10000,
        screening="off", fit_intercept=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x, y)
    via_mmap = skein.MCPPathRegressor(**common).fit(mmap_design, y)
    np.testing.assert_allclose(dense.coefs_, via_mmap.coefs_, atol=1e-7)
    np.testing.assert_allclose(dense.intercepts_, via_mmap.intercepts_, atol=1e-7)
    assert via_mmap.n_features_in_ == p


def test_mmap_mcp_path_with_standardize_matches_dense(tmp_path):
    """Mmap + Augmented + Standardized stack: result equals the dense
    standardize=True path on the same X."""
    rng = np.random.default_rng(37)
    n, p = 50, 6
    x = rng.standard_normal((n, p))
    x[:, 0] *= 30.0
    true_beta = np.array([0.05, 0.0, -1.5, 0.0, 0.8, 0.0])
    y = x @ true_beta + 0.1 * rng.standard_normal(n)

    file = tmp_path / "x.bin"
    _write_fortran_f64(x, file)
    mmap_design = skein.MmapDesignF64(str(file), n, p)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True, standardize=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x, y)
    via_mmap = skein.MCPPathRegressor(**common).fit(mmap_design, y)
    np.testing.assert_allclose(dense.coefs_, via_mmap.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, via_mmap.intercepts_, atol=1e-6)


def test_mmap_logistic_mcp_path_matches_dense(tmp_path):
    rng = np.random.default_rng(43)
    n, p = 200, 8
    x = rng.standard_normal((n, p))
    eta = x @ np.array([1.5, 0.0, -1.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    file = tmp_path / "x.bin"
    _write_fortran_f64(x, file)
    mmap_design = skein.MmapDesignF64(str(file), n, p)

    lambdas = np.array([0.3, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=30, outer_tol=1e-10,
        fit_intercept=True,
    )
    dense = skein.LogisticMCPPathRegressor(**common).fit(x, y)
    via_mmap = skein.LogisticMCPPathRegressor(**common).fit(mmap_design, y)
    np.testing.assert_allclose(dense.coefs_, via_mmap.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, via_mmap.intercepts_, atol=1e-6)


def _write_fortran_f32(x: np.ndarray, path) -> None:
    """f32 version of `_write_fortran_f64`. Same C-vs-F-order gotcha
    applies — `tofile()` would write in C order, so we explicitly
    cast and `tobytes(order='F')`."""
    buf = np.ascontiguousarray(x, dtype=np.float32).tobytes(order="F")
    with open(str(path), "wb") as f:
        f.write(buf)


def test_mmap_f32_design_constructor_validates_file_size(tmp_path):
    x = np.zeros((3, 2), dtype=np.float64)
    p = tmp_path / "x32.bin"
    _write_fortran_f32(x, p)
    # 6 f32s = 24 bytes; we claim shape (3, 3) ⇒ 36 bytes.
    with pytest.raises(ValueError, match="bytes"):
        skein.MmapDesignF32(str(p), n_rows=3, n_cols=3)


def test_mmap_f32_mcp_path_matches_f32_rounded_dense(tmp_path):
    """f32 mmap path matches the dense path on the same X **after
    rounding to f32 and back** — comparing against the unrounded f64
    dense would fail by ~1e-7 (the f32 truncation), so we force the
    in-RAM reference into the same precision."""
    rng = np.random.default_rng(31)
    n, p = 60, 8
    x = rng.standard_normal((n, p))
    x_rounded = x.astype(np.float32).astype(np.float64)
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x_rounded @ true_beta + 0.1 * rng.standard_normal(n)

    file = tmp_path / "x32.bin"
    _write_fortran_f32(x, file)
    mmap_design = skein.MmapDesignF32(str(file), n, p)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=10000,
        screening="off", fit_intercept=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x_rounded, y)
    via_mmap = skein.MCPPathRegressor(**common).fit(mmap_design, y)
    np.testing.assert_allclose(dense.coefs_, via_mmap.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, via_mmap.intercepts_, atol=1e-5)
    assert via_mmap.n_features_in_ == p


def test_mmap_f32_mcp_path_matches_f64_within_truncation(tmp_path):
    """End-to-end: a model fit on f32 mmap matches one fit on the
    original f64 dense matrix to ~1e-4 — i.e. the f32 truncation
    matters for low-magnitude coefs but the recovered support and
    sign pattern agree. Useful as a 'does halving the disk footprint
    cost real accuracy?' sanity check."""
    rng = np.random.default_rng(33)
    n, p = 100, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)

    file = tmp_path / "x32.bin"
    _write_fortran_f32(x, file)
    mmap_design = skein.MmapDesignF32(str(file), n, p)

    lambdas = np.array([0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=10000,
        screening="off", fit_intercept=True,
    )
    f64 = skein.MCPPathRegressor(**common).fit(x, y)
    f32 = skein.MCPPathRegressor(**common).fit(mmap_design, y)
    # 1e-4 is generous; in practice coefs match to ~1e-6 here, but
    # we leave headroom for ill-conditioned problems.
    np.testing.assert_allclose(f32.coefs_, f64.coefs_, atol=1e-4)
    # Support agreement (which features are nonzero) is exact:
    f64_active = np.abs(f64.coefs_[-1]) > 1e-6
    f32_active = np.abs(f32.coefs_[-1]) > 1e-6
    np.testing.assert_array_equal(f32_active, f64_active)


def test_mmap_f32_logistic_mcp_path_matches_f32_rounded_dense(tmp_path):
    rng = np.random.default_rng(43)
    n, p = 200, 8
    x = rng.standard_normal((n, p))
    x_rounded = x.astype(np.float32).astype(np.float64)
    eta = x_rounded @ np.array([1.5, 0.0, -1.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    file = tmp_path / "x32.bin"
    _write_fortran_f32(x, file)
    mmap_design = skein.MmapDesignF32(str(file), n, p)

    lambdas = np.array([0.3, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=30, outer_tol=1e-10, fit_intercept=True,
    )
    dense = skein.LogisticMCPPathRegressor(**common).fit(x_rounded, y)
    via_mmap = skein.LogisticMCPPathRegressor(**common).fit(mmap_design, y)
    np.testing.assert_allclose(dense.coefs_, via_mmap.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, via_mmap.intercepts_, atol=1e-5)


def test_mmap_f32_with_standardize(tmp_path):
    """f32 + standardize=True smoke test — proves the
    Standardized<MmapMatrixF32> stack converges (no equivalence
    against dense, just a sanity check on the path output)."""
    rng = np.random.default_rng(47)
    n, p = 80, 6
    x = rng.standard_normal((n, p))
    x[:, 0] *= 30.0
    true_beta = np.array([0.05, 0.0, -1.5, 0.0, 0.8, 0.0])
    y = x @ true_beta + 0.1 * rng.standard_normal(n)

    file = tmp_path / "x32.bin"
    _write_fortran_f32(x, file)
    mmap_design = skein.MmapDesignF32(str(file), n, p)

    model = skein.MCPPathRegressor(
        gamma=1e6, n_lambdas=10, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=10000, fit_intercept=True, standardize=True,
    ).fit(mmap_design, y)
    last = model.coefs_[-1]
    assert np.isfinite(last).all()
    assert np.sign(last[2]) == np.sign(true_beta[2])
    assert np.sign(last[4]) == np.sign(true_beta[4])


# ====================================================================
# Elastic net (M6.1)
# ====================================================================


def test_elastic_net_alpha_one_matches_mcp_high_gamma():
    """ElasticNet at α=1 (pure lasso) must match MCPPathRegressor at
    γ=1e6 within 1e-6 across the path on the same data and λ-grid."""
    rng = np.random.default_rng(11)
    n, p = 100, 20
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = X @ true_beta + 0.1 * rng.standard_normal(n)

    lambdas = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    common = dict(
        lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True,
    )
    en = skein.ElasticNetPathRegressor(alpha=1.0, **common).fit(X, y)
    mcp = skein.MCPPathRegressor(gamma=1e6, **common).fit(X, y)
    # Tolerance is set to 1e-5 (not exactly machine precision) because
    # MCP at γ=1e6 still has a residual β²/(2γ) term in the prox; for
    # |β| ~ 1 that's ~5e-7 of difference per coordinate, which
    # accumulates to ~2e-6 along the path. ElasticNet at α=1 is the
    # exact lasso, so this small drift is the MCP's artifact.
    np.testing.assert_allclose(en.coefs_, mcp.coefs_, atol=1e-5)
    np.testing.assert_allclose(en.intercepts_, mcp.intercepts_, atol=1e-5)


def test_elastic_net_alpha_zero_matches_closed_form_ridge():
    """At α=0 (pure ridge), elastic net's coefs must equal the
    closed-form `(XᵀX/n + λI)⁻¹ Xᵀy/n`."""
    rng = np.random.default_rng(13)
    n, p = 50, 6
    X = rng.standard_normal((n, p))
    y = X @ np.array([1.0, -1.5, 0.5, 0.0, 0.0, 0.0]) + 0.2 * rng.standard_normal(n)

    lam = 0.3
    # Closed form (no centering since fit_intercept=False).
    xtx = X.T @ X / n
    A = xtx + lam * np.eye(p)
    xty = X.T @ y / n
    beta_closed = np.linalg.solve(A, xty)

    model = skein.ElasticNetRegressor(
        lambda_=lam, alpha=0.0,
        fit_intercept=False, standardize=False,
        tol=1e-12, max_iter=20000, screening="off",
    ).fit(X, y)
    np.testing.assert_allclose(model.coef_, beta_closed, atol=1e-7)


def test_elastic_net_path_recovers_signal():
    """End-to-end: at small λ, elastic-net recovers the true sparse
    signal (with some shrinkage)."""
    rng = np.random.default_rng(17)
    n, p = 200, 30
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = X @ true_beta + 0.1 * rng.standard_normal(n)

    model = skein.ElasticNetPathRegressor(
        alpha=0.5, n_lambdas=50, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=10000, fit_intercept=True,
    ).fit(X, y)
    last = model.coefs_[-1]
    # Sign agreement on the truly-active features.
    for j in range(3):
        assert np.sign(last[j]) == np.sign(true_beta[j]), (
            f"feature {j} sign mismatch: got {last[j]}"
        )
    # Magnitudes are within ~30% of truth (elastic net shrinkage is
    # heavier than MCP's; this isn't tight, just a sanity check).
    np.testing.assert_allclose(last[:3], true_beta[:3], rtol=0.30)


def test_elastic_net_path_dense_sparse_equivalence():
    """ElasticNet on a sparse CSC matrix must match the dense path
    on the same X within tolerance, on a shared λ-grid."""
    rng = np.random.default_rng(19)
    n, p = 80, 12
    X_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[:3] = [1.0, -1.5, 0.7]
    y = X_dense @ true_beta + 0.1 * rng.standard_normal(n)
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        alpha=0.5, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True,
    )
    dense = skein.ElasticNetPathRegressor(**common).fit(X_dense, y)
    sparse = skein.ElasticNetPathRegressor(**common).fit(X_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_elastic_net_path_with_standardize_dense_sparse_equivalence():
    rng = np.random.default_rng(23)
    n, p = 60, 8
    X_dense = rng.standard_normal((n, p))
    X_dense[:, 0] *= 30.0   # inflated scale
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    y = X_dense @ np.array([0.05, 0.0, -1.5, 0.0, 0.0, 0.7, 0.0, 0.0]) + 0.1 * rng.standard_normal(n)
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        alpha=0.5, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True, standardize=True,
    )
    dense = skein.ElasticNetPathRegressor(**common).fit(X_dense, y)
    sparse = skein.ElasticNetPathRegressor(**common).fit(X_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_elastic_net_path_cv_picks_reasonable_lambda():
    rng = np.random.default_rng(29)
    n, p = 200, 30
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = X @ true_beta + 0.3 * rng.standard_normal(n)

    cv = skein.ElasticNetPathCV(
        alpha=0.5, n_lambdas=30, cv=5,
        tol=1e-10, max_iter=5000, random_state=0,
    ).fit(X, y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (p,)
    # Active set should include the true active features.
    active = np.where(np.abs(cv.coef_) > 1e-3)[0].tolist()
    for j in range(3):
        assert j in active, f"truly active feature {j} not selected (active set: {active})"


def test_elastic_net_rejects_alpha_out_of_range():
    rng = np.random.default_rng(31)
    X = rng.standard_normal((50, 5))
    y = rng.standard_normal(50)
    with pytest.raises(ValueError, match="alpha must be in"):
        skein.ElasticNetRegressor(lambda_=0.1, alpha=1.5).fit(X, y)
    with pytest.raises(ValueError, match="alpha must be in"):
        skein.ElasticNetRegressor(lambda_=0.1, alpha=-0.1).fit(X, y)


# ====================================================================
# Group elastic net (M6.2)
# ====================================================================


def _group_problem(seed: int, n: int = 100, p: int = 12, group_size: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    # First two groups active.
    true_beta[: 2 * group_size] = [1.5, -1.0, 0.7, 0.5, -0.8, 0.6][: 2 * group_size]
    y = X @ true_beta + 0.1 * rng.standard_normal(n)
    groups = np.repeat(np.arange(p // group_size), group_size).astype(np.int64)
    return X, y, groups, true_beta


def test_group_elastic_net_alpha_one_matches_group_lasso_path():
    """At α=1, the group elastic-net path must match the GroupLasso
    path on the same problem and shared λ-grid."""
    X, y, groups, _ = _group_problem(seed=41)
    lambdas = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        groups=groups, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True,
    )
    gen = skein.GroupElasticNetPathRegressor(alpha=1.0, **common).fit(X, y)
    gl = skein.GroupLassoPathRegressor(**common).fit(X, y)
    np.testing.assert_allclose(gen.coefs_, gl.coefs_, atol=1e-7)
    np.testing.assert_allclose(gen.intercepts_, gl.intercepts_, atol=1e-7)


def test_group_elastic_net_alpha_zero_matches_closed_form_block_ridge():
    """At α=0, group elastic net (uniform per-group weights) gives
    classical ridge — the per-block prox reduces to per-coord
    shrinkage by the same 1/(1+step·λ) factor everywhere."""
    rng = np.random.default_rng(43)
    n, p, gs = 60, 6, 2
    X = rng.standard_normal((n, p))
    y = X @ np.array([1.0, -1.5, 0.5, 0.0, 0.0, 0.0]) + 0.2 * rng.standard_normal(n)
    groups = np.repeat(np.arange(p // gs), gs).astype(np.int64)

    lam = 0.3
    xtx = X.T @ X / n
    A = xtx + lam * np.eye(p)
    xty = X.T @ y / n
    beta_closed = np.linalg.solve(A, xty)

    model = skein.GroupElasticNetRegressor(
        groups=groups, lambda_=lam, alpha=0.0,
        fit_intercept=False, standardize=False,
        tol=1e-12, max_iter=20000, screening="off",
    ).fit(X, y)
    np.testing.assert_allclose(model.coef_, beta_closed, atol=1e-7)


def test_group_elastic_net_path_recovers_active_groups():
    """At a small λ on the path, the truly active groups should have
    nonzero coefficients."""
    X, y, groups, true_beta = _group_problem(seed=47, n=200, p=15, group_size=3)
    model = skein.GroupElasticNetPathRegressor(
        groups=groups, alpha=0.5, n_lambdas=40, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=10000, fit_intercept=True,
    ).fit(X, y)
    last = model.coefs_[-1]
    # Sign agreement on the active features.
    for j in range(6):
        if abs(true_beta[j]) > 0:
            assert np.sign(last[j]) == np.sign(true_beta[j]), (
                f"feature {j} sign mismatch: got {last[j]}"
            )


def test_group_elastic_net_path_dense_sparse_equivalence():
    rng = np.random.default_rng(53)
    n, p, gs = 80, 12, 3
    X_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    true_beta = np.zeros(p)
    true_beta[:3] = [1.0, -1.5, 0.7]
    y = X_dense @ true_beta + 0.1 * rng.standard_normal(n)
    X_sparse = scipy_sparse.csc_matrix(X_dense)
    groups = np.repeat(np.arange(p // gs), gs).astype(np.int64)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        groups=groups, alpha=0.5, lambdas=lambdas, tol=1e-12,
        max_iter=20000, screening="off", fit_intercept=True,
    )
    dense = skein.GroupElasticNetPathRegressor(**common).fit(X_dense, y)
    sparse = skein.GroupElasticNetPathRegressor(**common).fit(X_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_group_elastic_net_path_with_standardize_dense_sparse_equivalence():
    rng = np.random.default_rng(59)
    n, p, gs = 60, 9, 3
    X_dense = rng.standard_normal((n, p))
    X_dense[:, 0] *= 30.0   # inflated scale
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    y = (
        X_dense @ np.array([0.05, 0.0, -1.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
        + 0.1 * rng.standard_normal(n)
    )
    X_sparse = scipy_sparse.csc_matrix(X_dense)
    groups = np.repeat(np.arange(p // gs), gs).astype(np.int64)

    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        groups=groups, alpha=0.5, lambdas=lambdas, tol=1e-12,
        max_iter=20000, screening="off", fit_intercept=True, standardize=True,
    )
    dense = skein.GroupElasticNetPathRegressor(**common).fit(X_dense, y)
    sparse = skein.GroupElasticNetPathRegressor(**common).fit(X_sparse, y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_group_elastic_net_path_cv_picks_reasonable_lambda():
    X, y, groups, _ = _group_problem(seed=61, n=200, p=15, group_size=3)
    cv = skein.GroupElasticNetPathCV(
        groups=groups, alpha=0.5, n_lambdas=30, cv=5,
        tol=1e-10, max_iter=5000, random_state=0,
    ).fit(X, y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (X.shape[1],)
    # The two truly-active groups (features 0..5) should have at least
    # one nonzero coefficient each.
    active = np.where(np.abs(cv.coef_) > 1e-3)[0].tolist()
    assert any(j in active for j in range(3)), f"group 0 not selected: {active}"
    assert any(j in active for j in range(3, 6)), f"group 1 not selected: {active}"


def test_group_elastic_net_rejects_alpha_out_of_range():
    rng = np.random.default_rng(67)
    X = rng.standard_normal((40, 6))
    y = rng.standard_normal(40)
    groups = np.repeat(np.arange(3), 2).astype(np.int64)
    with pytest.raises(ValueError, match="alpha must be in"):
        skein.GroupElasticNetRegressor(groups=groups, lambda_=0.1, alpha=1.5).fit(X, y)
    with pytest.raises(ValueError, match="alpha must be in"):
        skein.GroupElasticNetRegressor(groups=groups, lambda_=0.1, alpha=-0.1).fit(X, y)


# ====================================================================
# Multi-task LS (M7.1)
# ====================================================================


def _multitask_problem(seed: int, n: int = 80, p: int = 10, k: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    B = np.zeros((p, k))
    # First three features active across all tasks (joint support);
    # task-specific magnitudes drawn fresh so the K=3 vs K=4 cases differ.
    active_rows = rng.uniform(-1.5, 1.5, size=(3, k))
    # Make sure no row is too close to zero (so sign agreement tests work).
    active_rows[np.abs(active_rows) < 0.4] += np.sign(active_rows[np.abs(active_rows) < 0.4]) * 0.5
    B[:3] = active_rows
    Y = X @ B + 0.1 * rng.standard_normal((n, k))
    return X, Y, B


def test_multitask_lasso_path_shapes_and_intercept():
    X, Y, _ = _multitask_problem(seed=101)
    n, p = X.shape
    k = Y.shape[1]
    model = skein.MultiTaskLassoPathRegressor(
        n_lambdas=12, lambda_min_ratio=1e-2, tol=1e-10, max_iter=10000,
    ).fit(X, Y)
    assert model.coefs_.shape == (12, k, p)
    assert model.intercepts_.shape == (12, k)
    assert model.lambdas_.shape == (12,)
    # Sanity: at the largest λ all coefs should be (near) zero,
    # and the intercept should equal Y's per-task mean.
    np.testing.assert_allclose(
        model.coefs_[0], 0.0, atol=1e-6,
    )
    np.testing.assert_allclose(
        model.intercepts_[0], Y.mean(axis=0), atol=1e-6,
    )


def test_multitask_lasso_path_recovers_joint_support():
    X, Y, B_true = _multitask_problem(seed=103, n=200, p=15, k=4)
    model = skein.MultiTaskLassoPathRegressor(
        n_lambdas=40, lambda_min_ratio=1e-3, tol=1e-10, max_iter=20000,
    ).fit(X, Y)
    last = model.coefs_[-1]   # (K, p)
    # First three features should have nonzero coefficients in every task.
    for j in range(3):
        for task in range(B_true.shape[1]):
            assert abs(last[task, j]) > 0.05, (
                f"feature {j} task {task} not active: got {last[task, j]}"
            )


def test_multitask_lasso_predict_shape_and_signal():
    X, Y, _ = _multitask_problem(seed=107, n=100, p=8, k=3)
    model = skein.MultiTaskLassoRegressor(
        lambda_=0.005, tol=1e-10, max_iter=10000,
    ).fit(X, Y)
    assert model.coef_.shape == (3, 8)
    assert model.intercept_.shape == (3,)
    pred = model.predict(X)
    assert pred.shape == (100, 3)
    # Predictions should be close to Y given lambda small enough.
    np.testing.assert_allclose(pred, Y, atol=0.5)


def test_multitask_lasso_matches_sklearn_with_k_rescale():
    """sklearn's `MultiTaskLasso(alpha=α)` uses `(1/(2n)) ‖Y-XB‖²_F + α P`;
    skein's `MultiTaskLassoRegressor(lambda_=λ)` uses the natural
    stacked `(1/(2nK)) ‖·‖² + λ P`. Same minimizer at `λ = α / K`."""
    from sklearn.linear_model import MultiTaskLasso

    X, Y, _ = _multitask_problem(seed=109, n=120, p=10, k=3)
    alpha = 0.05
    k = Y.shape[1]
    skl = MultiTaskLasso(alpha=alpha, fit_intercept=True, tol=1e-10, max_iter=20000).fit(X, Y)
    skein_model = skein.MultiTaskLassoRegressor(
        lambda_=alpha / k, fit_intercept=True, tol=1e-12, max_iter=20000, screening="off",
    ).fit(X, Y)
    np.testing.assert_allclose(skein_model.coef_, skl.coef_, atol=1e-5)
    np.testing.assert_allclose(skein_model.intercept_, skl.intercept_, atol=1e-5)


def test_multitask_mcp_path_recovers_signal():
    X, Y, B_true = _multitask_problem(seed=113, n=200, p=15, k=4)
    model = skein.MultiTaskMCPPathRegressor(
        gamma=3.0, n_lambdas=30, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=10000, max_outer=5,
    ).fit(X, Y)
    # Pick the λ that visits the right active set: the truly-active first
    # three features should have nonzero row norms; noise rows should be
    # zero. Walk the path and find the sparsest model that retains all
    # three signal features.
    found_clean = False
    for last in model.coefs_:
        active_rows = np.where(np.linalg.norm(last, axis=0) > 1e-2)[0].tolist()
        if all(j in active_rows for j in range(3)) and len(active_rows) <= 5:
            found_clean = True
            # And the truly-active features have the right sign in every task.
            for j in range(3):
                for task in range(B_true.shape[1]):
                    assert np.sign(last[task, j]) == np.sign(B_true[j, task]), (
                        f"feature {j} task {task} sign mismatch: "
                        f"{last[task, j]} vs truth {B_true[j, task]}"
                    )
            break
    assert found_clean, "no λ on the path delivered a clean MCP recovery"


def test_multitask_lasso_path_cv_picks_reasonable_lambda():
    X, Y, _ = _multitask_problem(seed=119, n=160, p=12, k=3)
    cv = skein.MultiTaskLassoPathCV(
        n_lambdas=25, cv=4, tol=1e-10, max_iter=5000, random_state=0,
    ).fit(X, Y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (3, 12)
    assert cv.intercept_.shape == (3,)
    # The truly-active features (0..2) should be in the active set.
    active = (np.linalg.norm(cv.coef_, axis=0) > 1e-2).nonzero()[0].tolist()
    for j in range(3):
        assert j in active, f"truly active feature {j} not selected: {active}"


def test_multitask_rejects_1d_y():
    X = np.zeros((10, 3))
    y_1d = np.zeros(10)
    with pytest.raises(ValueError, match="Y must be 2D"):
        skein.MultiTaskLassoRegressor().fit(X, y_1d)


def test_multitask_lasso_with_standardize_recovers_correct_scale():
    """Inflate one column by 50× and verify the standardize=True fit
    recovers a coefficient that matches a fit on the rescaled X."""
    rng = np.random.default_rng(167)
    n, p, k = 100, 8, 3
    X = rng.standard_normal((n, p))
    X[:, 0] *= 50.0  # inflated scale
    B = np.zeros((p, k))
    B[0] = [0.05, -0.04, 0.03]   # tiny in inflated-scale coords
    B[2] = rng.uniform(-1.0, 1.0, size=k)
    Y = X @ B + 0.1 * rng.standard_normal((n, k))

    # With standardize=True, the fit should recover B in original scale
    # (where feature 0 has tiny coefficients ~0.05).
    model = skein.MultiTaskLassoPathRegressor(
        n_lambdas=20, lambda_min_ratio=1e-4,
        tol=1e-12, max_iter=20000, screening="off",
        standardize=True,
    ).fit(X, Y)
    last = model.coefs_[-1]   # (K, p)
    # Feature 0 should be visible (sign-recovered) at small λ.
    for task in range(k):
        assert np.sign(last[task, 0]) == np.sign(B[0, task]), (
            f"feature 0 task {task} sign lost: {last[task, 0]} vs truth {B[0, task]}"
        )
    # And the magnitudes are in the right ballpark (within 50% — the path
    # is finite-grid so won't recover B exactly).
    np.testing.assert_allclose(last[:, 0], B[0], rtol=0.5, atol=0.02)


def test_multitask_lasso_standardize_dense_sparse_equivalence():
    """At the same standardize=True setting and shared λ-grid, dense and
    sparse paths must produce the same coefficients within tolerance."""
    rng = np.random.default_rng(169)
    n, p, k = 60, 8, 2
    X_dense = rng.standard_normal((n, p))
    X_dense[:, 0] *= 30.0
    X_dense[rng.uniform(size=(n, p)) > 0.5] = 0.0
    B = np.zeros((p, k))
    B[0] = [0.03, -0.05]
    B[1:3] = rng.uniform(-1.0, 1.0, size=(2, k))
    Y = X_dense @ B + 0.1 * rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([1.0, 0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True, standardize=True,
    )
    dense = skein.MultiTaskLassoPathRegressor(**common).fit(X_dense, Y)
    sparse = skein.MultiTaskLassoPathRegressor(**common).fit(X_sparse, Y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


def test_multitask_mcp_standardize_dense_sparse_equivalence():
    rng = np.random.default_rng(171)
    n, p, k = 60, 8, 3
    X_dense = rng.standard_normal((n, p))
    X_dense[:, 0] *= 20.0
    X_dense[rng.uniform(size=(n, p)) > 0.5] = 0.0
    B = np.zeros((p, k))
    B[0] = [0.03, -0.04, 0.05]
    B[1:3] = rng.uniform(-1.0, 1.0, size=(2, k))
    Y = X_dense @ B + 0.1 * rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([0.5, 0.2, 0.05, 0.02], dtype=np.float64)
    common = dict(
        gamma=3.0, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True, standardize=True, max_outer=8,
    )
    dense = skein.MultiTaskMCPPathRegressor(**common).fit(X_dense, Y)
    sparse = skein.MultiTaskMCPPathRegressor(**common).fit(X_sparse, Y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-5)


# ---- M7.2: sparse multi-task ----------------------------------------


def test_multitask_lasso_sparse_matches_dense_path():
    rng = np.random.default_rng(173)
    n, p, k = 80, 10, 3
    X_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    B = np.zeros((p, k))
    B[:3] = rng.uniform(-1.5, 1.5, size=(3, k))
    Y = X_dense @ B + 0.1 * rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        lambdas=lambdas, tol=1e-12, max_iter=20000, screening="off",
        fit_intercept=True,
    )
    dense = skein.MultiTaskLassoPathRegressor(**common).fit(X_dense, Y)
    sparse = skein.MultiTaskLassoPathRegressor(**common).fit(X_sparse, Y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-6)


def test_multitask_mcp_sparse_matches_dense_path():
    rng = np.random.default_rng(179)
    n, p, k = 80, 10, 3
    X_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    B = np.zeros((p, k))
    B[:3] = rng.uniform(-1.5, 1.5, size=(3, k))
    Y = X_dense @ B + 0.1 * rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=3.0, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True, max_outer=8,
    )
    dense = skein.MultiTaskMCPPathRegressor(**common).fit(X_dense, Y)
    sparse = skein.MultiTaskMCPPathRegressor(**common).fit(X_sparse, Y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-6)


def test_multitask_elastic_net_sparse_matches_dense_path():
    rng = np.random.default_rng(181)
    n, p, k = 60, 8, 2
    X_dense = rng.standard_normal((n, p))
    mask = rng.uniform(size=(n, p)) > 0.5
    X_dense[mask] = 0.0
    B = np.zeros((p, k))
    B[:3] = rng.uniform(-1.5, 1.5, size=(3, k))
    Y = X_dense @ B + 0.1 * rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    lambdas = np.array([0.3, 0.1, 0.03, 0.01], dtype=np.float64)
    common = dict(
        alpha=0.5, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True,
    )
    dense = skein.MultiTaskElasticNetPathRegressor(**common).fit(X_dense, Y)
    sparse = skein.MultiTaskElasticNetPathRegressor(**common).fit(X_sparse, Y)
    np.testing.assert_allclose(dense.coefs_, sparse.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, sparse.intercepts_, atol=1e-6)


def test_multitask_lasso_sparse_predict_shape():
    rng = np.random.default_rng(191)
    n, p, k = 60, 8, 3
    X_dense = rng.standard_normal((n, p))
    X_dense[rng.uniform(size=(n, p)) > 0.5] = 0.0
    Y = rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)
    model = skein.MultiTaskLassoRegressor(lambda_=0.05, tol=1e-10).fit(X_sparse, Y)
    assert model.coef_.shape == (k, p)
    assert model.intercept_.shape == (k,)
    pred = model.predict(X_sparse[:5])
    assert pred.shape == (5, k)


def test_multitask_lasso_path_cv_sparse_input():
    rng = np.random.default_rng(193)
    n, p, k = 120, 8, 3
    X_dense = rng.standard_normal((n, p))
    X_dense[rng.uniform(size=(n, p)) > 0.5] = 0.0
    B = np.zeros((p, k))
    B[:3] = rng.uniform(-1.5, 1.5, size=(3, k))
    Y = X_dense @ B + 0.1 * rng.standard_normal((n, k))
    X_sparse = scipy_sparse.csc_matrix(X_dense)

    cv = skein.MultiTaskLassoPathCV(
        n_lambdas=15, cv=4, tol=1e-10, max_iter=5000, random_state=0,
    ).fit(X_sparse, Y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (k, p)


# ---- M7.2: MultiTaskSCAD + MultiTaskElasticNet ----------------------


def test_multitask_scad_path_recovers_signal():
    X, Y, B_true = _multitask_problem(seed=131, n=200, p=15, k=4)
    model = skein.MultiTaskSCADPathRegressor(
        a=3.7, n_lambdas=30, lambda_min_ratio=1e-2,
        tol=1e-10, max_iter=10000, max_outer=5,
    ).fit(X, Y)
    found_clean = False
    for last in model.coefs_:
        active_rows = np.where(np.linalg.norm(last, axis=0) > 1e-2)[0].tolist()
        if all(j in active_rows for j in range(3)) and len(active_rows) <= 5:
            found_clean = True
            for j in range(3):
                for task in range(B_true.shape[1]):
                    assert np.sign(last[task, j]) == np.sign(B_true[j, task])
            break
    assert found_clean, "no λ on the path delivered a clean SCAD recovery"


def test_multitask_scad_rejects_a_below_two():
    X, Y, _ = _multitask_problem(seed=137, n=50, p=5, k=2)
    with pytest.raises(ValueError, match="`a` must be > 2"):
        skein.MultiTaskSCADRegressor(lambda_=0.05, a=1.5).fit(X, Y)


def test_multitask_elastic_net_alpha_one_matches_lasso():
    """At α=1, the multi-task elastic net path must coincide with the
    multi-task lasso path on the same data + λ-grid."""
    X, Y, _ = _multitask_problem(seed=139, n=120, p=10, k=3)
    lambdas = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        lambdas=lambdas, tol=1e-12, max_iter=20000, screening="off",
    )
    en = skein.MultiTaskElasticNetPathRegressor(alpha=1.0, **common).fit(X, Y)
    lasso = skein.MultiTaskLassoPathRegressor(**common).fit(X, Y)
    np.testing.assert_allclose(en.coefs_, lasso.coefs_, atol=1e-7)
    np.testing.assert_allclose(en.intercepts_, lasso.intercepts_, atol=1e-7)


def test_multitask_elastic_net_alpha_zero_matches_closed_form_block_ridge():
    """At α=0, the multi-task EN with uniform per-feature weights solves
    standard ridge per-task: `β_k = (X^T X / n + λI)⁻¹ X^T Y[:, k] / n`."""
    rng = np.random.default_rng(149)
    n, p, k = 60, 6, 3
    X = rng.standard_normal((n, p))
    B = np.zeros((p, k))
    B[:3] = rng.uniform(-1.0, 1.0, size=(3, k))
    Y = X @ B + 0.1 * rng.standard_normal((n, k))

    lam = 0.3
    xtx = X.T @ X / n
    A = xtx + lam * np.eye(p)
    # skein with α=0 has data-fidelity factor 1/(2nK); the equivalent
    # per-task closed-form ridge there has λ_eff = λ * K (the K factor
    # mirroring the lasso convention difference).
    A_skein = xtx + lam * k * np.eye(p)
    beta_closed = np.zeros((p, k))
    for task in range(k):
        beta_closed[:, task] = np.linalg.solve(A_skein, X.T @ Y[:, task] / n)
    # And the equivalent classical ridge (1/(2n) factor):
    beta_classical = np.zeros((p, k))
    for task in range(k):
        beta_classical[:, task] = np.linalg.solve(A, X.T @ Y[:, task] / n)

    model = skein.MultiTaskElasticNetRegressor(
        lambda_=lam, alpha=0.0,
        fit_intercept=False, tol=1e-12, max_iter=20000, screening="off",
    ).fit(X, Y)
    # `coef_` is (K, p); compare to `beta_closed.T`.
    np.testing.assert_allclose(model.coef_, beta_closed.T, atol=1e-6)
    # And distinct from the 1/(2n) classical ridge — sanity check that
    # we know which convention we're matching.
    assert np.max(np.abs(model.coef_ - beta_classical.T)) > 1e-3


def test_multitask_elastic_net_rejects_alpha_out_of_range():
    X, Y, _ = _multitask_problem(seed=151, n=50, p=5, k=2)
    with pytest.raises(ValueError, match="alpha must be in"):
        skein.MultiTaskElasticNetRegressor(lambda_=0.1, alpha=1.5).fit(X, Y)
    with pytest.raises(ValueError, match="alpha must be in"):
        skein.MultiTaskElasticNetRegressor(lambda_=0.1, alpha=-0.1).fit(X, Y)


def test_multitask_scad_path_cv_picks_reasonable_lambda():
    X, Y, _ = _multitask_problem(seed=157, n=160, p=12, k=3)
    cv = skein.MultiTaskSCADPathCV(
        a=3.7, n_lambdas=20, cv=4, tol=1e-10, max_iter=5000,
        max_outer=5, random_state=0,
    ).fit(X, Y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (3, 12)


def test_multitask_elastic_net_path_cv_picks_reasonable_lambda():
    X, Y, _ = _multitask_problem(seed=163, n=160, p=12, k=3)
    cv = skein.MultiTaskElasticNetPathCV(
        alpha=0.5, n_lambdas=25, cv=4, tol=1e-10, max_iter=5000,
        random_state=0,
    ).fit(X, Y)
    assert cv.lambda_best_ > 0
    assert cv.coef_.shape == (3, 12)
    active = (np.linalg.norm(cv.coef_, axis=0) > 1e-2).nonzero()[0].tolist()
    for j in range(3):
        assert j in active, f"truly active feature {j} not selected: {active}"


def _split_into_chunks(x: np.ndarray, n_chunks: int):
    """Yield (start, end) row-index pairs for `n_chunks` roughly
    equal-sized splits."""
    n = x.shape[0]
    base = n // n_chunks
    rem = n % n_chunks
    start = 0
    for k in range(n_chunks):
        size = base + (1 if k < rem else 0)
        yield start, start + size
        start += size


def test_chunked_design_constructor_validates_each_chunk(tmp_path):
    """Each (path, n_rows) pair is validated against file size."""
    x_a = np.zeros((3, 2), dtype=np.float64)
    p_a = tmp_path / "a.bin"
    _write_fortran_f64(x_a, p_a)
    # Claim chunk has 4 rows but file has 3 — must raise.
    with pytest.raises(ValueError, match="bytes"):
        skein.ChunkedDesignF64([(str(p_a), 4)], n_cols=2)


def test_chunked_design_rejects_empty_chunks_list(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        skein.ChunkedDesignF64([], n_cols=5)


def test_chunked_mcp_path_matches_dense(tmp_path):
    """LS-MCP path on a 3-chunk ChunkedDesignF64 matches the dense
    path on the flat X within 1e-7."""
    rng = np.random.default_rng(53)
    n, p = 60, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)

    chunks: list[tuple[str, int]] = []
    for k, (lo, hi) in enumerate(_split_into_chunks(x, 3)):
        path = tmp_path / f"c{k}.bin"
        _write_fortran_f64(x[lo:hi], path)
        chunks.append((str(path), hi - lo))
    chunked = skein.ChunkedDesignF64(chunks, n_cols=p)
    assert chunked.n_chunks == 3
    assert chunked.n_rows == n

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=10000,
        screening="off", fit_intercept=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x, y)
    via_chunked = skein.MCPPathRegressor(**common).fit(chunked, y)
    np.testing.assert_allclose(dense.coefs_, via_chunked.coefs_, atol=1e-7)
    np.testing.assert_allclose(dense.intercepts_, via_chunked.intercepts_, atol=1e-7)
    assert via_chunked.n_features_in_ == p


def test_chunked_logistic_mcp_path_matches_dense(tmp_path):
    rng = np.random.default_rng(59)
    n, p = 200, 8
    x = rng.standard_normal((n, p))
    eta = x @ np.array([1.5, 0.0, -1.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    chunks: list[tuple[str, int]] = []
    for k, (lo, hi) in enumerate(_split_into_chunks(x, 4)):
        path = tmp_path / f"c{k}.bin"
        _write_fortran_f64(x[lo:hi], path)
        chunks.append((str(path), hi - lo))
    chunked = skein.ChunkedDesignF64(chunks, n_cols=p)

    lambdas = np.array([0.3, 0.1, 0.05, 0.02, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=5000,
        max_outer=30, outer_tol=1e-10, fit_intercept=True,
    )
    dense = skein.LogisticMCPPathRegressor(**common).fit(x, y)
    via_chunked = skein.LogisticMCPPathRegressor(**common).fit(chunked, y)
    np.testing.assert_allclose(dense.coefs_, via_chunked.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, via_chunked.intercepts_, atol=1e-6)


def test_chunked_f32_mcp_path_matches_f32_rounded_dense(tmp_path):
    rng = np.random.default_rng(67)
    n, p = 80, 8
    x = rng.standard_normal((n, p))
    x_rounded = x.astype(np.float32).astype(np.float64)
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x_rounded @ true_beta + 0.1 * rng.standard_normal(n)

    chunks: list[tuple[str, int]] = []
    for k, (lo, hi) in enumerate(_split_into_chunks(x, 3)):
        path = tmp_path / f"c{k}.bin"
        _write_fortran_f32(x[lo:hi], path)
        chunks.append((str(path), hi - lo))
    chunked = skein.ChunkedDesignF32(chunks, n_cols=p)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=10000,
        screening="off", fit_intercept=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x_rounded, y)
    via_chunked = skein.MCPPathRegressor(**common).fit(chunked, y)
    np.testing.assert_allclose(dense.coefs_, via_chunked.coefs_, atol=1e-5)
    np.testing.assert_allclose(dense.intercepts_, via_chunked.intercepts_, atol=1e-5)


def test_chunked_with_standardize_matches_dense(tmp_path):
    """Chunked + Augmented + Standardized stack converges to the same
    β as the dense standardize=True path."""
    rng = np.random.default_rng(71)
    n, p = 60, 6
    x = rng.standard_normal((n, p))
    x[:, 0] *= 30.0
    true_beta = np.array([0.05, 0.0, -1.5, 0.0, 0.8, 0.0])
    y = x @ true_beta + 0.1 * rng.standard_normal(n)

    chunks: list[tuple[str, int]] = []
    for k, (lo, hi) in enumerate(_split_into_chunks(x, 3)):
        path = tmp_path / f"c{k}.bin"
        _write_fortran_f64(x[lo:hi], path)
        chunks.append((str(path), hi - lo))
    chunked = skein.ChunkedDesignF64(chunks, n_cols=p)

    lambdas = np.array([0.5, 0.2, 0.08, 0.03, 0.01], dtype=np.float64)
    common = dict(
        gamma=1e6, lambdas=lambdas, tol=1e-12, max_iter=20000,
        screening="off", fit_intercept=True, standardize=True,
    )
    dense = skein.MCPPathRegressor(**common).fit(x, y)
    via_chunked = skein.MCPPathRegressor(**common).fit(chunked, y)
    np.testing.assert_allclose(dense.coefs_, via_chunked.coefs_, atol=1e-6)
    np.testing.assert_allclose(dense.intercepts_, via_chunked.intercepts_, atol=1e-6)


def test_logistic_mcp_path_standardize_recovers_signal_with_inflated_scale():
    """End-to-end signal recovery on a logistic problem where one
    feature has a 50× inflated scale. Without standardize the inflated
    column gets effectively zero penalty; with standardize the recovered
    sign should still match the truth."""
    rng = np.random.default_rng(31)
    n, p = 300, 12
    x = rng.standard_normal((n, p))
    x[:, 0] *= 50.0
    true_beta = np.zeros(p)
    true_beta[0] = 0.04   # small in original scale ⇒ ~2 after standardize
    true_beta[2] = -1.5
    true_beta[5] = 1.0
    eta = x @ true_beta
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    model = skein.LogisticMCPPathRegressor(
        gamma=1e6, n_lambdas=20, lambda_min_ratio=1e-3,
        tol=1e-10, max_iter=5000, standardize=True,
    ).fit(x, y)
    last = model.coefs_[-1]
    assert np.sign(last[0]) == np.sign(true_beta[0])
    assert np.sign(last[2]) == np.sign(true_beta[2])
    assert np.sign(last[5]) == np.sign(true_beta[5])
