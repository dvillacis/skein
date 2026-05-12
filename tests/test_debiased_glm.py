"""Tests for the M5.x-b debiased GLM extensions.

Mirror of `tests/test_debiased.py` for the LS case. The load-bearing
correctness check is again the empirical CI coverage simulation; the
rest is plumbing.

Built on the M3.x convex `LogisticLassoRegressor` / `PoissonLassoRegressor`
primitives — debiasing on top of the prior `MCP(γ=1e9)` approximation
would have inherited its bias.
"""
from __future__ import annotations

import numpy as np
import pytest

from skein_glm import (
    DebiasedLogisticLassoRegressor,
    DebiasedPoissonLassoRegressor,
    debiased_logistic_lasso,
    debiased_poisson_lasso,
)
from skein_glm.debiased import DebiasedGLMResult


# --- problem generators ---------------------------------------------


def _make_logistic_problem(
    n=200, p=30, s=3, *, signal=0.7, seed=0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = signal * rng.choice([-1.0, 1.0], size=s)
    prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = (rng.uniform(size=n) < prob).astype(float)
    return X, y, beta


def _make_poisson_problem(
    n=200, p=30, s=3, *, signal=0.3, seed=0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = signal * rng.choice([-1.0, 1.0], size=s)
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(np.float64)
    return X, y, beta


# --- shape / plumbing -----------------------------------------------


def test_logistic_returns_dataclass_with_shapes() -> None:
    X, y, _ = _make_logistic_problem()
    res = debiased_logistic_lasso(X, y, n_jobs=1)
    assert isinstance(res, DebiasedGLMResult)
    p = X.shape[1]
    assert res.coef_debiased.shape == (p,)
    assert res.coef_glm.shape == (p,)
    assert res.se.shape == (p,)
    assert res.ci_lower.shape == (p,)
    assert res.ci_upper.shape == (p,)
    assert res.pvalues.shape == (p,)
    assert res.z_scores.shape == (p,)
    assert res.Theta.shape == (p, p)
    assert res.mu_fitted.shape == (X.shape[0],)
    assert res.family == "binomial"


def test_poisson_returns_dataclass_with_shapes() -> None:
    X, y, _ = _make_poisson_problem()
    res = debiased_poisson_lasso(X, y, n_jobs=1)
    assert res.coef_debiased.shape == (X.shape[1],)
    assert res.family == "poisson"
    # μ̂ should be positive everywhere for Poisson.
    assert np.all(res.mu_fitted > 0)


def test_logistic_all_finite() -> None:
    X, y, _ = _make_logistic_problem(seed=1)
    res = debiased_logistic_lasso(X, y, n_jobs=1)
    for name in (
        "coef_debiased", "coef_glm", "se", "ci_lower", "ci_upper",
        "pvalues", "z_scores", "Theta", "mu_fitted", "lambda_nodewise",
    ):
        assert np.all(np.isfinite(getattr(res, name))), f"non-finite in {name}"


def test_poisson_all_finite() -> None:
    X, y, _ = _make_poisson_problem(seed=2)
    res = debiased_poisson_lasso(X, y, n_jobs=1)
    for name in (
        "coef_debiased", "coef_glm", "se", "ci_lower", "ci_upper",
        "pvalues", "z_scores", "Theta", "mu_fitted",
    ):
        assert np.all(np.isfinite(getattr(res, name))), f"non-finite in {name}"


def test_ci_ordering_both_families() -> None:
    X, y, _ = _make_logistic_problem(seed=3)
    res = debiased_logistic_lasso(X, y, n_jobs=1)
    assert np.all(res.ci_lower <= res.coef_debiased + 1e-12)
    assert np.all(res.coef_debiased <= res.ci_upper + 1e-12)
    Xp, yp, _ = _make_poisson_problem(seed=3)
    rp = debiased_poisson_lasso(Xp, yp, n_jobs=1)
    assert np.all(rp.ci_lower <= rp.coef_debiased + 1e-12)
    assert np.all(rp.coef_debiased <= rp.ci_upper + 1e-12)


def test_se_nonnegative() -> None:
    X, y, _ = _make_logistic_problem(seed=4)
    res = debiased_logistic_lasso(X, y, n_jobs=1)
    assert np.all(res.se >= 0)


# --- correctness: debiasing direction --------------------------------


def test_logistic_debiased_pvalue_smaller_on_true_active() -> None:
    """At a moderate signal, true active features should have smaller
    p-values than inactive ones, and the first few should be < 0.05."""
    X, y, _ = _make_logistic_problem(n=400, p=30, s=3, signal=1.0, seed=10)
    res = debiased_logistic_lasso(X, y, n_jobs=1)
    assert res.pvalues[:3].mean() < res.pvalues[3:].mean()
    assert np.all(res.pvalues[:3] < 0.05)


def test_poisson_debiased_pvalue_smaller_on_true_active() -> None:
    X, y, _ = _make_poisson_problem(n=400, p=30, s=3, signal=0.4, seed=11)
    res = debiased_poisson_lasso(X, y, n_jobs=1)
    assert res.pvalues[:3].mean() < res.pvalues[3:].mean()
    assert np.all(res.pvalues[:3] < 0.05)


# --- coverage simulation (load-bearing) -----------------------------


def test_logistic_empirical_coverage() -> None:
    """Repeated-experiments coverage on inactive coordinates — these
    are the cleanest signal for variance correctness, since their truth
    is exactly zero and the asymptotic theory applies most cleanly.

    On true-zero features `β_j = 0`, the fraction of 95% CIs containing
    zero should be ≥ 80% (relaxed from nominal 95% for finite-sample
    noise across a moderate number of replications)."""
    rng_seed_base = 1000
    n, p, s = 200, 30, 3
    n_reps = 40
    inactive = slice(s, p)

    covered_inactive = 0
    total_inactive = 0
    for r in range(n_reps):
        X, y, _ = _make_logistic_problem(
            n=n, p=p, s=s, signal=0.8, seed=rng_seed_base + r,
        )
        res = debiased_logistic_lasso(X, y, n_jobs=1)
        in_ci = (res.ci_lower <= 0.0) & (0.0 <= res.ci_upper)
        covered_inactive += int(in_ci[inactive].sum())
        total_inactive += p - s
    rate = covered_inactive / total_inactive
    assert 0.80 <= rate <= 1.0, (
        f"inactive-coordinate coverage {rate:.3f} outside [0.80, 1.0]"
    )


def test_poisson_empirical_coverage() -> None:
    rng_seed_base = 2000
    n, p, s = 200, 30, 3
    n_reps = 40
    inactive = slice(s, p)

    covered_inactive = 0
    total_inactive = 0
    for r in range(n_reps):
        X, y, _ = _make_poisson_problem(
            n=n, p=p, s=s, signal=0.3, seed=rng_seed_base + r,
        )
        res = debiased_poisson_lasso(X, y, n_jobs=1)
        in_ci = (res.ci_lower <= 0.0) & (0.0 <= res.ci_upper)
        covered_inactive += int(in_ci[inactive].sum())
        total_inactive += p - s
    rate = covered_inactive / total_inactive
    assert 0.80 <= rate <= 1.0, (
        f"Poisson inactive-coordinate coverage {rate:.3f} outside [0.80, 1.0]"
    )


# --- offset for Poisson ---------------------------------------------


def test_poisson_offset_changes_fit() -> None:
    """Passing a Poisson offset shifts μ̂ and hence the debiased fit
    relative to the no-offset call."""
    X, y, _ = _make_poisson_problem(n=200, p=15, s=3, seed=30)
    rng = np.random.default_rng(30)
    offset = rng.uniform(-0.3, 0.3, size=200)
    a = debiased_poisson_lasso(X, y, n_jobs=1)
    b = debiased_poisson_lasso(X, y, offset=offset, n_jobs=1)
    assert not np.allclose(a.coef_debiased, b.coef_debiased)
    # Both still produce valid Wald CIs.
    assert np.all(np.isfinite(b.se))


def test_poisson_rejects_bad_offset_shape() -> None:
    X, y, _ = _make_poisson_problem(n=40, p=10, s=2, seed=31)
    with pytest.raises(ValueError, match="offset must be 1D"):
        debiased_poisson_lasso(X, y, offset=np.zeros(99))


# --- input validation ----------------------------------------------


def test_logistic_rejects_non_binary_y() -> None:
    X, y, _ = _make_logistic_problem()
    y_bad = y.copy()
    y_bad[0] = 2.0
    with pytest.raises(ValueError, match="y ∈"):
        debiased_logistic_lasso(X, y_bad)


def test_poisson_rejects_negative_y() -> None:
    X, y, _ = _make_poisson_problem()
    y_bad = y.copy()
    y_bad[0] = -1.0
    with pytest.raises(ValueError, match="y ≥ 0"):
        debiased_poisson_lasso(X, y_bad)


def test_rejects_bad_alpha() -> None:
    X, y, _ = _make_logistic_problem()
    with pytest.raises(ValueError, match="alpha"):
        debiased_logistic_lasso(X, y, alpha=1.5)


# --- sklearn-style wrappers ----------------------------------------


def test_logistic_regressor_pipeline() -> None:
    X, y, _ = _make_logistic_problem(seed=40)
    est = DebiasedLogisticLassoRegressor(n_jobs=1).fit(X, y)
    assert est.coef_.shape == (X.shape[1],)
    assert est.n_features_in_ == X.shape[1]
    for name in (
        "se_", "ci_lower_", "ci_upper_", "pvalues_", "z_scores_",
        "Theta_", "mu_fitted_", "coef_glm_", "family_",
    ):
        assert hasattr(est, name)
    # predict_proba ∈ [0, 1]; predict ∈ {0, 1}.
    proba = est.predict_proba(X)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    classes = est.predict(X)
    assert set(np.unique(classes)).issubset({0.0, 1.0})


def test_poisson_regressor_pipeline() -> None:
    X, y, _ = _make_poisson_problem(seed=41)
    est = DebiasedPoissonLassoRegressor(n_jobs=1).fit(X, y)
    assert est.coef_.shape == (X.shape[1],)
    # predict returns μ̂ = exp(η̂); must be positive.
    mu_hat = est.predict(X)
    assert np.all(mu_hat > 0)
    assert mu_hat.shape == (X.shape[0],)


def test_logistic_regressor_matches_free_function() -> None:
    X, y, _ = _make_logistic_problem(seed=50)
    res = debiased_logistic_lasso(X, y, alpha=0.10, n_jobs=1)
    est = DebiasedLogisticLassoRegressor(alpha=0.10, n_jobs=1).fit(X, y)
    np.testing.assert_allclose(est.coef_, res.coef_debiased)
    np.testing.assert_allclose(est.se_, res.se)
    np.testing.assert_allclose(est.pvalues_, res.pvalues)


def test_n_jobs_serial_parallel_agree() -> None:
    X, y, _ = _make_logistic_problem(n=100, p=15, seed=60)
    r1 = debiased_logistic_lasso(X, y, n_jobs=1)
    r4 = debiased_logistic_lasso(X, y, n_jobs=4)
    np.testing.assert_allclose(r1.coef_debiased, r4.coef_debiased)
    np.testing.assert_allclose(r1.se, r4.se)
