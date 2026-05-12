"""Tests for the debiased / desparsified lasso (M5.x).

Anchored on Van de Geer–Bühlmann–Ritov (2014). The coverage test is
the load-bearing correctness check; everything else verifies
plumbing.
"""
from __future__ import annotations

import numpy as np
import pytest

from skein_glm import (
    DebiasedLassoRegressor,
    debiased_lasso,
)
from skein_glm.debiased import DebiasedLassoResult


def _make_sparse_problem(
    n: int, p: int, s: int, *, sigma: float = 0.5, rho: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthetic linear model with first ``s`` features active.

    Returns ``(X, y, beta_true, sigma)``. Design has Toeplitz
    correlation ``rho^|i-j|`` (``rho=0`` is iid Gaussian).
    """
    rng = np.random.default_rng(seed)
    if rho > 0:
        # Cholesky of AR(1) Σ — explicit so we don't depend on scipy.
        cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((n, p))
        X = Z @ L.T
    else:
        X = rng.standard_normal((n, p))
    beta = np.zeros(p, dtype=np.float64)
    beta[:s] = rng.uniform(0.8, 1.5, size=s) * rng.choice([-1.0, 1.0], size=s)
    y = X @ beta + sigma * rng.standard_normal(n)
    return X, y, beta, sigma


# --- plumbing / shapes ----------------------------------------------


def test_return_shapes_and_dataclass() -> None:
    X, y, _, _ = _make_sparse_problem(n=80, p=20, s=3, seed=0)
    res = debiased_lasso(X, y, n_jobs=1)
    assert isinstance(res, DebiasedLassoResult)
    p = X.shape[1]
    assert res.coef_debiased.shape == (p,)
    assert res.coef_lasso.shape == (p,)
    assert res.se.shape == (p,)
    assert res.ci_lower.shape == (p,)
    assert res.ci_upper.shape == (p,)
    assert res.pvalues.shape == (p,)
    assert res.z_scores.shape == (p,)
    assert res.Theta.shape == (p, p)
    assert res.lambda_nodewise.shape == (p,)
    assert isinstance(res.sigma_hat, float)
    assert isinstance(res.intercept_, float)
    assert isinstance(res.lambda_main, float)


def test_all_attributes_finite() -> None:
    X, y, _, _ = _make_sparse_problem(n=80, p=30, s=3, seed=1)
    res = debiased_lasso(X, y, n_jobs=1)
    for name in (
        "coef_debiased", "coef_lasso", "se", "ci_lower", "ci_upper",
        "pvalues", "z_scores", "Theta", "lambda_nodewise",
    ):
        arr = getattr(res, name)
        assert np.all(np.isfinite(arr)), f"non-finite values in {name}"
    assert np.isfinite(res.sigma_hat)
    assert np.isfinite(res.intercept_)


def test_ci_ordering() -> None:
    X, y, _, _ = _make_sparse_problem(n=100, p=30, s=4, seed=2)
    res = debiased_lasso(X, y, n_jobs=1)
    assert np.all(res.ci_lower <= res.coef_debiased + 1e-12)
    assert np.all(res.coef_debiased <= res.ci_upper + 1e-12)
    # Two-sided p-value sanity: pvalue ∈ [0, 1].
    assert np.all(res.pvalues >= 0.0)
    assert np.all(res.pvalues <= 1.0)


def test_se_nonnegative() -> None:
    X, y, _, _ = _make_sparse_problem(n=80, p=20, s=3, seed=3)
    res = debiased_lasso(X, y, n_jobs=1)
    assert np.all(res.se >= 0.0)


# --- correctness: VBR de-biasing direction --------------------------


def test_debiased_is_closer_to_truth_than_plain_lasso_on_active() -> None:
    """Lasso shrinks toward zero; debiased recovers most of the bias.

    On true-active features ``j < s``, the debiased estimate should
    be closer in absolute value to the truth than the plain lasso
    fit, averaged over coordinates."""
    n, p, s = 200, 80, 5
    X, y, beta, _ = _make_sparse_problem(n, p, s, sigma=0.3, seed=10)
    res = debiased_lasso(X, y, n_jobs=1)
    err_lasso = np.abs(res.coef_lasso[:s] - beta[:s])
    err_debiased = np.abs(res.coef_debiased[:s] - beta[:s])
    # Average L1 error should drop after debiasing.
    assert err_debiased.mean() < err_lasso.mean()


def test_pvalues_smaller_on_true_active() -> None:
    n, p, s = 200, 80, 5
    X, y, _, _ = _make_sparse_problem(n, p, s, sigma=0.3, seed=11)
    res = debiased_lasso(X, y, n_jobs=1)
    assert res.pvalues[:s].mean() < res.pvalues[s:].mean()
    # All true-active features should have p < 0.05 at this SNR.
    assert np.all(res.pvalues[:s] < 0.05)


# --- coverage simulation (load-bearing) -----------------------------


def test_empirical_coverage_of_95pct_cis() -> None:
    """Repeated-experiments coverage: across ``n_reps`` independent
    draws of `(X, y)` from the same DGP, the fraction of CIs that
    contain the true ``β_j`` should be close to nominal 95%.

    Allowing a 10-percentage-point slack on each side: theory is
    asymptotic and we are at moderate ``n``. A failure here means the
    se / Theta math is wrong, not just noisy."""
    rng_seed_base = 1000
    n, p, s = 150, 40, 3
    n_reps = 60

    truth = None
    covered = np.zeros(p, dtype=np.int64)

    for r in range(n_reps):
        X, y, beta, _ = _make_sparse_problem(
            n, p, s, sigma=0.5, seed=rng_seed_base + r,
        )
        if truth is None:
            truth = beta.copy()
        else:
            # Truth is drawn fresh every replication; we record
            # coverage relative to each replication's own truth so
            # the test is exact rather than approximate.
            truth = beta
        res = debiased_lasso(X, y, n_jobs=1)
        in_ci = (res.ci_lower <= truth) & (truth <= res.ci_upper)
        covered += in_ci.astype(np.int64)

    coverage = covered / n_reps
    # Aggregate coverage (mean over coordinates) should be close to
    # 0.95. We tolerate 0.85–1.0 to absorb finite-sample / finite-rep
    # noise. The signal we care about: NOT 0.6, NOT 0.4 — those would
    # indicate a wrong variance.
    overall = float(coverage.mean())
    assert 0.85 <= overall <= 1.0, (
        f"empirical 95% CI coverage {overall:.3f} outside [0.85, 1.0]; "
        f"per-coord coverage = {coverage}"
    )


def test_active_coordinate_coverage_above_chance() -> None:
    """Coverage on the *active* coordinates alone — these are the
    ones where the bias matters most. Should still be high."""
    rng_seed_base = 2000
    n, p, s = 150, 40, 3
    n_reps = 40

    cover_active = 0
    total_active = 0
    for r in range(n_reps):
        X, y, beta, _ = _make_sparse_problem(
            n, p, s, sigma=0.5, seed=rng_seed_base + r,
        )
        res = debiased_lasso(X, y, n_jobs=1)
        in_ci = (res.ci_lower <= beta) & (beta <= res.ci_upper)
        cover_active += int(in_ci[:s].sum())
        total_active += s
    rate = cover_active / total_active
    # On 3 active features × 40 reps = 120 trials, nominal 95% with
    # binomial SE ~2% — allow 80% as the floor.
    assert rate >= 0.80, f"active-feature coverage {rate:.3f} too low"


# --- robustness to options -----------------------------------------


def test_user_supplied_lambda_main() -> None:
    X, y, _, _ = _make_sparse_problem(n=100, p=25, s=3, seed=20)
    res = debiased_lasso(X, y, lambda_=0.1, n_jobs=1)
    assert res.lambda_main == 0.1


def test_user_supplied_lambda_nodewise_scalar() -> None:
    X, y, _, _ = _make_sparse_problem(n=100, p=25, s=3, seed=21)
    res = debiased_lasso(X, y, lambda_nodewise=0.05, n_jobs=1)
    assert np.allclose(res.lambda_nodewise, 0.05)


def test_user_supplied_lambda_nodewise_array() -> None:
    X, y, _, _ = _make_sparse_problem(n=100, p=15, s=3, seed=22)
    p = X.shape[1]
    lam_nw = np.linspace(0.02, 0.1, p)
    res = debiased_lasso(X, y, lambda_nodewise=lam_nw, n_jobs=1)
    assert np.allclose(res.lambda_nodewise, lam_nw)


def test_no_intercept() -> None:
    """fit_intercept=False on centered data — intercept must be 0."""
    rng = np.random.default_rng(30)
    n, p = 100, 15
    X = rng.standard_normal((n, p))
    X = X - X.mean(axis=0)
    beta = np.zeros(p)
    beta[:2] = [1.0, -1.0]
    y = X @ beta + 0.3 * rng.standard_normal(n)
    y = y - y.mean()
    res = debiased_lasso(X, y, fit_intercept=False, n_jobs=1)
    assert res.intercept_ == 0.0


def test_n_jobs_serial_parallel_agree() -> None:
    X, y, _, _ = _make_sparse_problem(n=80, p=20, s=3, seed=40)
    r1 = debiased_lasso(X, y, n_jobs=1)
    r4 = debiased_lasso(X, y, n_jobs=4)
    np.testing.assert_allclose(r1.coef_debiased, r4.coef_debiased)
    np.testing.assert_allclose(r1.se, r4.se)
    np.testing.assert_allclose(r1.Theta, r4.Theta)


# --- input validation ----------------------------------------------


def test_rejects_3d_X() -> None:
    X = np.zeros((10, 5, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="X must be 2D"):
        debiased_lasso(X, y)


def test_rejects_mismatched_y() -> None:
    X = np.zeros((10, 5))
    y = np.zeros(8)
    with pytest.raises(ValueError, match="y must be 1D with length 10"):
        debiased_lasso(X, y)


def test_rejects_alpha_out_of_range() -> None:
    X, y, _, _ = _make_sparse_problem(n=40, p=10, s=2, seed=50)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha must be in"):
            debiased_lasso(X, y, alpha=bad)


def test_rejects_bad_lambda_nodewise_shape() -> None:
    X, y, _, _ = _make_sparse_problem(n=40, p=10, s=2, seed=51)
    with pytest.raises(ValueError, match="lambda_nodewise must be scalar"):
        debiased_lasso(X, y, lambda_nodewise=np.array([0.1, 0.2]))


def test_rejects_nonpositive_lambda_nodewise() -> None:
    X, y, _, _ = _make_sparse_problem(n=40, p=10, s=2, seed=52)
    with pytest.raises(ValueError, match="lambda_nodewise entries"):
        debiased_lasso(X, y, lambda_nodewise=np.zeros(10))


def test_rejects_p_lt_2() -> None:
    X = np.random.default_rng(0).standard_normal((20, 1))
    y = X[:, 0] + 0.1 * np.random.default_rng(1).standard_normal(20)
    with pytest.raises(ValueError, match="p ≥ 2"):
        debiased_lasso(X, y)


# --- sklearn-style wrapper -----------------------------------------


def test_regressor_fit_predict_pipeline() -> None:
    X, y, _, _ = _make_sparse_problem(n=100, p=20, s=3, seed=60)
    est = DebiasedLassoRegressor(n_jobs=1).fit(X, y)
    # Required sklearn attributes.
    assert est.coef_.shape == (20,)
    assert isinstance(est.intercept_, float)
    assert est.n_features_in_ == 20
    # VBR-specific attributes.
    for name in (
        "se_", "ci_lower_", "ci_upper_", "pvalues_", "z_scores_",
        "Theta_", "coef_lasso_", "sigma_hat_", "lambda_main_",
        "lambda_nodewise_",
    ):
        assert hasattr(est, name), f"missing attribute {name}"
    # predict returns sensible shape.
    yhat = est.predict(X)
    assert yhat.shape == (100,)
    # Correlation with y should be positive on a real signal.
    assert np.corrcoef(yhat, y)[0, 1] > 0.5


def test_regressor_get_set_params_roundtrip() -> None:
    est = DebiasedLassoRegressor(alpha=0.10, n_jobs=2, fit_intercept=False)
    params = est.get_params()
    assert params["alpha"] == 0.10
    assert params["n_jobs"] == 2
    assert params["fit_intercept"] is False
    est2 = DebiasedLassoRegressor().set_params(**params)
    assert est2.alpha == 0.10


def test_regressor_matches_free_function() -> None:
    X, y, _, _ = _make_sparse_problem(n=80, p=15, s=3, seed=70)
    res = debiased_lasso(X, y, alpha=0.10, n_jobs=1)
    est = DebiasedLassoRegressor(alpha=0.10, n_jobs=1).fit(X, y)
    np.testing.assert_allclose(est.coef_, res.coef_debiased)
    np.testing.assert_allclose(est.se_, res.se)
    np.testing.assert_allclose(est.pvalues_, res.pvalues)
