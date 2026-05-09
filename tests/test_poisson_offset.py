"""Tests for Poisson offsets (M3.7 / glmnet parity)."""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _rate_problem(seed: int = 0, n: int = 300, p: int = 6):
    """Sparse-truth Poisson rate problem with per-sample exposure.
    `y_i ~ Poisson(exposure_i · exp(X_i β))` and `offset_i = log(exposure_i)`."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 0.5
    true_beta[2] = -0.4
    exposure = rng.uniform(0.5, 2.5, n)
    rate = exposure * np.exp(np.clip(x @ true_beta, -3, 3))
    y = rng.poisson(rate).astype(np.float64)
    offset = np.log(exposure)
    return x, y, offset, true_beta


def test_offset_zeros_matches_no_offset_path():
    """Offset of all-zeros must produce identical β to no offset."""
    x, y, _, _ = _rate_problem(0, n=120)
    m1 = skein_glm.PoissonMCPPathRegressor(
        gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2,
    ).fit(x, y)
    m2 = skein_glm.PoissonMCPPathRegressor(
        gamma=3.0, lambdas=m1.lambdas_, offset=np.zeros(x.shape[0]),
    ).fit(x, y)
    np.testing.assert_allclose(m1.coefs_, m2.coefs_, atol=1e-9)
    np.testing.assert_allclose(m1.intercepts_, m2.intercepts_, atol=1e-9)


def test_offset_shifts_intercept_when_constant():
    """Constant offset c is statistically equivalent to a fixed shift in
    the intercept: β_off (with offset c) and β_no (no offset) should
    differ only in intercept by c (`α_off + c = α_no`), since the
    likelihood depends on η_full = X·β + offset (or = X·β + α_no)."""
    x, y, _, _ = _rate_problem(1, n=200)
    c = 0.5
    m_off = skein_glm.PoissonMCPRegressor(
        lambda_=0.01, gamma=3.0, offset=np.full(x.shape[0], c),
    ).fit(x, y)
    m_no = skein_glm.PoissonMCPRegressor(
        lambda_=0.01, gamma=3.0,
    ).fit(x, y)
    # Coefficients identical; intercept shifted.
    np.testing.assert_allclose(m_off.coef_, m_no.coef_, atol=1e-7)
    assert abs((m_off.intercept_ + c) - m_no.intercept_) < 1e-7


def test_offset_path_shape_and_recovery():
    x, y, offset, true_beta = _rate_problem(2, n=400)
    m = skein_glm.PoissonMCPPathRegressor(
        gamma=3.0, n_lambdas=15, lambda_min_ratio=1e-3, offset=offset,
    ).fit(x, y)
    assert m.coefs_.shape == (15, 6)
    last = m.coefs_[-1]
    # Active features 0 and 2 should match in sign with true β.
    assert np.sign(last[0]) == np.sign(true_beta[0])
    assert np.sign(last[2]) == np.sign(true_beta[2])


def test_offset_validates_length():
    x, y, _, _ = _rate_problem(3, n=50)
    bad = np.zeros(x.shape[0] + 5)
    with pytest.raises(ValueError, match=r"offset length"):
        skein_glm.PoissonMCPPathRegressor(
            gamma=3.0, n_lambdas=4, offset=bad,
        ).fit(x, y)


def test_offset_validates_finite():
    x, y, _, _ = _rate_problem(4, n=40)
    bad = np.zeros(x.shape[0]); bad[0] = np.inf
    with pytest.raises(ValueError, match=r"offset must be finite"):
        skein_glm.PoissonMCPPathRegressor(
            gamma=3.0, n_lambdas=4, offset=bad,
        ).fit(x, y)


def test_offset_threads_through_group_lasso():
    x, y, offset, true_beta = _rate_problem(5, n=300)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    m = skein_glm.PoissonGroupLassoPathRegressor(
        groups=groups, n_lambdas=10, lambda_min_ratio=1e-2, offset=offset,
    ).fit(x, y)
    assert m.coefs_.shape == (10, 6)
    # Active groups 0 (features 0, 1) and 1 (feature 2) should have
    # nonzero magnitude at the smallest λ.
    last = m.coefs_[-1]
    assert np.linalg.norm(last[:2]) > 0.1
    assert np.linalg.norm(last[2:4]) > 0.1


def test_offset_threads_through_sparse_group_mcp():
    x, y, offset, _ = _rate_problem(6, n=300)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    m = skein_glm.PoissonSparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=8, lambda_min_ratio=1e-2,
        offset=offset,
    ).fit(x, y)
    assert m.coefs_.shape == (8, 6)


def test_offset_pathcv_picks_lambda():
    x, y, offset, true_beta = _rate_problem(7, n=400)
    cv = skein_glm.PoissonMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=12, lambda_min_ratio=1e-2,
        offset=offset,
    ).fit(x, y)
    assert cv.coef_.shape == (6,)
    assert cv.lambda_best_ in cv.lambdas_
    # Active features should match in sign.
    assert np.sign(cv.coef_[0]) == np.sign(true_beta[0])
    assert np.sign(cv.coef_[2]) == np.sign(true_beta[2])


def test_offset_pathcv_slices_per_fold():
    """The CV fit must slice offset to match each fold's train indices.
    Mismatch length would error inside the underlying path solver."""
    x, y, offset, _ = _rate_problem(8, n=80)
    cv = skein_glm.PoissonMCPPathCV(
        gamma=3.0, cv=3, random_state=0, n_lambdas=6, lambda_min_ratio=1e-2,
        offset=offset,
    )
    # No raise; CV completes.
    cv.fit(x, y)
    assert cv.coef_.shape == (x.shape[1],)


def test_offset_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y, offset, _ = _rate_problem(9, n=80)
    x_csc = sparse.csc_matrix(x)
    m_d = skein_glm.PoissonMCPPathRegressor(
        gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2, offset=offset,
    ).fit(x, y)
    m_s = skein_glm.PoissonMCPPathRegressor(
        gamma=3.0, lambdas=m_d.lambdas_, offset=offset,
    ).fit(x_csc, y)
    np.testing.assert_allclose(m_d.coefs_, m_s.coefs_, atol=1e-6)
