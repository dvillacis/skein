"""Tests for Cox PH Efron tie handling (M3.x / glmnet parity)."""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _problem_with_ties(seed: int = 0, n: int = 200, p: int = 6, n_unique: int = 8):
    """Cox problem with heavy ties — round times to a small set of unique
    values so each tie-block has multiple events."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    eta = 0.5 * x[:, 0] - 0.4 * x[:, 2]
    raw_time = rng.exponential(1.0 / np.exp(np.clip(eta, -3, 3)))
    # Bin into n_unique buckets to force ties.
    edges = np.linspace(0, raw_time.max() + 1e-9, n_unique + 1)
    time = edges[np.digitize(raw_time, edges) - 1] + 0.1
    event = (rng.uniform(size=n) < 0.7).astype(np.float64)
    return x, time, event


def _problem_no_ties(seed: int = 0, n: int = 100, p: int = 5):
    """Cox problem with all-unique times — Efron and Breslow should
    coincide."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    time = np.sort(rng.exponential(1.0, n)) + np.linspace(0, 1e-6, n)
    event = (rng.uniform(size=n) < 0.7).astype(np.float64)
    if event.sum() == 0:
        event[0] = 1.0
    return x, time, event


def test_cox_ties_unique_times_produces_identical_breslow_and_efron():
    x, time, event = _problem_no_ties(0)
    m_b = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="breslow", n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, time, event)
    m_e = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="efron", lambdas=m_b.lambdas_
    ).fit(x, time, event)
    np.testing.assert_allclose(m_b.coefs_, m_e.coefs_, atol=1e-9)


def test_cox_ties_heavy_ties_breslow_and_efron_diverge():
    x, time, event = _problem_with_ties(1, n_unique=6)
    m_b = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="breslow", n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, time, event)
    m_e = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="efron", lambdas=m_b.lambdas_
    ).fit(x, time, event)
    # Coefs should differ visibly when ties are heavy.
    assert np.max(np.abs(m_b.coefs_[-1] - m_e.coefs_[-1])) > 1e-3


def test_cox_default_ties_is_breslow():
    """`ties` keyword defaults to 'breslow' for backward compat."""
    x, time, event = _problem_no_ties(2)
    m_default = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, n_lambdas=6, lambda_min_ratio=1e-2
    ).fit(x, time, event)
    m_explicit_breslow = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="breslow", lambdas=m_default.lambdas_
    ).fit(x, time, event)
    np.testing.assert_allclose(m_default.coefs_, m_explicit_breslow.coefs_, atol=1e-12)


def test_cox_efron_threads_through_group_penalties():
    x, time, event = _problem_with_ties(3, n_unique=5)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    m = skein_glm.CoxGroupLassoPathRegressor(
        groups=groups, ties="efron", n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, time, event)
    assert m.coefs_.shape == (8, 6)


def test_cox_efron_threads_through_sparse_group_mcp():
    x, time, event = _problem_with_ties(4, n_unique=5)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    m = skein_glm.CoxSparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, ties="efron",
        n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, time, event)
    assert m.coefs_.shape == (8, 6)


def test_cox_efron_path_cv_picks_lambda():
    x, time, event = _problem_with_ties(5, n=300, n_unique=8)
    cv = skein_glm.CoxMCPPathCV(
        gamma=3.0, ties="efron", cv=3, random_state=0,
        n_lambdas=10, lambda_min_ratio=1e-2,
    ).fit(x, time, event)
    assert cv.coef_.shape == (6,)
    assert cv.lambda_best_ in cv.lambdas_


def test_cox_ties_validates_string():
    x, time, event = _problem_no_ties(6, n=40)
    with pytest.raises(ValueError, match=r"ties must be"):
        skein_glm.CoxMCPPathRegressor(
            gamma=3.0, ties="exact", n_lambdas=4
        ).fit(x, time, event)


def test_cox_efron_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, time, event = _problem_with_ties(7, n=80, n_unique=5)
    x_csc = sparse.csc_matrix(x)
    m_d = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="efron", n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, time, event)
    m_s = skein_glm.CoxMCPPathRegressor(
        gamma=3.0, ties="efron", lambdas=m_d.lambdas_,
    ).fit(x_csc, time, event)
    np.testing.assert_allclose(m_d.coefs_, m_s.coefs_, atol=1e-7)
