"""Tests for plain GroupSCAD + AdaptiveGroupSCAD estimators."""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _problem(seed: int = 0, n: int = 200, p: int = 8):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -0.8
    true_beta[4] = 0.7
    true_beta[5] = -0.4
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return x, y, groups, true_beta


def test_group_scad_path_recovers_active_groups():
    x, y, groups, true_beta = _problem(0, n=300)
    m = skein_glm.GroupSCADPathRegressor(
        groups=groups, a=3.7, n_lambdas=15, lambda_min_ratio=1e-3
    ).fit(x, y)
    last = m.coefs_[-1]
    for j in [0, 1, 4, 5]:
        assert np.sign(last[j]) == np.sign(true_beta[j])
        assert abs(last[j]) > 0.3
    for j in [2, 3, 6, 7]:
        assert abs(last[j]) < 0.1


def test_group_scad_path_at_large_a_matches_group_lasso():
    """At very large `a`, SCAD's surrogate weights stay at the base
    weights everywhere, reducing to plain group lasso."""
    x, y, groups, _ = _problem(1, n=80)
    gl = skein_glm.GroupLassoPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, y)
    sc = skein_glm.GroupSCADPathRegressor(
        groups=groups, a=1e6, lambdas=gl.lambdas_,
    ).fit(x, y)
    np.testing.assert_allclose(sc.coefs_, gl.coefs_, atol=1e-5)


def test_group_scad_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse
    x, y, groups, _ = _problem(2, n=80)
    x_csc = sparse.csc_matrix(x)
    m_d = skein_glm.GroupSCADPathRegressor(
        groups=groups, a=3.7, n_lambdas=8, lambda_min_ratio=1e-2,
    ).fit(x, y)
    m_s = skein_glm.GroupSCADPathRegressor(
        groups=groups, a=3.7, lambdas=m_d.lambdas_,
    ).fit(x_csc, y)
    np.testing.assert_allclose(m_d.coefs_, m_s.coefs_, atol=1e-7)


def test_group_scad_rejects_a_below_two():
    x, y, groups, _ = _problem(3, n=40)
    with pytest.raises(ValueError, match="must be > 2"):
        skein_glm.GroupSCADRegressor(groups=groups, a=2.0).fit(x, y)


def test_group_scad_path_cv_picks_active_groups():
    x, y, groups, true_beta = _problem(4, n=200)
    cv = skein_glm.GroupSCADPathCV(
        groups=groups, a=3.7, cv=3, random_state=0, n_lambdas=10, lambda_min_ratio=1e-3,
    ).fit(x, y)
    assert cv.coef_.shape == (8,)
    assert cv.lambda_best_ in cv.lambdas_
    for j in [0, 1, 4, 5]:
        assert np.sign(cv.coef_[j]) == np.sign(true_beta[j])


def test_adaptive_group_scad_recovers_signal():
    x, y, groups, true_beta = _problem(5, n=300)
    m = skein_glm.AdaptiveGroupSCADPathRegressor(
        groups=groups, a=3.7, n_lambdas=15, lambda_min_ratio=1e-3,
    ).fit(x, y)
    last = m.coefs_[-1]
    for j in [0, 1, 4, 5]:
        assert abs(last[j]) > 0.3
    # Per-group adaptive weights: 4 entries, 2 small + 2 huge.
    assert m.weights_.shape == (4,)
    assert np.sum(m.weights_ > 1e3) >= 2


def test_adaptive_group_scad_cv():
    x, y, groups, _ = _problem(6, n=200)
    cv = skein_glm.AdaptiveGroupSCADPathCV(
        groups=groups, a=3.7, cv=3, random_state=0, n_lambdas=10, lambda_min_ratio=1e-2,
    ).fit(x, y)
    assert cv.coef_.shape == (8,)
    assert cv.weights_.shape == (4,)
