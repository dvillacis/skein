"""Smoke + parity tests for sparse-group SCAD estimators (M6.x).

The Rust surrogate (`surrogate_sparse_group_scad`) was already shipped in
M2.7 / M2.8; M6 wires it through to user-facing PyO3 entries and sklearn
estimators across all four datafits. The tests here exercise the full
stack — they don't re-test the surrogate math (covered by cargo tests
in `solver/lla.rs`).
"""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _ls_problem(seed: int = 0, n: int = 120, p: int = 6):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    y = x[:, 0] - 0.7 * x[:, 1] + 0.1 * rng.standard_normal(n)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    return x, y, groups


def _classification_y(y_ls):
    return (y_ls > 0).astype(np.float64)


def _poisson_y(rng, n, x):
    eta = 0.4 * x[:, 0] - 0.3 * x[:, 1]
    return rng.poisson(np.exp(np.clip(eta, -3, 3))).astype(np.float64)


# =========================================================================
# LS sparse-group SCAD
# =========================================================================


def test_sg_scad_ls_path_shape_and_lambda_descending():
    x, y, groups = _ls_problem(0)
    path = skein_glm.SparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=12, lambda_min_ratio=1e-2
    ).fit(x, y)
    assert path.coefs_.shape == (12, 6)
    assert path.lambdas_.shape == (12,)
    assert all(
        path.lambdas_[i] > path.lambdas_[i + 1] for i in range(len(path.lambdas_) - 1)
    )


def test_sg_scad_ls_recovers_active_groups():
    x, y, groups = _ls_problem(1, n=200)
    path = skein_glm.SparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=20, lambda_min_ratio=1e-3
    ).fit(x, y)
    last = path.coefs_[-1]
    # Group 0 (features 0, 1) should be the most active; groups 1, 2 dwarfed.
    g0_norm = np.linalg.norm(last[:2])
    g1_norm = np.linalg.norm(last[2:4])
    g2_norm = np.linalg.norm(last[4:6])
    assert g0_norm > 0.5
    assert g0_norm > 3 * max(g1_norm, g2_norm)


def test_sg_scad_ls_dense_sparse_equivalence():
    # After M14d (direct-CD on the nonconvex SCAD prox, replacing the LLA-
    # wrapped weighted-sparse-group-lasso surrogate), the dense and sparse
    # design paths can converge to slightly different stationary points of
    # the same SCAD objective. The dense path centers the response and
    # design; the sparse path appends a 1s intercept column. These two
    # equivalent reformulations produce identical iterates under LLA (each
    # outer iter solves a convex weighted-lasso subproblem with a unique
    # solution), but direct-CD on the multimodal SCAD prox is sensitive
    # to warm-start trajectory and can land on different valid local
    # minima. Supports still match; coefficient values differ by a few
    # percent at the tail of the path. SparseGroupMCP has the same
    # characteristic (max-diff ~4e-2 on this problem).
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y, groups = _ls_problem(2, n=80)
    x_csc = sparse.csc_matrix(x)
    path_d = skein_glm.SparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    path_s = skein_glm.SparseGroupSCADPathRegressor(
        groups=groups, lambdas=path_d.lambdas_,
    ).fit(x_csc, y)
    np.testing.assert_array_equal(path_d.coefs_ != 0, path_s.coefs_ != 0)
    np.testing.assert_allclose(path_d.coefs_, path_s.coefs_, atol=5e-2)


def test_sg_scad_ls_rejects_a_below_two():
    x, y, groups = _ls_problem(3, n=40)
    with pytest.raises(ValueError, match="must be > 2"):
        skein_glm.SparseGroupSCADRegressor(groups=groups, a=1.5).fit(x, y)


def test_sg_scad_ls_high_a_approximates_sparse_group_lasso():
    """At very large `a`, SCAD's surrogate weights stay close to the base
    weights for all `‖β_g‖_2 < a·λ`, so the path approximates plain
    sparse-group lasso. Compare on a shared λ-grid at small λ
    (where MCP/SCAD departs most from lasso)."""
    x, y, groups = _ls_problem(4, n=80)
    sgl = skein_glm.SparseGroupLassoPathRegressor(
        groups=groups, alpha=0.5, n_lambdas=6, lambda_min_ratio=1e-2
    ).fit(x, y)
    sg = skein_glm.SparseGroupSCADPathRegressor(
        groups=groups, alpha=0.5, lambdas=sgl.lambdas_, a=1e6,
    ).fit(x, y)
    np.testing.assert_allclose(sg.coefs_, sgl.coefs_, atol=1e-5)


def test_sg_scad_ls_path_cv_picks_reasonable_lambda():
    x, y, groups = _ls_problem(5, n=150)
    cv = skein_glm.SparseGroupSCADPathCV(
        groups=groups, cv=3, random_state=0, n_lambdas=12, lambda_min_ratio=1e-2
    ).fit(x, y)
    assert cv.coef_.shape == (6,)
    assert cv.lambda_best_ in cv.lambdas_


# =========================================================================
# Logistic / Poisson / Cox sparse-group SCAD smoke
# =========================================================================


def test_sg_scad_logistic_predict_proba_smoke():
    x, y_ls, groups = _ls_problem(10, n=100)
    y = _classification_y(y_ls)
    path = skein_glm.LogisticSparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    proba = path.predict_proba(x)
    assert proba.shape == (x.shape[0], 8)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_sg_scad_logistic_dense_sparse_equivalence():
    pytest.importorskip("scipy")
    from scipy import sparse

    x, y_ls, groups = _ls_problem(11, n=80)
    y = _classification_y(y_ls)
    x_csc = sparse.csc_matrix(x)
    path_d = skein_glm.LogisticSparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    path_s = skein_glm.LogisticSparseGroupSCADPathRegressor(
        groups=groups, lambdas=path_d.lambdas_
    ).fit(x_csc, y)
    np.testing.assert_allclose(path_d.coefs_, path_s.coefs_, atol=1e-6)


def test_sg_scad_poisson_smoke():
    rng = np.random.default_rng(20)
    n, p = 120, 6
    x = rng.standard_normal((n, p))
    y = _poisson_y(rng, n, x)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    path = skein_glm.PoissonSparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, y)
    assert path.coefs_.shape == (8, 6)


def test_sg_scad_cox_smoke():
    rng = np.random.default_rng(30)
    n, p = 120, 6
    x = rng.standard_normal((n, p))
    eta = 0.5 * x[:, 0] - 0.4 * x[:, 1]
    time = rng.exponential(1.0 / np.exp(np.clip(eta, -3, 3)))
    event = (rng.uniform(size=n) < 0.6).astype(np.float64)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    path = skein_glm.CoxSparseGroupSCADPathRegressor(
        groups=groups, n_lambdas=8, lambda_min_ratio=1e-2
    ).fit(x, time, event)
    assert path.coefs_.shape == (8, 6)
    risk = path.predict(x)
    assert risk.shape == (x.shape[0], 8)


def test_sg_scad_glm_families_reject_a_below_two():
    x, y_ls, groups = _ls_problem(40, n=40)
    with pytest.raises(ValueError, match="must be > 2"):
        skein_glm.LogisticSparseGroupSCADRegressor(
            groups=groups, a=2.0
        ).fit(x, _classification_y(y_ls))
    with pytest.raises(ValueError, match="must be > 2"):
        skein_glm.PoissonSparseGroupSCADRegressor(
            groups=groups, a=2.0
        ).fit(x, np.maximum(y_ls, 0))
