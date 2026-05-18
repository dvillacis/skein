"""Convex-region detection for nonconvex path regressors (grpreg `convex.min`).

Each nonconvex *PathRegressor* gains a `convex_min_idx_` attribute populated
after `fit`: the smallest λ-index along the path at which the local objective
ceases to be locally convex, or ``None`` when the whole path stays convex.
A ``UserWarning`` fires once when the boundary is crossed.

The boundary condition is closed-form once columns are scaled to unit norm
(so per-coordinate / per-group Lipschitz collapses to ≈ 1.0):

* scalar MCP:  convex ⟺ ``1 ≥ 1/γ`` ⟺ ``γ ≥ 1``
* scalar SCAD: convex ⟺ ``1 ≥ 1/(a-1)`` ⟺ ``a ≥ 2``

These tests exercise both regimes and verify the warning surfaces only when
the path is detected as non-convex.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import skein_glm
from skein_glm import _core


@pytest.fixture
def regression_problem():
    rng = np.random.default_rng(7)
    n, p = 80, 30
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:5] = [2.0, -1.5, 1.0, -1.2, 0.8]
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return x, y


def _count_user_warnings(record, substring: str) -> int:
    return sum(
        1
        for w in record
        if issubclass(w.category, UserWarning) and substring in str(w.message)
    )


def test_mcp_path_tight_gamma_enters_nonconvex_region(regression_problem):
    x, y = regression_problem
    model = skein_glm.MCPPathRegressor(gamma=1.1, n_lambdas=30, standardize=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(x, y)
    assert model.convex_min_idx_ is not None
    assert _count_user_warnings(record, "non-convex region") == 1


def test_mcp_path_loose_gamma_stays_convex(regression_problem):
    x, y = regression_problem
    model = skein_glm.MCPPathRegressor(gamma=1e6, n_lambdas=30, standardize=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(x, y)
    assert model.convex_min_idx_ is None
    assert _count_user_warnings(record, "non-convex region") == 0


def test_scad_path_concavity_matches_mcp_boundary(regression_problem):
    """SCAD a=2.1 has concavity 1/(2.1-1) ≈ 1/1.1 — same as MCP γ=1.1."""
    x, y = regression_problem
    mcp = skein_glm.MCPPathRegressor(gamma=1.1, n_lambdas=30, standardize=True).fit(x, y)
    scad = skein_glm.SCADPathRegressor(a=2.1, n_lambdas=30, standardize=True).fit(x, y)
    assert mcp.convex_min_idx_ == scad.convex_min_idx_


def test_scad_path_loose_a_stays_convex(regression_problem):
    x, y = regression_problem
    model = skein_glm.SCADPathRegressor(a=37.0, n_lambdas=30, standardize=True).fit(x, y)
    assert model.convex_min_idx_ is None


def test_group_mcp_path_tight_gamma_triggers(regression_problem):
    """For groups of 5 with column-scaled X, ‖X_g‖_op²/n ≈ 1; γ < 1 ⇒ violator."""
    x, y = regression_problem
    groups = np.repeat(np.arange(x.shape[1] // 5), 5).astype(np.int64)
    model = skein_glm.GroupMCPPathRegressor(
        groups=groups, gamma=0.7, n_lambdas=30, standardize=True,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(x, y)
    assert model.convex_min_idx_ is not None
    assert _count_user_warnings(record, "non-convex region") == 1


def test_group_mcp_path_loose_gamma_stays_convex(regression_problem):
    x, y = regression_problem
    groups = np.repeat(np.arange(x.shape[1] // 5), 5).astype(np.int64)
    model = skein_glm.GroupMCPPathRegressor(
        groups=groups, gamma=100.0, n_lambdas=30, standardize=True,
    ).fit(x, y)
    assert model.convex_min_idx_ is None


def test_group_scad_path_loose_a_stays_convex(regression_problem):
    x, y = regression_problem
    groups = np.repeat(np.arange(x.shape[1] // 5), 5).astype(np.int64)
    model = skein_glm.GroupSCADPathRegressor(
        groups=groups, a=37.0, n_lambdas=30, standardize=True,
    ).fit(x, y)
    assert model.convex_min_idx_ is None


def test_sparse_group_mcp_path_exposes_attribute(regression_problem):
    x, y = regression_problem
    groups = np.repeat(np.arange(x.shape[1] // 5), 5).astype(np.int64)
    model = skein_glm.SparseGroupMCPPathRegressor(
        groups=groups, gamma=3.0, alpha=0.5, n_lambdas=30, standardize=True,
    ).fit(x, y)
    assert hasattr(model, "convex_min_idx_")


def test_sparse_group_scad_path_exposes_attribute(regression_problem):
    x, y = regression_problem
    groups = np.repeat(np.arange(x.shape[1] // 5), 5).astype(np.int64)
    model = skein_glm.SparseGroupSCADPathRegressor(
        groups=groups, a=3.7, alpha=0.5, n_lambdas=30, standardize=True,
    ).fit(x, y)
    assert hasattr(model, "convex_min_idx_")


def test_core_binding_scalar_returns_index():
    """Direct check of the Rust binding without the estimator wrapper."""
    betas = np.array([[1.0, 1e-10], [1.0, 0.3]])
    col_lip = np.array([2.0, 0.5])
    assert _core.convex_min_idx_scalar(betas, col_lip, 1.0, 1e-8) == 1


def test_core_binding_scalar_returns_none_for_convex_penalty():
    betas = np.array([[1.0, 0.3]])
    col_lip = np.array([2.0, 0.5])
    assert _core.convex_min_idx_scalar(betas, col_lip, 0.0, 1e-8) is None


def test_core_binding_group_returns_index():
    betas = np.array([[1.0, -0.5, 1e-12, 0.0], [1.0, -0.5, 0.4, 0.2]])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    glip = np.array([2.0, 0.5])
    assert _core.convex_min_idx_group(betas, labels, glip, 1.0, 1e-8) == 1


def test_core_binding_group_lipschitz_dense_matches_naive():
    """Rust power-iteration approximates the exact operator norm at ~1e-3."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 6))
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    lip = _core.group_lipschitz_dense(x, labels)
    n = x.shape[0]
    expected = np.array(
        [np.linalg.norm(x[:, :3], 2) ** 2 / n, np.linalg.norm(x[:, 3:], 2) ** 2 / n]
    )
    np.testing.assert_allclose(lip, expected, rtol=1e-3)
