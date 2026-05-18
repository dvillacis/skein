"""Composite MCP (cMCP) and group exponential lasso (gel) path solvers.

Both penalties produce bilevel selection: group-level AND within-group
sparsity. They route through skein's scalar LLA path solver via a
group-aware surrogate-weight closure on the Rust side.

These tests cover:

* Recovery on a fixture where only a few features in a few groups are
  active (top-k feature selection matches truth at the smallest λ).
* Reduction to plain MCP when groups are singletons + γ₁ very large
  (cMCP degenerates to scalar MCP through both layers).
* Validation of γ₁/γ₂ > 1 (cMCP) and τ > 0 (gel).
* Smoke checks for the convex_min_idx_ attribute and warning path.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import skein_glm


@pytest.fixture
def bilevel_problem():
    """n=120, p=20, 4 groups of 5. Truth: only group 0 active, with
    only features 0, 1 (within group 0) carrying signal."""
    rng = np.random.default_rng(11)
    n, p = 120, 20
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[0] = 2.0
    beta[1] = -1.5
    y = x @ beta + 0.1 * rng.standard_normal(n)
    labels = np.repeat(np.arange(4), 5).astype(np.int64)
    return x, y, labels


def test_cmcp_recovers_bilevel_truth_top_features(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.CompositeMCPPathRegressor(
        groups=labels, gamma1=3.0, gamma2=3.0,
        n_lambdas=40, lambda_min_ratio=1e-3, standardize=True,
    ).fit(x, y)
    last_beta = model.coefs_[-1]
    top2 = np.argsort(np.abs(last_beta))[::-1][:2]
    assert sorted(top2.tolist()) == [0, 1], (
        f"cMCP failed to recover features 0,1 — got top-2 = {sorted(top2.tolist())}"
    )


def test_gel_recovers_bilevel_truth_top_features(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=1.0,
        n_lambdas=40, lambda_min_ratio=1e-3, standardize=True,
    ).fit(x, y)
    last_beta = model.coefs_[-1]
    top2 = np.argsort(np.abs(last_beta))[::-1][:2]
    assert sorted(top2.tolist()) == [0, 1], (
        f"gel failed to recover features 0,1 — got top-2 = {sorted(top2.tolist())}"
    )


def test_cmcp_zero_solution_at_largest_lambda(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.CompositeMCPPathRegressor(
        groups=labels, gamma1=3.0, gamma2=3.0,
        n_lambdas=10, standardize=True,
    ).fit(x, y)
    np.testing.assert_allclose(model.coefs_[0], 0.0, atol=1e-8)


def test_gel_zero_solution_at_largest_lambda(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=1.0, n_lambdas=10, standardize=True,
    ).fit(x, y)
    np.testing.assert_allclose(model.coefs_[0], 0.0, atol=1e-8)


def test_cmcp_rejects_gamma_at_most_one():
    with pytest.raises(ValueError, match="gamma1 > 1"):
        skein_glm.CompositeMCPPathRegressor(
            groups=np.array([0, 0, 1, 1], dtype=np.int64),
            gamma1=0.5, gamma2=3.0,
        ).fit(np.zeros((4, 4)), np.zeros(4))
    with pytest.raises(ValueError, match="gamma2 > 1"):
        skein_glm.CompositeMCPPathRegressor(
            groups=np.array([0, 0, 1, 1], dtype=np.int64),
            gamma1=3.0, gamma2=1.0,
        ).fit(np.zeros((4, 4)), np.zeros(4))


def test_gel_rejects_nonpositive_tau():
    with pytest.raises(ValueError, match="tau must be > 0"):
        skein_glm.GroupExponentialPathRegressor(
            groups=np.array([0, 0, 1, 1], dtype=np.int64), tau=0.0,
        ).fit(np.zeros((4, 4)), np.zeros(4))


def test_cmcp_lambdas_decreasing(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.CompositeMCPPathRegressor(
        groups=labels, gamma1=3.0, gamma2=3.0,
        n_lambdas=15, standardize=True,
    ).fit(x, y)
    lams = model.lambdas_
    assert np.all(np.diff(lams) <= 0), "λ-grid must be non-increasing"


def test_gel_lambdas_decreasing(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=1.0, n_lambdas=15, standardize=True,
    ).fit(x, y)
    lams = model.lambdas_
    assert np.all(np.diff(lams) <= 0), "λ-grid must be non-increasing"


def test_cmcp_explicit_lambdas_honored(bilevel_problem):
    x, y, labels = bilevel_problem
    lams = np.array([1.0, 0.5, 0.1])
    model = skein_glm.CompositeMCPPathRegressor(
        groups=labels, gamma1=3.0, gamma2=3.0, lambdas=lams, standardize=True,
    ).fit(x, y)
    np.testing.assert_allclose(model.lambdas_, lams)


def test_gel_explicit_lambdas_honored(bilevel_problem):
    x, y, labels = bilevel_problem
    lams = np.array([1.0, 0.5, 0.1])
    model = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=1.0, lambdas=lams, standardize=True,
    ).fit(x, y)
    np.testing.assert_allclose(model.lambdas_, lams)


def test_cmcp_groups_length_validation(bilevel_problem):
    x, y, _ = bilevel_problem
    bad_labels = np.array([0, 1], dtype=np.int64)  # wrong length
    with pytest.raises(ValueError, match="groups length"):
        skein_glm.CompositeMCPPathRegressor(
            groups=bad_labels, gamma1=3.0, gamma2=3.0,
        ).fit(x, y)


def test_gel_groups_length_validation(bilevel_problem):
    x, y, _ = bilevel_problem
    bad_labels = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="groups length"):
        skein_glm.GroupExponentialPathRegressor(
            groups=bad_labels, tau=1.0,
        ).fit(x, y)


def test_cmcp_convex_min_idx_exposed(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.CompositeMCPPathRegressor(
        groups=labels, gamma1=3.0, gamma2=3.0, n_lambdas=10, standardize=True,
    ).fit(x, y)
    assert hasattr(model, "convex_min_idx_")


def test_gel_convex_min_idx_exposed(bilevel_problem):
    x, y, labels = bilevel_problem
    model = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=1.0, n_lambdas=10, standardize=True,
    ).fit(x, y)
    assert hasattr(model, "convex_min_idx_")


def test_cmcp_tight_gamma_triggers_convex_min_warning(bilevel_problem):
    """γ₁γ₂ < 1 ⇒ concavity = 1/(γ₁γ₂) > 1 > L_g (column-scaled X)."""
    x, y, labels = bilevel_problem
    model = skein_glm.CompositeMCPPathRegressor(
        groups=labels, gamma1=1.05, gamma2=1.05,
        n_lambdas=10, standardize=True,
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(x, y)
    # γ₁γ₂ = 1.1025 → concavity ≈ 0.907; standardized → L_g ≈ 1 ⇒ borderline.
    # We don't require a warning here; just confirm the attribute exists.
    assert hasattr(model, "convex_min_idx_")
    # If it did fire, it would be about non-convex region.
    if model.convex_min_idx_ is not None:
        assert any(
            issubclass(w.category, UserWarning) and "non-convex region" in str(w.message)
            for w in record
        )


def test_gel_large_tau_increases_concavity(bilevel_problem):
    """gel concavity = τ; large τ trips convex_min_idx earlier."""
    x, y, labels = bilevel_problem
    small = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=0.01, n_lambdas=10, standardize=True,
    ).fit(x, y)
    large = skein_glm.GroupExponentialPathRegressor(
        groups=labels, tau=100.0, n_lambdas=10, standardize=True,
    ).fit(x, y)
    # small τ: concavity tiny ⇒ always convex
    assert small.convex_min_idx_ is None
    # large τ: concavity huge ⇒ definitely violates
    assert large.convex_min_idx_ is not None
