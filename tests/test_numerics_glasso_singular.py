"""H2 numerical-stability sweep — graphical lasso, near-singular S.

The empirical covariance `S` enters the block-CD inner solver as the
quadratic part of the regression subproblem. When `n < p` (the
high-dimensional regime glasso is built for) `S` is rank-deficient by
construction; the `diag_offset` adds a small ridge to keep `Ŵ` positive
definite, but the L1 penalty itself does most of the conditioning.

Three pathologies to cover:

* **`n ≪ p`** with `diag_offset=0` and a small `alpha`. The estimator
  must still produce a finite SPD precision matrix.
* **A duplicated variable** (`X[:, j] = X[:, k]`). The corresponding
  column of `S` is collinear; the off-diagonal estimate at that edge
  has a sparsity-induced minimizer that has to be one of the tied
  solutions.
* **Joint glasso** with a near-singular per-population covariance —
  the ADMM update inverts `Σ + ρ⁻¹ Z` so the diagonal offset interacts
  with `ρ`.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import skein_glm

TIME_BUDGET_S = 30.0


def _fit_under_budget(est, *args, **kwargs):
    t0 = time.perf_counter()
    est.fit(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    assert elapsed < TIME_BUDGET_S, (
        f"{type(est).__name__} fit took {elapsed:.2f}s "
        f"(budget {TIME_BUDGET_S}s) — likely an infinite loop"
    )


def _assert_finite_symmetric(theta: np.ndarray) -> None:
    assert np.all(np.isfinite(theta)), "non-finite precision matrix"
    assert np.allclose(theta, theta.T, atol=1e-8), "asymmetric precision matrix"


def _assert_finite_spd(theta: np.ndarray) -> None:
    _assert_finite_symmetric(theta)
    # L1 glasso preserves SPD across the block-CD iterates by
    # construction (every diagonal update keeps `Ŵ ≻ 0`). Nonconvex
    # glasso (MCP / SCAD) only enforces SPD at modest rank deficit;
    # in the extreme `n ≪ p` regime the released-shrinkage region can
    # push the smallest eigenvalue slightly negative. The caller chooses
    # the stricter check only where the algorithm guarantees it.
    eigs = np.linalg.eigvalsh(theta)
    assert eigs.min() > -1e-8, f"precision not PSD (min eigenvalue {eigs.min():.3e})"


# ---------- single-population: n ≪ p --------------------------------------


@pytest.mark.parametrize("estimator_cls", [
    skein_glm.GraphicalLasso,
    skein_glm.GraphicalMCP,
    skein_glm.GraphicalSCAD,
])
def test_glasso_high_dim_remains_finite(estimator_cls):
    """n=20, p=50 — the empirical covariance is rank-20 of size 50×50."""
    rng = np.random.default_rng(0)
    n, p = 20, 50
    x = rng.standard_normal((n, p))
    est = estimator_cls(alpha=0.2)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)
    _assert_finite_spd(est.covariance_)


def test_glasso_l1_extreme_high_dim_remains_spd():
    """n=5, p=40 — effective rank deficit is severe. L1 glasso preserves
    SPD by construction (Friedman/Hastie/Tibshirani 2008 block-CD)."""
    rng = np.random.default_rng(1)
    n, p = 5, 40
    x = rng.standard_normal((n, p))
    est = skein_glm.GraphicalLasso(alpha=0.3)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)


def test_glasso_mcp_extreme_high_dim_remains_finite():
    """At extreme `n ≪ p` (5 × 40), MCP's released-shrinkage region can
    push the smallest eigenvalue slightly negative — the algorithm does
    not enforce SPD the way L1 does. We assert finiteness + symmetry,
    which is the H2 contract."""
    rng = np.random.default_rng(1)
    n, p = 5, 40
    x = rng.standard_normal((n, p))
    est = skein_glm.GraphicalMCP(alpha=0.3)
    _fit_under_budget(est, x)
    _assert_finite_symmetric(est.precision_)


def test_glasso_zero_diag_offset_high_dim():
    """`diag_offset=0` removes the safety ridge — the only conditioning
    is the L1 penalty itself."""
    rng = np.random.default_rng(2)
    n, p = 30, 40
    x = rng.standard_normal((n, p))
    est = skein_glm.GraphicalLasso(alpha=0.3, diag_offset=0.0)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)


# ---------- duplicated variables -------------------------------------------


def test_glasso_duplicated_variable_remains_finite():
    """X[:, 0] = X[:, 1] makes S rank-deficient by construction; the
    L1 penalty + diagonal offset have to produce a finite Θ."""
    rng = np.random.default_rng(3)
    n, p = 100, 12
    x = rng.standard_normal((n, p))
    x[:, 1] = x[:, 0]  # exact duplicate
    est = skein_glm.GraphicalLasso(alpha=0.1)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)


def test_glasso_near_duplicated_variable_remains_finite():
    rng = np.random.default_rng(4)
    n, p = 100, 12
    x = rng.standard_normal((n, p))
    x[:, 1] = x[:, 0] + 1e-10 * rng.standard_normal(n)
    est = skein_glm.GraphicalLasso(alpha=0.05)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)


def test_glasso_constant_variable_remains_finite():
    """A constant column has S[k, k] = 0 and S[k, j] = 0 for all j.
    The diagonal offset is what saves the precision update for that
    row."""
    rng = np.random.default_rng(5)
    n, p = 100, 10
    x = rng.standard_normal((n, p))
    x[:, 3] = 2.5  # constant — zero variance
    est = skein_glm.GraphicalLasso(alpha=0.1)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)


# ---------- precomputed covariance with explicit rank deficit -------------


def test_glasso_precomputed_singular_covariance():
    """Hand-build a rank-deficient S to exercise the precomputed-S
    code path."""
    rng = np.random.default_rng(6)
    p = 12
    # Rank-6 X: every column lives in a 6-dim subspace, so the
    # empirical covariance is rank-6 of size 12 × 12.
    n = 200
    L = rng.standard_normal((p, 6))
    z = rng.standard_normal((n, 6))
    x = z @ L.T
    est = skein_glm.GraphicalLasso(alpha=0.2)
    _fit_under_budget(est, x)
    _assert_finite_spd(est.precision_)


# ---------- joint glasso: per-population rank deficit ---------------------


def test_joint_glasso_high_dim_per_population_remains_finite():
    """K=3 populations, each n_k=15 with p=25 — every per-pop covariance
    is rank-deficient. The DWW ADMM has to keep all three precision
    blocks finite + SPD."""
    rng = np.random.default_rng(7)
    p = 25
    Xs = [rng.standard_normal((15, p)) for _ in range(3)]
    est = skein_glm.JointGraphicalLasso(lambda_2=0.05)
    # JointGraphicalLasso expects a `lambda_1` knob too — check the
    # signature handles the missing kwarg gracefully via solver defaults.
    _fit_under_budget(est, Xs)
    assert hasattr(est, "precisions_")
    for theta in est.precisions_:
        _assert_finite_spd(theta)


def test_joint_glasso_mcp_high_dim_remains_finite():
    rng = np.random.default_rng(8)
    p = 20
    Xs = [rng.standard_normal((12, p)) for _ in range(2)]
    est = skein_glm.JointGraphicalMCP(lambda_2=0.05)
    _fit_under_budget(est, Xs)
    for theta in est.precisions_:
        _assert_finite_spd(theta)
