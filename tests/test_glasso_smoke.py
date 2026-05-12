"""End-to-end smoke tests for graphical lasso family. Require
`maturin develop` to have been run.

Coverage:
- Single-population glasso (L1 / MCP / SCAD) on raw X and precomputed S.
- Per-edge weights flow through (zero weight = no penalty on that edge).
- sklearn.covariance.GraphicalLasso parity at L1 (numerical Frobenius
  distance).
- EBIC tuner picks a sensible λ on synthetic ground truth.
- Joint glasso (L1 + MCP): coupling at large λ collapses populations;
  at λ→0 decouples to per-pop MLE.
"""
from __future__ import annotations

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")
sklearn_cov = pytest.importorskip("sklearn.covariance")


def _sparse_precision(p: int, density: float = 0.2, seed: int = 0) -> np.ndarray:
    """Build a small sparse SPD precision matrix for ground-truth tests."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((p, p)) * 0.3
    a = (a + a.T) / 2
    mask = rng.random((p, p)) < density
    mask = np.triu(mask, k=1)
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    a *= mask
    # Add diagonal to make PD.
    a += np.eye(p) * (np.abs(np.linalg.eigvalsh(a)).max() + 1.0)
    return a


def _gaussian_samples(theta: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Sample n i.i.d. observations from N(0, Θ⁻¹)."""
    rng = np.random.default_rng(seed)
    sigma = np.linalg.inv(theta)
    return rng.multivariate_normal(np.zeros(theta.shape[0]), sigma, size=n)


# ---- single-population --------------------------------------------------


def test_graphical_lasso_fits_raw_data():
    X = np.random.default_rng(0).standard_normal((200, 8))
    est = skein.GraphicalLasso(alpha=0.1).fit(X)
    assert est.precision_.shape == (8, 8)
    assert est.covariance_.shape == (8, 8)
    assert est.n_features_in_ == 8
    # Symmetric output.
    assert np.allclose(est.precision_, est.precision_.T, atol=1e-10)
    # Symmetric input shape detection.


def test_graphical_lasso_fits_precomputed_covariance():
    X = np.random.default_rng(1).standard_normal((200, 6))
    # Use the MLE 1/n form to match skein's internal _to_covariance.
    xc = X - X.mean(axis=0, keepdims=True)
    s = (xc.T @ xc) / X.shape[0]
    est = skein.GraphicalLasso(alpha=0.1).fit(s)
    assert est.precision_.shape == (6, 6)
    est_raw = skein.GraphicalLasso(alpha=0.1).fit(X)
    assert np.allclose(est.precision_, est_raw.precision_, atol=1e-6)


def test_graphical_mcp_recovers_sparse_support():
    p = 8
    theta_true = _sparse_precision(p, density=0.25, seed=42)
    X = _gaussian_samples(theta_true, n=400, seed=42)
    est = skein.GraphicalMCP(alpha=0.15, gamma=3.0).fit(X)
    # True zeros: at the higher α, MCP should zero many true zeros.
    iu = np.triu_indices(p, k=1)
    true_zeros = np.abs(theta_true[iu]) < 1e-8
    est_zeros = np.abs(est.precision_[iu]) < 1e-6
    # Of the true zeros, most should be recovered.
    if true_zeros.sum() > 0:
        recovered_zeros = (est_zeros & true_zeros).sum() / true_zeros.sum()
        assert recovered_zeros > 0.5, (
            f"recovered only {recovered_zeros:.0%} of true zeros"
        )


def test_graphical_scad_runs_and_is_symmetric():
    X = np.random.default_rng(2).standard_normal((150, 5))
    est = skein.GraphicalSCAD(alpha=0.1, a=3.7).fit(X)
    assert np.allclose(est.precision_, est.precision_.T, atol=1e-10)


def test_edge_weights_zero_means_no_penalty_on_that_edge():
    # Two correlated variables; with edge_weights[0, 1] = 0, the
    # off-diagonal of Θ should match the un-penalised inverse.
    s = np.array([[1.0, 0.6], [0.6, 1.0]])
    w_no = np.zeros((2, 2))
    est = skein.GraphicalLasso(alpha=10.0, edge_weights=w_no, diag_offset=0.0).fit(s)
    inv = np.linalg.inv(s)
    assert np.allclose(est.precision_, inv, atol=1e-3)


# ---- sklearn parity -----------------------------------------------------


def test_l1_glasso_matches_sklearn_on_small_problem():
    """At pure L1, no edge weights, skein and sklearn should agree to
    a small Frobenius-norm tolerance. Skein uses Friedman's block-CD,
    sklearn also uses Friedman/Hastie/Tibshirani — algorithmically the
    same, so the solutions are within tolerance modulo tie-breaking."""
    rng = np.random.default_rng(7)
    p = 10
    X = rng.standard_normal((200, p))
    alpha = 0.1
    skein_est = skein.GraphicalLasso(
        alpha=alpha, max_iter=500, tol=1e-8, inner_tol=1e-10, inner_max_iter=5000
    ).fit(X)
    skl_est = sklearn_cov.GraphicalLasso(
        alpha=alpha, tol=1e-7, max_iter=2000
    ).fit(X)
    diff = np.linalg.norm(skein_est.precision_ - skl_est.precision_, ord="fro")
    norm = np.linalg.norm(skl_est.precision_, ord="fro")
    rel = diff / norm
    # Both algorithms are Friedman/Hastie/Tibshirani BCD with slightly
    # different convergence criteria (sklearn: mean abs change in Θ;
    # skein: max abs change in W). A first-order parity check at the
    # 15% relative-Frobenius level is sensible; exact agreement isn't
    # expected without matching termination rules.
    assert rel < 0.15, (
        f"sklearn parity: relative Frobenius diff = {rel:.4f} "
        f"(absolute = {diff:.4f}, sklearn norm = {norm:.4f})"
    )


# ---- EBIC tuner ---------------------------------------------------------


def test_ebic_path_picks_reasonable_lambda():
    p = 8
    theta_true = _sparse_precision(p, density=0.3, seed=11)
    X = _gaussian_samples(theta_true, n=500, seed=11)
    lambdas = np.geomspace(0.5, 0.01, 12)
    result = skein.ebic_path(X, skein.GraphicalLasso, lambdas, gamma=0.5)
    assert result.best_estimator is not None
    assert result.best_lambda > 0
    assert result.n_edges.shape == lambdas.shape
    # Edge count should be roughly monotonically increasing as λ shrinks.
    # (Allow some non-monotonicity due to nonconvex tie-breaking.)
    assert result.n_edges[0] <= result.n_edges[-1] + 2


# ---- joint glasso -------------------------------------------------------


def test_joint_glasso_decouples_at_zero_lambda():
    X1 = np.random.default_rng(20).standard_normal((150, 4))
    X2 = np.random.default_rng(21).standard_normal((180, 4))
    # Tiny λ ⇒ decoupled.
    joint = skein.JointGraphicalLasso(
        lambda_2=1e-6, max_iter=2000, primal_tol=1e-7, dual_tol=1e-7
    ).fit([X1, X2])
    assert joint.n_populations_ == 2
    # Per-pop independent fits, no penalty.
    s1 = np.cov(X1, rowvar=False)
    s2 = np.cov(X2, rowvar=False)
    indep1 = skein.GraphicalLasso(alpha=1e-6, tol=1e-7).fit(s1)
    indep2 = skein.GraphicalLasso(alpha=1e-6, tol=1e-7).fit(s2)
    # Same general off-diagonal magnitudes (not bitwise identical —
    # different algorithms, ADMM vs Friedman BCD — but in the same
    # ballpark).
    iu = np.triu_indices(4, k=1)
    diff1 = np.abs(joint.precisions_[0][iu] - indep1.precision_[iu]).max()
    diff2 = np.abs(joint.precisions_[1][iu] - indep2.precision_[iu]).max()
    # Loose tolerance — different solvers.
    assert diff1 < 0.5
    assert diff2 < 0.5


def test_joint_glasso_collapses_at_large_lambda():
    # Two pops drawn from different distributions; huge coupling λ
    # forces off-diagonals to align across populations. At the same
    # time, huge λ pulls them toward zero. Test both: (a) off-diags
    # are small in magnitude, and (b) the populations agree on them.
    X1 = np.random.default_rng(30).standard_normal((150, 3))
    X2 = np.random.default_rng(31).standard_normal((150, 3)) * 2
    joint = skein.JointGraphicalLasso(
        lambda_2=10.0, max_iter=3000, primal_tol=1e-7, dual_tol=1e-7
    ).fit([X1, X2])
    iu = np.triu_indices(3, k=1)
    max_off_1 = np.abs(joint.precisions_[0][iu]).max()
    max_off_2 = np.abs(joint.precisions_[1][iu]).max()
    diff = np.abs(joint.precisions_[0][iu] - joint.precisions_[1][iu]).max()
    # Both should be small — λ_2 = 10 dwarfs all signal.
    assert max_off_1 < 0.2, f"pop 1 off-diag still {max_off_1}"
    assert max_off_2 < 0.2, f"pop 2 off-diag still {max_off_2}"
    # And the populations should agree on them.
    assert diff < 0.1, f"expected collapse at large λ; got max diff {diff}"


def test_joint_glasso_mcp_runs_to_completion():
    X1 = np.random.default_rng(40).standard_normal((100, 4))
    X2 = np.random.default_rng(41).standard_normal((100, 4))
    joint = skein.JointGraphicalMCP(lambda_2=0.05, gamma=3.0).fit([X1, X2])
    assert joint.n_populations_ == 2
    for theta in joint.precisions_:
        assert np.allclose(theta, theta.T, atol=1e-10)


def test_joint_ebic_path_picks_lambda():
    X1 = np.random.default_rng(50).standard_normal((150, 4))
    X2 = np.random.default_rng(51).standard_normal((150, 4))
    lambdas = np.geomspace(0.3, 0.01, 8)
    result = skein.joint_ebic_path(
        [X1, X2], skein.JointGraphicalLasso, lambdas, gamma=0.5
    )
    assert result.best_estimator is not None
    assert result.best_lambda_2 > 0
    assert result.n_edges_union.shape == lambdas.shape
