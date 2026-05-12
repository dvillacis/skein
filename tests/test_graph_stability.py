"""Tests for `GraphicalStabilitySelection` and `GraphicalBootstrap`.

Coverage:
- Single-population MB stability selection: shape, monotonicity in
  threshold, signal recovery on a known sparse precision matrix.
- Joint MB stability selection: per-population stable edges, shapes.
- Non-parametric bootstrap: CI ordering, edge selection probability
  bounds, signal recovery, reproducibility.
- Parameter validation.
- Reproducibility (fixed `random_state`).
- Sparse / covariance-input rejection.
"""
from __future__ import annotations

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")


def _sparse_precision(p: int, density: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((p, p)) * 0.3
    a = (a + a.T) / 2
    mask = rng.random((p, p)) < density
    mask = np.triu(mask, k=1)
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    a *= mask
    a += np.eye(p) * (np.abs(np.linalg.eigvalsh(a)).max() + 1.0)
    return a


def _gaussian_samples(theta: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.linalg.inv(theta)
    return rng.multivariate_normal(np.zeros(theta.shape[0]), sigma, size=n)


# ---- single-population stability selection -------------------------


def test_graphical_stability_single_shapes_and_symmetry():
    p = 6
    X = np.random.default_rng(0).standard_normal((150, p))
    lambdas = np.geomspace(0.5, 0.05, 5)
    ss = skein.GraphicalStabilitySelection(
        base_estimator=skein.GraphicalLasso(),
        lambdas=lambdas,
        n_bootstraps=10,
        threshold=0.6,
        random_state=0,
    ).fit(X)
    assert ss.selection_probabilities_.shape == (5, p, p)
    assert ss.max_probabilities_.shape == (p, p)
    # Symmetric (i,j == j,i) — graphical models are by construction.
    assert np.allclose(
        ss.max_probabilities_, ss.max_probabilities_.T, atol=1e-12
    )
    # Diagonal is zero (edges are off-diagonal).
    assert np.all(np.diag(ss.max_probabilities_) == 0.0)
    assert ss.stable_edges_.ndim == 2 and ss.stable_edges_.shape[1] == 2
    # All "stable" pairs should be strictly upper triangular and pass threshold.
    for i, j in ss.stable_edges_:
        assert i < j
        assert ss.max_probabilities_[i, j] >= 0.6
    assert ss.n_features_in_ == p


def test_graphical_stability_single_recovers_sparse_support():
    """On a problem with strong sparse signal, stability selection
    should mark the true non-zero edges as stable far more than the
    true zeros."""
    p = 8
    theta_true = _sparse_precision(p, density=0.25, seed=42)
    X = _gaussian_samples(theta_true, n=400, seed=42)
    lambdas = np.geomspace(0.4, 0.04, 6)
    ss = skein.GraphicalStabilitySelection(
        base_estimator=skein.GraphicalMCP(gamma=3.0),
        lambdas=lambdas,
        n_bootstraps=40,
        sample_fraction=0.5,
        threshold=0.6,
        random_state=0,
    ).fit(X)
    iu = np.triu_indices(p, k=1)
    true_nonzero = np.abs(theta_true[iu]) > 1e-8
    max_prob_iu = ss.max_probabilities_[iu]
    if true_nonzero.sum() > 0:
        # True non-zeros should have higher avg max-prob than true zeros.
        assert max_prob_iu[true_nonzero].mean() > max_prob_iu[~true_nonzero].mean()


def test_graphical_stability_threshold_monotonicity():
    """Higher threshold ⇒ fewer (or equal) stable edges."""
    X = np.random.default_rng(3).standard_normal((100, 5))
    lambdas = np.geomspace(0.4, 0.05, 4)
    common = dict(
        base_estimator=skein.GraphicalLasso(),
        lambdas=lambdas,
        n_bootstraps=15,
        random_state=0,
    )
    ss_low = skein.GraphicalStabilitySelection(threshold=0.51, **common).fit(X)
    ss_high = skein.GraphicalStabilitySelection(threshold=0.9, **common).fit(X)
    assert ss_high.stable_edges_.shape[0] <= ss_low.stable_edges_.shape[0]


def test_graphical_stability_reproducible_with_fixed_seed():
    X = np.random.default_rng(5).standard_normal((100, 4))
    lambdas = np.geomspace(0.3, 0.05, 3)
    kw = dict(
        base_estimator=skein.GraphicalLasso(),
        lambdas=lambdas,
        n_bootstraps=8,
        random_state=123,
    )
    a = skein.GraphicalStabilitySelection(**kw).fit(X)
    b = skein.GraphicalStabilitySelection(**kw).fit(X)
    np.testing.assert_array_equal(
        a.selection_probabilities_, b.selection_probabilities_
    )


def test_graphical_stability_njobs_consistency():
    """Parallel and serial agree byte-for-byte at fixed seed."""
    pytest.importorskip("joblib")
    X = np.random.default_rng(6).standard_normal((80, 4))
    lambdas = np.geomspace(0.3, 0.05, 3)
    common = dict(
        base_estimator=skein.GraphicalLasso(),
        lambdas=lambdas,
        n_bootstraps=6,
        random_state=42,
    )
    a = skein.GraphicalStabilitySelection(n_jobs=None, **common).fit(X)
    b = skein.GraphicalStabilitySelection(n_jobs=2, **common).fit(X)
    np.testing.assert_array_equal(
        a.selection_probabilities_, b.selection_probabilities_
    )


# ---- joint stability selection -------------------------------------


def test_graphical_stability_joint_shapes():
    p = 5
    X1 = np.random.default_rng(10).standard_normal((100, p))
    X2 = np.random.default_rng(11).standard_normal((120, p))
    lambdas = np.geomspace(0.4, 0.05, 4)
    ss = skein.GraphicalStabilitySelection(
        base_estimator=skein.JointGraphicalLasso(),
        lambdas=lambdas,
        n_bootstraps=6,
        threshold=0.55,
        random_state=0,
    ).fit([X1, X2])
    assert ss.selection_probabilities_.shape == (4, 2, p, p)
    assert ss.max_probabilities_.shape == (2, p, p)
    assert ss.n_populations_ == 2
    assert isinstance(ss.stable_edges_, list)
    assert len(ss.stable_edges_) == 2
    for arr in ss.stable_edges_:
        assert arr.ndim == 2 and arr.shape[1] == 2


def test_graphical_stability_joint_mcp_runs():
    p = 4
    X1 = np.random.default_rng(20).standard_normal((80, p))
    X2 = np.random.default_rng(21).standard_normal((90, p))
    ss = skein.GraphicalStabilitySelection(
        base_estimator=skein.JointGraphicalMCP(gamma=3.0),
        lambdas=np.geomspace(0.3, 0.05, 3),
        n_bootstraps=4,
        random_state=0,
    ).fit([X1, X2])
    assert ss.n_populations_ == 2


# ---- non-parametric bootstrap --------------------------------------


def test_graphical_bootstrap_single_shapes_and_ci_ordering():
    p = 6
    X = np.random.default_rng(0).standard_normal((150, p))
    bs = skein.GraphicalBootstrap(
        base_estimator=skein.GraphicalLasso(alpha=0.1),
        n_bootstraps=20,
        alpha=0.05,
        random_state=0,
    ).fit(X)
    assert bs.precisions_.shape == (20, p, p)
    for name in ("mean_", "std_", "ci_lower_", "ci_upper_",
                "edge_selection_probabilities_"):
        assert getattr(bs, name).shape == (p, p)
    # ci_lower <= mean <= ci_upper element-wise.
    assert np.all(bs.ci_lower_ <= bs.mean_ + 1e-12)
    assert np.all(bs.mean_ <= bs.ci_upper_ + 1e-12)
    # Selection probability in [0, 1], diagonal zero.
    assert np.all(bs.edge_selection_probabilities_ >= 0.0)
    assert np.all(bs.edge_selection_probabilities_ <= 1.0)
    assert np.all(np.diag(bs.edge_selection_probabilities_) == 0.0)


def test_graphical_bootstrap_signal_recovery():
    """True non-zero edges should have higher bootstrap selection
    probability than true zero edges."""
    p = 8
    theta_true = _sparse_precision(p, density=0.3, seed=99)
    X = _gaussian_samples(theta_true, n=400, seed=99)
    bs = skein.GraphicalBootstrap(
        base_estimator=skein.GraphicalLasso(alpha=0.1),
        n_bootstraps=40,
        random_state=0,
    ).fit(X)
    iu = np.triu_indices(p, k=1)
    sel = bs.edge_selection_probabilities_[iu]
    nz = np.abs(theta_true[iu]) > 1e-8
    if nz.sum() > 0 and (~nz).sum() > 0:
        assert sel[nz].mean() > sel[~nz].mean()


def test_graphical_bootstrap_joint_shapes():
    p = 4
    X1 = np.random.default_rng(30).standard_normal((80, p))
    X2 = np.random.default_rng(31).standard_normal((90, p))
    bs = skein.GraphicalBootstrap(
        base_estimator=skein.JointGraphicalLasso(lambda_2=0.1),
        n_bootstraps=8,
        random_state=0,
    ).fit([X1, X2])
    assert bs.precisions_.shape == (8, 2, p, p)
    for name in ("mean_", "std_", "ci_lower_", "ci_upper_",
                "edge_selection_probabilities_"):
        assert getattr(bs, name).shape == (2, p, p)
    assert bs.n_populations_ == 2


def test_graphical_bootstrap_reproducible():
    X = np.random.default_rng(50).standard_normal((100, 4))
    kw = dict(
        base_estimator=skein.GraphicalLasso(alpha=0.1),
        n_bootstraps=8,
        random_state=7,
    )
    a = skein.GraphicalBootstrap(**kw).fit(X)
    b = skein.GraphicalBootstrap(**kw).fit(X)
    np.testing.assert_array_equal(a.precisions_, b.precisions_)


# ---- validation ----------------------------------------------------


def test_stability_rejects_precomputed_covariance():
    s = np.eye(4)
    with pytest.raises(ValueError, match="precomputed covariance"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.GraphicalLasso(),
            lambdas=np.array([0.1]),
            n_bootstraps=2,
        ).fit(s)


def test_bootstrap_rejects_precomputed_covariance():
    s = np.eye(4)
    with pytest.raises(ValueError, match="precomputed covariance"):
        skein.GraphicalBootstrap(
            base_estimator=skein.GraphicalLasso(alpha=0.1),
            n_bootstraps=2,
        ).fit(s)


def test_stability_validates_params():
    X = np.random.default_rng(0).standard_normal((50, 3))
    # threshold ≤ 0.5
    with pytest.raises(ValueError, match="threshold"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.GraphicalLasso(),
            lambdas=np.array([0.1]),
            threshold=0.5,
        ).fit(X)
    # sample_fraction out of range
    with pytest.raises(ValueError, match="sample_fraction"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.GraphicalLasso(),
            lambdas=np.array([0.1]),
            sample_fraction=1.0,
        ).fit(X)
    # n_bootstraps < 1
    with pytest.raises(ValueError, match="n_bootstraps"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.GraphicalLasso(),
            lambdas=np.array([0.1]),
            n_bootstraps=0,
        ).fit(X)
    # empty lambdas
    with pytest.raises(ValueError, match="lambdas"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.GraphicalLasso(),
            lambdas=np.array([]),
        ).fit(X)
    # non-positive lambdas
    with pytest.raises(ValueError, match="lambdas"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.GraphicalLasso(),
            lambdas=np.array([0.1, 0.0]),
        ).fit(X)


def test_bootstrap_validates_params():
    X = np.random.default_rng(0).standard_normal((50, 3))
    with pytest.raises(ValueError, match="alpha"):
        skein.GraphicalBootstrap(
            base_estimator=skein.GraphicalLasso(alpha=0.1),
            alpha=0.0,
        ).fit(X)
    with pytest.raises(ValueError, match="n_bootstraps"):
        skein.GraphicalBootstrap(
            base_estimator=skein.GraphicalLasso(alpha=0.1),
            n_bootstraps=1,
        ).fit(X)


def test_stability_joint_requires_list():
    X = np.random.default_rng(0).standard_normal((50, 4))
    with pytest.raises(ValueError, match="list"):
        skein.GraphicalStabilitySelection(
            base_estimator=skein.JointGraphicalLasso(),
            lambdas=np.array([0.1]),
            n_bootstraps=2,
        ).fit(X)
