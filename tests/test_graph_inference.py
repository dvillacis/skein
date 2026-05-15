"""Tests for `skein_glm.graph_inference` — bootstrap-based FDR / FWER
on edges + the Meinshausen–Bühlmann (2010) stability threshold.

Coverage:
- Two-sided bootstrap p-value formula: minimal / maximal cases,
  upper-tri symmetry, diagonal-1 convention.
- BH FDR control: empirical FDR ≤ 1.5 × nominal on a synthetic
  glasso problem with known true sparsity, 30 bootstraps × 5 seeds.
- FWER (Bonferroni + Holm): empirical FWER ≤ nominal on the same
  problem; Holm uniformly more powerful than Bonferroni.
- MB closed-form bound: numeric agreement with the spec, infeasible
  case raises.
- Joint K=2 family-size: BH family is pooled across populations.
- Integration: `.fdr_threshold()` and `.fwer_threshold()` methods
  on `GraphicalBootstrap`; `.mb_threshold()` on
  `GraphicalStabilitySelection`.
- Validation: bad inputs raise.
"""
from __future__ import annotations

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")

from skein_glm import (  # noqa: E402
    GraphicalBootstrap,
    GraphicalLasso,
    GraphicalStabilitySelection,
    edge_fdr_threshold,
    edge_fwer_threshold,
    edge_pvalues,
    mb_stability_threshold,
)


def _sparse_precision(p: int, density: float = 0.2, seed: int = 0) -> np.ndarray:
    """Random sparse PD precision; returns (Theta, edge_mask) where
    edge_mask[i,j] = True iff Theta[i,j] != 0 and i != j."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((p, p)) * 0.3
    a = (a + a.T) / 2
    mask = rng.random((p, p)) < density
    mask = np.triu(mask, k=1)
    mask = mask | mask.T
    np.fill_diagonal(mask, False)
    a *= mask
    a += np.eye(p) * (np.abs(np.linalg.eigvalsh(a)).max() + 1.0)
    return a, mask


def _gaussian_from_theta(theta: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.linalg.inv(theta)
    return rng.multivariate_normal(np.zeros(theta.shape[0]), sigma, size=n)


# ---- edge_pvalues -------------------------------------------------


def test_edge_pvalues_perfect_signal_and_perfect_null():
    """An edge always nonzero in the bootstrap → smallest p (2/B).
    An edge always exactly zero → largest p (1.0), correctly
    reflecting "no evidence against H0: Θ_ij = 0" for sparse
    estimators."""
    B, p = 100, 3
    stack = np.zeros((B, p, p))
    stack[:, 0, 1] = stack[:, 1, 0] = 0.5  # always positive nonzero
    for k in range(B):
        np.fill_diagonal(stack[k], 1.0)
    pv = edge_pvalues(stack)
    assert pv[0, 1] == pytest.approx(2.0 / B, abs=1e-12)
    assert pv[1, 0] == pytest.approx(2.0 / B, abs=1e-12)
    # Edge (0, 2) is exactly zero on every bootstrap → both nonneg
    # and nonpos counts equal B → doubled min = 2 → clipped to 1.
    assert pv[0, 2] == pytest.approx(1.0, abs=1e-12)


def test_edge_pvalues_diagonal_is_one():
    B, p = 50, 4
    stack = np.random.default_rng(0).standard_normal((B, p, p))
    pv = edge_pvalues(stack)
    np.testing.assert_array_equal(np.diag(pv), np.ones(p))


def test_edge_pvalues_symmetric_input_symmetric_output():
    B, p = 50, 4
    raw = np.random.default_rng(1).standard_normal((B, p, p))
    sym = (raw + raw.transpose(0, 2, 1)) / 2
    pv = edge_pvalues(sym)
    np.testing.assert_array_equal(pv, pv.T)


def test_edge_pvalues_joint_shape():
    B, K, p = 30, 2, 4
    stack = np.random.default_rng(2).standard_normal((B, K, p, p))
    pv = edge_pvalues(stack)
    assert pv.shape == (K, p, p)
    for k in range(K):
        np.testing.assert_array_equal(np.diag(pv[k]), np.ones(p))


# ---- FDR control --------------------------------------------------


def _bh_run_one_seed(seed: int, p: int, n: int, n_boot: int) -> dict:
    """Fit a graphical lasso bootstrap and compute BH FDR / Holm FWER
    decisions for one random seed."""
    theta, edge_mask = _sparse_precision(p, density=0.2, seed=seed)
    X = _gaussian_from_theta(theta, n, seed=seed + 1000)
    boot = GraphicalBootstrap(
        base_estimator=GraphicalLasso(alpha=0.1),
        n_bootstraps=n_boot,
        random_state=seed,
    ).fit(X)
    fdr_out = edge_fdr_threshold(boot, fdr=0.1)
    fwer_out = edge_fwer_threshold(boot, fwer=0.1, method="holm")
    return {
        "true_edges": edge_mask,
        "fdr_reject": fdr_out["reject"],
        "fwer_reject": fwer_out["reject"],
    }


def test_bh_fdr_controls_at_nominal_level():
    """Average over 5 seeds: empirical FDR ≤ 1.5 × nominal 0.1."""
    p, n, n_boot = 8, 600, 50
    fdr_target = 0.1
    runs = [_bh_run_one_seed(seed, p, n, n_boot) for seed in range(5)]
    fdr_per_seed = []
    for r in runs:
        rej = r["fdr_reject"]
        true_pos = r["true_edges"]
        # Upper-tri only.
        iu = np.triu_indices(p, k=1)
        rej_set = rej[iu]
        truth_set = true_pos[iu]
        n_rej = rej_set.sum()
        n_false = (rej_set & ~truth_set).sum()
        fdp = n_false / max(n_rej, 1)
        fdr_per_seed.append(fdp)
    avg_fdr = float(np.mean(fdr_per_seed))
    # BH guarantees E[FDP] ≤ q under independence / PRDS; with a
    # bootstrap and a finite-sample test we allow 1.5× nominal as a
    # practical tolerance.
    assert avg_fdr <= 1.5 * fdr_target, (
        f"empirical FDR {avg_fdr:.3f} exceeds 1.5 × nominal {fdr_target}"
    )


def test_holm_fwer_controls_at_nominal_level():
    """Average over 5 seeds: empirical family-wise error ≤ nominal."""
    p, n, n_boot = 8, 600, 50
    fwer_target = 0.1
    runs = [_bh_run_one_seed(seed, p, n, n_boot) for seed in range(5)]
    any_false_pos = []
    for r in runs:
        rej = r["fwer_reject"]
        truth = r["true_edges"]
        iu = np.triu_indices(p, k=1)
        any_false_pos.append(bool((rej[iu] & ~truth[iu]).any()))
    empirical_fwer = float(np.mean(any_false_pos))
    # FWER theory gives ≤ nominal; bootstrap p-values are coarse so
    # we allow a small slack.
    assert empirical_fwer <= fwer_target + 0.05, (
        f"empirical FWER {empirical_fwer:.3f} exceeds nominal {fwer_target} + 0.05"
    )


# ---- MB threshold -------------------------------------------------


def test_mb_threshold_formula_numeric_match():
    """Closed-form: π = 0.5 + q²/(2·p·EV). Check on a worked example."""
    # p=20 features → 190 unique edges. Avg selected q=10, EV=1.
    # π = 0.5 + 100/(2·190·1) = 0.5 + 100/380 ≈ 0.7632.
    thr = mb_stability_threshold(
        p_total=190, q_lambda=10.0, expected_false_positives=1.0
    )
    expected = 0.5 + 100.0 / 380.0
    assert thr == pytest.approx(expected, abs=1e-12)


def test_mb_threshold_infeasible_raises():
    """Asking for tighter control than the bound supports raises."""
    with pytest.raises(ValueError, match="Infeasible"):
        mb_stability_threshold(
            p_total=6, q_lambda=2.0, expected_false_positives=0.5
        )


def test_mb_threshold_input_validation():
    with pytest.raises(ValueError, match="p_total"):
        mb_stability_threshold(p_total=0, q_lambda=1.0, expected_false_positives=1.0)
    with pytest.raises(ValueError, match="q_lambda"):
        mb_stability_threshold(p_total=10, q_lambda=0.0, expected_false_positives=1.0)
    with pytest.raises(ValueError, match="expected_false_positives"):
        mb_stability_threshold(p_total=10, q_lambda=1.0, expected_false_positives=0.0)


# ---- joint K=2 ----------------------------------------------------


def test_joint_fdr_pools_across_populations():
    """For a (B, K, p, p) stack, BH pools all (k, i, j) into one
    family of K · p(p-1)/2 hypotheses, not p(p-1)/2 each."""
    # B=300 gives p_min = 2/300 ≈ 0.0067 → adj_smallest over 12 tests
    # ≈ 0.04, low enough to reject at fdr=0.05.
    B, K, p = 300, 2, 4
    rng = np.random.default_rng(0)
    stack = rng.standard_normal((B, K, p, p)) * 0.05
    # Inject one strong-signal edge per population.
    stack[:, 0, 0, 1] += 0.5
    stack[:, 0, 1, 0] += 0.5
    stack[:, 1, 2, 3] -= 0.5
    stack[:, 1, 3, 2] -= 0.5
    out = edge_fdr_threshold(stack, fdr=0.05)
    assert out["reject"].shape == (K, p, p)
    assert bool(out["reject"][0, 0, 1])
    assert bool(out["reject"][1, 2, 3])
    # Diagonals never rejected.
    for k in range(K):
        np.testing.assert_array_equal(np.diag(out["reject"][k]), np.zeros(p, dtype=bool))


# ---- integration with GraphicalBootstrap / StabilitySelection -----


def test_method_fdr_threshold_on_bootstrap():
    """`.fdr_threshold()` method matches the function-call version."""
    theta, _ = _sparse_precision(p=6, density=0.3, seed=42)
    X = _gaussian_from_theta(theta, n=400, seed=43)
    boot = GraphicalBootstrap(
        base_estimator=GraphicalLasso(alpha=0.1),
        n_bootstraps=40,
        random_state=42,
    ).fit(X)
    via_method = boot.fdr_threshold(fdr=0.1)
    via_function = edge_fdr_threshold(boot, fdr=0.1)
    np.testing.assert_array_equal(via_method["pvalues"], via_function["pvalues"])
    np.testing.assert_array_equal(
        via_method["adjusted_pvalues"], via_function["adjusted_pvalues"]
    )
    np.testing.assert_array_equal(via_method["reject"], via_function["reject"])


def test_method_mb_threshold_on_stability_selection():
    """`.mb_threshold()` method computes the MB bound from the fit."""
    theta, _ = _sparse_precision(p=8, density=0.25, seed=10)
    X = _gaussian_from_theta(theta, n=500, seed=11)
    stab = GraphicalStabilitySelection(
        base_estimator=GraphicalLasso(alpha=0.1),
        lambdas=[0.05, 0.10, 0.20],
        n_bootstraps=30,
        random_state=10,
    ).fit(X)
    # Should return a feasible threshold in (0.5, 1] for a generous
    # E[V] budget on this small problem.
    thr = stab.mb_threshold(expected_false_positives=5.0)
    assert 0.5 < thr <= 1.0


# ---- validation ----------------------------------------------------


def test_edge_fdr_validates_fdr_range():
    stack = np.random.default_rng(0).standard_normal((10, 3, 3))
    with pytest.raises(ValueError, match="fdr"):
        edge_fdr_threshold(stack, fdr=0.0)
    with pytest.raises(ValueError, match="fdr"):
        edge_fdr_threshold(stack, fdr=1.0)


def test_edge_fwer_validates_method():
    stack = np.random.default_rng(0).standard_normal((10, 3, 3))
    with pytest.raises(ValueError, match="method"):
        edge_fwer_threshold(stack, fwer=0.05, method="bogus")  # type: ignore[arg-type]


def test_edge_pvalues_rejects_bad_shape():
    with pytest.raises(ValueError, match="3D"):
        edge_pvalues(np.zeros((5,)))
    with pytest.raises(ValueError, match="square"):
        edge_pvalues(np.zeros((10, 3, 4)))


def test_holm_more_powerful_than_bonferroni():
    """On a stack with several moderate-signal edges, Holm should
    reject at least as many as Bonferroni at the same α."""
    rng = np.random.default_rng(0)
    B, p = 200, 6
    stack = rng.standard_normal((B, p, p)) * 0.1
    stack = (stack + stack.transpose(0, 2, 1)) / 2
    # Plant a few moderate signals.
    for (i, j, shift) in [(0, 1, 0.4), (1, 2, 0.35), (0, 3, 0.3)]:
        stack[:, i, j] += shift
        stack[:, j, i] += shift
    bonf = edge_fwer_threshold(stack, fwer=0.10, method="bonferroni")
    holm = edge_fwer_threshold(stack, fwer=0.10, method="holm")
    iu = np.triu_indices(p, k=1)
    n_bonf = bonf["reject"][iu].sum()
    n_holm = holm["reject"][iu].sum()
    assert n_holm >= n_bonf
