"""Tests for Meinshausen-Bühlmann stability selection (M5.x)."""

from __future__ import annotations

import numpy as np
import pytest

skein_glm = pytest.importorskip("skein_glm")


def _ls_problem(seed: int = 0, n: int = 200, p: int = 20):
    """Sparse-truth problem with up to 4 active features. Clamps the
    active indices to ``[0, p)`` so the same helper works for every
    problem size in the suite."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    candidates = [(0, 1.5), (3, -1.0), (7, 0.8), (12, -1.2)]
    active = np.array([j for j, _ in candidates if j < p])
    for j, v in candidates:
        if j < p:
            true_beta[j] = v
    y = x @ true_beta + 0.3 * rng.standard_normal(n)
    return x, y, active


def test_stability_recovers_active_features_at_high_threshold():
    """High threshold (0.95) should pin down the truly active set with
    few false positives on a clear signal. The MB pattern uses a
    moderate λ-path (no near-OLS tail) so that noise features stay
    inactive across bootstraps."""
    x, y, active = _ls_problem(0, n=300)
    ss = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=15, lambda_min_ratio=5e-2),
        n_bootstraps=60, threshold=0.95, random_state=0, n_jobs=1,
    )
    ss.fit(x, y)
    # All true actives must be in the stable set.
    assert set(active.tolist()).issubset(set(ss.stable_features_.tolist()))
    # At threshold 0.95 with a moderate λ-path the stable set should
    # be tight (well under half of 20 features).
    assert len(ss.stable_features_) <= 8


def test_stability_threshold_filters_monotonically():
    """Higher threshold ⇒ smaller (or equal) stable set."""
    x, y, _ = _ls_problem(1, n=200)
    ss_low = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=15, lambda_min_ratio=1e-2),
        n_bootstraps=30, threshold=0.55, random_state=0, n_jobs=1,
    ).fit(x, y)
    ss_high = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=15, lambda_min_ratio=1e-2),
        n_bootstraps=30, threshold=0.95, random_state=0, n_jobs=1,
    ).fit(x, y)
    assert len(ss_high.stable_features_) <= len(ss_low.stable_features_)


def test_stability_attribute_shapes():
    x, y, _ = _ls_problem(2, n=80, p=15)
    ss = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2),
        n_bootstraps=10, threshold=0.6, random_state=0, n_jobs=1,
    ).fit(x, y)
    assert ss.selection_probabilities_.shape == (10, 15)
    assert ss.max_probabilities_.shape == (15,)
    assert ss.lambdas_.shape == (10,)
    assert ss.n_features_in_ == 15
    assert ss.stable_features_.dtype == np.int64
    # All probabilities in [0, 1].
    assert (ss.selection_probabilities_ >= 0).all()
    assert (ss.selection_probabilities_ <= 1).all()


def test_stability_transform_subsets_columns():
    x, y, _ = _ls_problem(3, n=80, p=12)
    ss = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2),
        n_bootstraps=10, threshold=0.6, random_state=0, n_jobs=1,
    ).fit(x, y)
    x_sub = ss.transform(x)
    assert x_sub.shape == (x.shape[0], len(ss.stable_features_))
    np.testing.assert_array_equal(x_sub, x[:, ss.stable_features_])


def test_stability_reproducibility():
    """Same random_state ⇒ identical bootstrap indices ⇒ identical
    selection probabilities."""
    x, y, _ = _ls_problem(4, n=80)
    common = dict(
        n_bootstraps=15, threshold=0.6, random_state=42, n_jobs=1,
    )
    a = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2),
        **common,
    ).fit(x, y)
    b = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2),
        **common,
    ).fit(x, y)
    np.testing.assert_array_equal(a.stable_features_, b.stable_features_)
    np.testing.assert_allclose(a.selection_probabilities_, b.selection_probabilities_)


def test_stability_validates_threshold_above_half():
    x, y, _ = _ls_problem(5, n=40)
    base = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=4)
    with pytest.raises(ValueError, match="threshold must be in"):
        skein_glm.StabilitySelection(base, threshold=0.5, n_bootstraps=2).fit(x, y)
    with pytest.raises(ValueError, match="threshold must be in"):
        skein_glm.StabilitySelection(base, threshold=0.0, n_bootstraps=2).fit(x, y)


def test_stability_validates_sample_fraction():
    x, y, _ = _ls_problem(6, n=40)
    base = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=4)
    with pytest.raises(ValueError, match="sample_fraction must be in"):
        skein_glm.StabilitySelection(
            base, sample_fraction=1.0, n_bootstraps=2
        ).fit(x, y)


def test_stability_grouped_estimator_selects_at_group_level():
    """For a GroupLassoPathRegressor, every feature in the same group
    should share the same selection probability."""
    x, y, _ = _ls_problem(7, n=120, p=12)
    groups = np.repeat(np.arange(6), 2).astype(np.int64)  # 6 groups of 2
    ss = skein_glm.StabilitySelection(
        skein_glm.GroupLassoPathRegressor(
            groups=groups, n_lambdas=10, lambda_min_ratio=1e-2,
        ),
        n_bootstraps=15, threshold=0.7, random_state=0, n_jobs=1,
    ).fit(x, y)
    # Within each group, the per-feature selection probabilities should
    # be identical (group-level decision).
    for g in range(6):
        in_group = np.where(groups == g)[0]
        probs_g = ss.selection_probabilities_[:, in_group]
        for k in range(probs_g.shape[0]):
            assert np.allclose(probs_g[k], probs_g[k, 0])


def test_stability_cox_outcomes_as_tuple():
    """Cox base estimator is detected by the `ties` attribute; outcomes
    must be passed as ``y=(time, event)``."""
    rng = np.random.default_rng(8)
    n, p = 120, 15
    x = rng.standard_normal((n, p))
    time = rng.exponential(1.0, n)
    event = (rng.uniform(size=n) < 0.7).astype(np.float64)
    ss = skein_glm.StabilitySelection(
        skein_glm.CoxMCPPathRegressor(gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2),
        n_bootstraps=10, threshold=0.6, random_state=0, n_jobs=1,
    ).fit(x, (time, event))
    assert ss.selection_probabilities_.shape == (8, p)
    assert ss.n_features_in_ == p


def test_stability_n_jobs_consistency():
    """n_jobs > 1 should produce the same probabilities as n_jobs=1
    (the bootstrap indices are pre-generated, so parallel execution is
    deterministic for a fixed random_state)."""
    x, y, _ = _ls_problem(9, n=100, p=10)
    common = dict(
        base_estimator=skein_glm.MCPPathRegressor(
            gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2,
        ),
        n_bootstraps=15,
        threshold=0.6,
        random_state=0,
    )
    a = skein_glm.StabilitySelection(n_jobs=1, **common).fit(x, y)
    b = skein_glm.StabilitySelection(n_jobs=2, **common).fit(x, y)
    np.testing.assert_allclose(a.selection_probabilities_, b.selection_probabilities_)
    np.testing.assert_array_equal(a.stable_features_, b.stable_features_)


def test_stability_with_sparse_input():
    pytest.importorskip("scipy")
    from scipy import sparse
    x, y, _ = _ls_problem(10, n=80, p=12)
    x_csc = sparse.csc_matrix(x)
    ss = skein_glm.StabilitySelection(
        skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2),
        n_bootstraps=10, threshold=0.6, random_state=0, n_jobs=1,
    ).fit(x_csc, y)
    assert ss.selection_probabilities_.shape == (8, 12)
    # transform should also work on sparse input.
    sub = ss.transform(x_csc)
    assert sub.shape == (x_csc.shape[0], len(ss.stable_features_))
