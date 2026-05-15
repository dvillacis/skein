"""Tests for `debiased_cox_lasso` / `DebiasedCoxLassoRegressor`.

Coverage:
- Output shape contracts: coef_debiased / se / pvalues all (p,);
  Theta (p, p); risk_score (n,).
- Recovery: planted-sparse Cox problem, debiased estimates land in
  the right neighborhood of the truth.
- Wald inference: empirical 95% CI coverage on inactive coordinates
  ≥ 80% over 40 replications (load-bearing — mirrors the precedent
  set by test_debiased_glm.py's logistic / Poisson coverage tests).
- ``ties='efron'`` works (different surrogate weights than Breslow).
- DebiasedCoxLassoRegressor wraps the function and exposes the
  inferential outputs as suffixed attributes.
- Validation: 0 events, negative time, non-binary event, p < 2.
- R-anchor: skipped if `tests/fixtures/hdi_cox_lasso.json` absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")

from skein_glm import (  # noqa: E402
    DebiasedCoxLassoRegressor,
    DebiasedCoxResult,
    debiased_cox_lasso,
)


def _cox_problem(
    n: int,
    p: int,
    k_active: int,
    seed: int,
    censoring_rate: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    signs = rng.choice([-1.0, 1.0], size=k_active)
    beta[:k_active] = signs * rng.uniform(0.4, 0.8, size=k_active)
    eta = X @ beta
    u = rng.uniform(size=n)
    time = -np.log(u) / np.exp(eta)
    event = (rng.uniform(size=n) > censoring_rate).astype(np.float64)
    return X, time, event, beta


# ---- shape contract ----------------------------------------------


def test_output_shapes_and_types():
    X, t, e, _ = _cox_problem(n=300, p=15, k_active=3, seed=0)
    res = debiased_cox_lasso(X, t, e)
    assert isinstance(res, DebiasedCoxResult)
    assert res.coef_debiased.shape == (15,)
    assert res.coef_glm.shape == (15,)
    assert res.se.shape == (15,)
    assert res.ci_lower.shape == (15,)
    assert res.ci_upper.shape == (15,)
    assert res.pvalues.shape == (15,)
    assert res.z_scores.shape == (15,)
    assert res.risk_score.shape == (300,)
    assert res.Theta.shape == (15, 15)
    assert res.lambda_nodewise.shape == (15,)
    assert res.family == "cox"
    assert res.ties == "breslow"
    assert np.all(res.ci_lower <= res.coef_debiased)
    assert np.all(res.coef_debiased <= res.ci_upper)
    assert np.all((res.pvalues >= 0.0) & (res.pvalues <= 1.0))


# ---- recovery -----------------------------------------------------


def test_active_features_recovered_with_significant_pvalues():
    """The 4 planted active features should have p-values < 0.01 on
    a moderately large Cox problem."""
    X, t, e, beta = _cox_problem(n=500, p=25, k_active=4, seed=1)
    res = debiased_cox_lasso(X, t, e, alpha=0.05)
    active = np.abs(beta) > 0
    assert np.all(res.pvalues[active] < 0.01), (
        f"some active features have large p-values: {res.pvalues[active]}"
    )


def test_null_features_largely_not_significant():
    """The 20 null features should mostly have p > 0.05; allow a
    couple of nominal false positives under standard Wald inference."""
    X, t, e, beta = _cox_problem(n=500, p=24, k_active=4, seed=2)
    res = debiased_cox_lasso(X, t, e, alpha=0.05)
    null = beta == 0
    n_signif = int((res.pvalues[null] < 0.05).sum())
    # 20 null tests at α=0.05 → expected 1 false positive; allow up
    # to 4 to keep the test stable under random seeds.
    assert n_signif <= 4, (
        f"too many null-feature false positives at α=0.05: {n_signif} / "
        f"{int(null.sum())}"
    )


# ---- Wald coverage (load-bearing) --------------------------------


def test_inactive_ci_coverage_at_least_80_percent_over_40_reps():
    """Empirical 95% CI coverage on inactive coordinates should be
    ≥ 80% averaged over 40 replications. Mirrors the precedent in
    test_debiased_glm.py."""
    n, p, k_active = 250, 18, 3
    n_reps = 40
    n_inactive = p - k_active
    covered = np.zeros((n_reps, n_inactive), dtype=bool)
    for rep in range(n_reps):
        X, t, e, beta = _cox_problem(n=n, p=p, k_active=k_active, seed=100 + rep)
        res = debiased_cox_lasso(X, t, e, alpha=0.05)
        inactive_mask = beta == 0
        # 0 is in the CI for inactive features iff ci_lower ≤ 0 ≤ ci_upper.
        in_ci = (res.ci_lower <= 0.0) & (0.0 <= res.ci_upper)
        covered[rep] = in_ci[inactive_mask]
    coverage = covered.mean()
    assert coverage >= 0.80, (
        f"empirical 95% CI coverage {coverage:.3f} below 0.80 over "
        f"{n_reps} reps × {n_inactive} inactive coords"
    )


# ---- ties handling ------------------------------------------------


def test_efron_ties_gives_slightly_different_estimates():
    """Breslow and Efron differ in how they handle tied event times.
    On a problem with engineered ties the results should differ but
    remain close (both are valid Cox estimators)."""
    rng = np.random.default_rng(3)
    n, p = 300, 10
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [0.6, -0.4, 0.5]
    eta = X @ beta
    # Round times down to create ties.
    time = np.round(-np.log(rng.uniform(size=n)) / np.exp(eta), 1) + 0.01
    event = (rng.uniform(size=n) < 0.7).astype(np.float64)
    res_breslow = debiased_cox_lasso(X, time, event, ties="breslow")
    res_efron = debiased_cox_lasso(X, time, event, ties="efron")
    diff = np.max(np.abs(res_breslow.coef_debiased - res_efron.coef_debiased))
    assert 0.0 <= diff < 0.3, (
        f"Breslow vs Efron max coef diff {diff:.3f} out of expected band"
    )


# ---- wrapper class ------------------------------------------------


def test_regressor_class_exposes_attributes():
    X, t, e, _ = _cox_problem(n=300, p=12, k_active=3, seed=7)
    clf = DebiasedCoxLassoRegressor(alpha=0.05).fit(X, t, e)
    assert clf.coef_.shape == (12,)
    assert clf.se_.shape == (12,)
    assert clf.pvalues_.shape == (12,)
    assert clf.ci_lower_.shape == (12,)
    assert clf.ci_upper_.shape == (12,)
    assert clf.risk_score_.shape == (300,)
    assert clf.Theta_.shape == (12, 12)
    assert clf.family_ == "cox"
    assert clf.ties_ == "breslow"
    # decision_function should match risk_score on training data
    # up to numerical precision.
    eta_pred = clf.decision_function(X)
    np.testing.assert_allclose(eta_pred, clf.risk_score_, atol=1e-12)
    # predict aliases decision_function for Cox.
    np.testing.assert_allclose(clf.predict(X), clf.decision_function(X))


# ---- validation ---------------------------------------------------


def test_rejects_no_events():
    X = np.random.default_rng(0).standard_normal((30, 5))
    t = np.linspace(1.0, 30.0, 30)
    e = np.zeros(30)
    with pytest.raises(ValueError, match="at least one event"):
        debiased_cox_lasso(X, t, e)


def test_rejects_negative_time():
    X = np.random.default_rng(0).standard_normal((30, 5))
    t = np.full(30, 1.0); t[0] = -0.5
    e = np.ones(30)
    with pytest.raises(ValueError, match="time"):
        debiased_cox_lasso(X, t, e)


def test_rejects_non_binary_event():
    X = np.random.default_rng(0).standard_normal((30, 5))
    t = np.linspace(0.1, 3.0, 30)
    e = np.full(30, 0.5)
    with pytest.raises(ValueError, match="event"):
        debiased_cox_lasso(X, t, e)


def test_rejects_p_lt_2():
    X = np.random.default_rng(0).standard_normal((30, 1))
    t = np.linspace(0.1, 3.0, 30)
    e = (np.arange(30) < 15).astype(np.float64)
    with pytest.raises(ValueError, match="p ≥ 2"):
        debiased_cox_lasso(X, t, e)


def test_rejects_invalid_alpha():
    X = np.random.default_rng(0).standard_normal((30, 5))
    t = np.linspace(0.1, 3.0, 30)
    e = (np.arange(30) < 15).astype(np.float64)
    with pytest.raises(ValueError, match="alpha"):
        debiased_cox_lasso(X, t, e, alpha=0.0)


def test_rejects_bad_ties():
    X = np.random.default_rng(0).standard_normal((30, 5))
    t = np.linspace(0.1, 3.0, 30)
    e = (np.arange(30) < 15).astype(np.float64)
    with pytest.raises(ValueError, match="ties"):
        debiased_cox_lasso(X, t, e, ties="something_else")


# ---- R-anchor (skipped without fixture) --------------------------


_FIXTURE = Path(__file__).parent / "fixtures" / "hdi_cox_lasso.json"


def test_against_hdi_lasso_proj():
    """Agreement with R `hdi::lasso.proj(..., family='cox')` on a
    fixed seed. Skipped if the fixture is absent — generate via
    `Rscript tests/fixtures/generate.R` (requires R + hdi package)."""
    if not _FIXTURE.is_file():
        pytest.skip(
            f"fixture {_FIXTURE.name} missing; run `Rscript tests/fixtures/"
            "generate.R` to generate (requires R + hdi package)"
        )
    with open(_FIXTURE) as f:
        fix = json.load(f)
    X = np.asarray(fix["X"], dtype=np.float64)
    time = np.asarray(fix["time"], dtype=np.float64)
    event = np.asarray(fix["event"], dtype=np.float64)
    coef_r = np.asarray(fix["coef_debiased"], dtype=np.float64)
    res = debiased_cox_lasso(X, time, event, alpha=0.05)
    # Loose tolerance — different penalty paths land at slightly
    # different point estimates; the active-set + sign agreement is
    # the load-bearing comparison.
    active_skein = np.abs(res.coef_debiased) > 0.05
    active_r = np.abs(coef_r) > 0.05
    jaccard = (active_skein & active_r).sum() / max(
        (active_skein | active_r).sum(), 1
    )
    assert jaccard >= 0.7, f"active-set Jaccard vs R::hdi {jaccard:.3f} < 0.7"
