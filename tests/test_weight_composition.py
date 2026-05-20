"""Randomized weight-composition properties through the public Python API (H3).

The Rust crate has property coverage on the prox / GLM surrogate / standardize
identities (see ``crates/skein-core/src/{prox,datafit,standardize}``). This
file complements that by checking invariances that must hold across the
PyO3 boundary — per-feature ``weights``, per-sample ``sample_weights``, and
per-group ``weights`` arguments composing into the same fit when set to
their identity values (``None`` ↔ ``ones(…)``), and when permutations are
applied in lockstep.

The properties protect against:

- The no-weights fast path drifting from the explicit-weights path in
  the dispatch helpers (``_glm_dispatch_inputs``, ``_ls_group_dispatch_inputs``);
- An indexing bug between per-feature / per-group weights and the
  column / group they multiply (caught by the permutation-equivariance
  test);
- ``sample_weights`` becoming a silent no-op (caught by the explicit
  positive test asserting non-uniform weights *do* change the fit).

Hypothesis drives the RNG seed only; ``X``, ``y``, and the regularization
path are then generated deterministically from that seed into a Gaussian-
style synthetic that's well-conditioned by construction. The bit-equality
invariances we are testing are not strengthened by element-wise X / y
fuzzing — they are strengthened by exercising the dispatch surface across
many *runs* — and tightly-bounded synthetic inputs sidestep the long tail
of degenerate draws that element-wise random X would generate.

## Why ``sample_weights=None`` is not bit-equal to ``sample_weights=ones(n)``

These two configurations take structurally different code paths in
``crates/skein-py/src/ls.rs``: the no-weights path centers via
``standardize`` / ``destandardize_path`` and recovers the intercept
post-hoc; the explicit-sample-weights path appends a 1s column to ``X``
with weight 0 (unpenalised augmented intercept) and solves directly.
Both formulations target the same penalised LS objective and converge to
the same optimum *in the limit*, but at any finite ``tol`` the iterate
trajectory and the path λ-grid differ — so the two are not a bit-
equality invariance and we don't test them as one. The positive
``sample_weights`` test below verifies the pathway is actually wired
through, which is the regression the H3 tier was meant to catch.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

skein = pytest.importorskip("skein_glm")

N_SAMPLES = 16
N_FEATURES = 6
ATOL = 1e-10
RTOL = 1e-10

_FIT_KWARGS = dict(
    gamma=3.0,
    n_lambdas=8,
    lambda_min_ratio=1e-2,
    max_iter=50,
    tol=1e-6,
    standardize=False,
    fit_intercept=True,
)


def _synth(seed: int):
    """Well-conditioned Gaussian design + sparse signal — the same shape
    used by the v1.0 smoke tests. Two-tap signal so the path's first
    few λ's enter active features (catching any pre-solve weight
    handling) while later λ's exercise the screening tail."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_SAMPLES, N_FEATURES))
    true_beta = np.zeros(N_FEATURES)
    true_beta[0] = 1.5
    true_beta[1] = -0.8
    y = x @ true_beta + 0.1 * rng.standard_normal(N_SAMPLES)
    return x, y


def _fit_mcp_path(x, y, *, weights=None, sample_weights=None):
    model = skein.MCPPathRegressor(
        weights=weights,
        sample_weights=sample_weights,
        **_FIT_KWARGS,
    ).fit(x, y)
    return model.coefs_, model.intercepts_


# Hypothesis envelope — 30 examples is plenty since each test fits two
# 8-step paths on n=16/p=6. `deadline=4000` is generous (each fit takes
# <50 ms warm).
_HYP_SETTINGS = settings(
    max_examples=30,
    deadline=4000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)


# --------------------------------------------------------------------------
# Per-feature / per-group identity: passing ``ones(…)`` must be bit-equal
# to passing ``None``. Both inputs take the same code path internally
# (``None`` is rewritten to ``Array1::ones(p)`` at the PyO3 boundary), so
# any divergence here is a real regression.
# --------------------------------------------------------------------------


@_HYP_SETTINGS
@given(seed=st.integers(0, 2**31 - 1))
def test_ones_feature_weights_match_none(seed: int) -> None:
    x, y = _synth(seed)
    base_coef, base_int = _fit_mcp_path(x, y)
    ones_coef, ones_int = _fit_mcp_path(
        x, y, weights=np.ones(N_FEATURES, dtype=np.float64)
    )
    np.testing.assert_allclose(base_coef, ones_coef, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(base_int, ones_int, atol=ATOL, rtol=RTOL)


@_HYP_SETTINGS
@given(seed=st.integers(0, 2**31 - 1))
def test_ones_group_weights_match_none(seed: int) -> None:
    """Per-group ones ≡ no weights for the group lasso path."""
    x, y = _synth(seed)
    # Three contiguous groups of size 2 (N_FEATURES = 6).
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    def fit(weights):
        return skein.GroupLassoPathRegressor(
            groups=groups,
            n_lambdas=8,
            lambda_min_ratio=1e-2,
            weights=weights,
            max_iter=50,
            tol=1e-6,
            standardize=False,
            fit_intercept=True,
        ).fit(x, y)

    base = fit(None)
    ones = fit(np.ones(3, dtype=np.float64))
    np.testing.assert_allclose(base.coefs_, ones.coefs_, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        base.intercepts_, ones.intercepts_, atol=ATOL, rtol=RTOL
    )


# --------------------------------------------------------------------------
# Permutation equivariance for per-feature ``weights``. A column-
# permutation π applied to X *and* to the weights vector must produce
# the same fit as the original X with the same permutation applied to
# the coefficients. This is the property that catches an indexing bug
# between weights and their target columns.
#
# Run at tight tol so nonconvex-MCP local-minimum drift is below the
# assertion noise floor (at default ``tol=1e-6`` and ``max_iter=50``
# different column orderings can converge to different local minima
# even though they parametrise the same problem).
# --------------------------------------------------------------------------


@_HYP_SETTINGS
@given(seed=st.integers(0, 2**31 - 1))
def test_mcp_path_column_permutation_equivariant(seed: int) -> None:
    x, y = _synth(seed)
    rng = np.random.default_rng(seed ^ 0xABCDEF)
    perm = rng.permutation(N_FEATURES)
    weights = rng.uniform(0.5, 2.0, size=N_FEATURES)

    # Pin λ-grid — auto-derived grid weights ``max |X · y|`` by ``weights``
    # and ``max`` is permutation-invariant, but the path solver may
    # short-circuit on a `lambda_max` rounded differently between fits.
    # Fixing ``lambdas`` makes the comparison bit-clean.
    lams = np.geomspace(1.0, 1e-2, 8)

    def fit(x_in, w_in):
        return skein.MCPPathRegressor(
            gamma=3.0,
            lambdas=lams,
            weights=w_in,
            max_iter=2000,
            tol=1e-12,
            standardize=False,
            fit_intercept=True,
        ).fit(x_in, y)

    base = fit(x, weights)
    permuted = fit(x[:, perm], weights[perm])

    np.testing.assert_allclose(
        permuted.coefs_, base.coefs_[:, perm], atol=1e-9, rtol=1e-9
    )
    np.testing.assert_allclose(
        permuted.intercepts_, base.intercepts_, atol=1e-9, rtol=1e-9
    )


# --------------------------------------------------------------------------
# Positive test: ``sample_weights`` must actually affect the fit. A silent
# no-op (the Rust solver dropping the weights argument) would slip past
# every identity / equivariance test above, so we explicitly assert
# non-uniform sample_weights produce a coef path that differs from None
# by more than mere solver tolerance.
# --------------------------------------------------------------------------


@_HYP_SETTINGS
@given(seed=st.integers(0, 2**31 - 1))
def test_nonuniform_sample_weights_changes_fit(seed: int) -> None:
    x, y = _synth(seed)
    fit_kwargs = {**_FIT_KWARGS, "max_iter": 200, "tol": 1e-8}
    rng = np.random.default_rng(seed ^ 0x123456)
    # Mixed 0.5 / 2.0 weights — 4× contrast across samples, clearly
    # observable in the fitted β if the dispatch is wired.
    sw = rng.choice([0.5, 2.0], size=N_SAMPLES).astype(np.float64)

    base = skein.MCPPathRegressor(**fit_kwargs).fit(x, y)
    weighted = skein.MCPPathRegressor(sample_weights=sw, **fit_kwargs).fit(x, y)

    # The two paths use different intercept formulations (see the
    # module-level docstring), so the auto-derived λ-grids differ.
    # Compare via the max absolute coef discrepancy across the *interior*
    # of the path — both fits should produce nonzero β somewhere, and
    # those β's must measurably differ.
    diff = np.abs(base.coefs_ - weighted.coefs_).max()
    assert diff > 1e-3, (
        f"sample_weights={sw[:4]}... appears to be a no-op (max coef diff "
        f"vs None is {diff:.3e}); this would mean the explicit-sample-weights "
        f"dispatch is dead code."
    )
