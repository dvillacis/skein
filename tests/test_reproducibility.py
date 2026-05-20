"""Reproducibility audit for every randomized estimator (H4).

Every public estimator that consumes an RNG (CV fold construction via
``KFold(shuffle=True, random_state=...)``, stability-selection bootstrap
subsampling, graphical bootstrap, multinomial KFold, adaptive-CV
pilot/refit folds) is pinned here. Each test asserts two things:

1. **Same ``random_state`` → bit-identical fit.** Two independent fits
   with the same seed produce ``np.array_equal``-identical coefficient
   tensors, CV score grids, stability selection probabilities, etc.
2. **Different ``random_state`` → measurably different fit.** Catches
   a silent-no-op regression where the ``random_state`` constructor
   kwarg gets parsed but never reaches the RNG consumer (e.g., a
   refactor that drops ``random_state`` from the ``KFold`` call site).

The bit-equality assertion targets the natural state-vector of each
estimator family: CV exposes ``coef_`` + ``cv_scores_`` + ``lambda_best_``;
stability selection exposes ``selection_probabilities_`` +
``stable_features_``; graphical bootstrap exposes
``edge_selection_probabilities_`` + CI bounds; etc.

BLAS-thread caveat: the Rust core's path solver is deterministic
(coordinate descent has no RNG), but hardware BLAS kernels (Accelerate
on macOS, OpenBLAS on Linux) can produce off-by-ulp results when work
is split across threads — dot-product summation order changes with
thread scheduling. The reproducibility we're asserting is *for the
fold construction and bootstrap resampling*, which is pure Python
RNG and unaffected by BLAS threading; ``np.array_equal`` does what
we want at the small problem sizes used here (BLAS stays single-
threaded). If a future test grows beyond the single-thread regime,
gate it with ``OMP_NUM_THREADS=1`` / ``OPENBLAS_NUM_THREADS=1``.
"""
from __future__ import annotations

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")


# --------------------------------------------------------------------------
# Shared synthetic. Small (n=40, p=8) so the whole file completes in a
# few seconds; well-conditioned Gaussian design with a sparse two-tap
# signal — same shape as the v1.0 smoke tests.
# --------------------------------------------------------------------------


def _xy(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 40, 8
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[1] = -1.0
    y = x @ true_beta + 0.2 * rng.standard_normal(n)
    return x, y


def _xy_classification(seed: int = 0):
    """Binary labels for logistic CV."""
    x, y_cont = _xy(seed)
    y = (y_cont > 0.0).astype(np.float64)
    # Ensure both classes are present even on adverse seeds.
    if y.sum() in (0, len(y)):
        y[0] = 1.0 - y[0]
    return x, y


def _xy_multiclass(seed: int = 0):
    """Three-class labels by bucketing a continuous target."""
    x, y_cont = _xy(seed)
    qs = np.quantile(y_cont, [1.0 / 3, 2.0 / 3])
    y = np.digitize(y_cont, qs).astype(np.int64)
    return x, y


def _xy_glasso(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 60, 5
    return rng.standard_normal((n, p))


# --------------------------------------------------------------------------
# Helper: assert two estimators' state-vectors agree exactly (same seed)
# / disagree measurably (different seed).
# --------------------------------------------------------------------------


def _assert_arrays_equal(a, b, label: str) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    # Bit-equality on the random-fold state vector. The Rust solver is
    # deterministic, so same folds + same hyperparameters = same β.
    assert np.array_equal(a, b), (
        f"{label}: same random_state should give bit-identical output; "
        f"max abs diff = {np.abs(a - b).max():.3e}"
    )


def _assert_arrays_differ(a, b, label: str, atol: float = 1e-6) -> None:
    a, b = np.asarray(a), np.asarray(b)
    diff = np.abs(a - b).max()
    assert diff > atol, (
        f"{label}: different random_state appears to be a no-op "
        f"(max abs diff = {diff:.3e}, threshold = {atol:.3e})"
    )


# --------------------------------------------------------------------------
# CV — LS / GLM / group representatives. All CV estimators go through
# the same ``KFold(shuffle=True, random_state=...)`` path, so testing a
# handful covers the family.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator_factory",
    [
        lambda rs: skein.MCPPathCV(
            gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2, cv=4,
            max_iter=200, tol=1e-8, random_state=rs,
        ),
        lambda rs: skein.GroupLassoPathCV(
            groups=np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
            n_lambdas=8, lambda_min_ratio=1e-2, cv=4,
            max_iter=200, tol=1e-8, random_state=rs,
        ),
    ],
    ids=["MCPPathCV", "GroupLassoPathCV"],
)
def test_ls_cv_reproducible(estimator_factory) -> None:
    x, y = _xy()
    a = estimator_factory(0).fit(x, y)
    b = estimator_factory(0).fit(x, y)
    c = estimator_factory(1).fit(x, y)

    # Same seed → bit-identical CV grid + winning β.
    _assert_arrays_equal(a.coef_, b.coef_, "coef_")
    _assert_arrays_equal(a.cv_scores_, b.cv_scores_, "cv_scores_")
    assert a.lambda_best_ == b.lambda_best_

    # Different seed → different folds → different CV scores (and
    # typically different best λ). Bound the assertion to cv_scores_
    # since coef_ would also catch this but cv_scores_ is the direct
    # signal that folds differ.
    _assert_arrays_differ(a.cv_scores_, c.cv_scores_, "cv_scores_")


def test_logistic_cv_reproducible() -> None:
    x, y = _xy_classification()
    factory = lambda rs: skein.LogisticLassoPathCV(  # noqa: E731
        n_lambdas=8, lambda_min_ratio=1e-2, cv=4,
        max_iter=200, tol=1e-7, max_outer=15, outer_tol=1e-6,
        random_state=rs,
    )
    a = factory(0).fit(x, y)
    b = factory(0).fit(x, y)
    c = factory(1).fit(x, y)

    _assert_arrays_equal(a.coef_, b.coef_, "coef_")
    _assert_arrays_equal(a.cv_scores_, b.cv_scores_, "cv_scores_")
    _assert_arrays_differ(a.cv_scores_, c.cv_scores_, "cv_scores_")


# --------------------------------------------------------------------------
# Stability selection — bootstrap subsampling driven by ``random_state``.
# --------------------------------------------------------------------------


def test_stability_selection_reproducible() -> None:
    x, y = _xy()
    base = skein.MCPPathRegressor(
        gamma=3.0, n_lambdas=8, lambda_min_ratio=1e-2,
        max_iter=200, tol=1e-7,
    )
    factory = lambda rs: skein.StabilitySelection(  # noqa: E731
        base_estimator=base, n_bootstraps=10, sample_fraction=0.5,
        threshold=0.6, random_state=rs,
    )

    a = factory(0).fit(x, y)
    b = factory(0).fit(x, y)
    c = factory(1).fit(x, y)

    _assert_arrays_equal(
        a.selection_probabilities_, b.selection_probabilities_,
        "selection_probabilities_",
    )
    _assert_arrays_equal(a.stable_features_, b.stable_features_, "stable_features_")

    # Different seeds → different bootstrap samples → different per-λ
    # selection frequencies. The aggregate ``max_probabilities_`` can
    # coincide by chance at the 0.5 / 0.6 thresholds on this small
    # problem, so we check the full grid.
    _assert_arrays_differ(
        a.selection_probabilities_, c.selection_probabilities_,
        "selection_probabilities_", atol=1e-3,
    )


# --------------------------------------------------------------------------
# Graphical stability + bootstrap — graph-side analogues of the above.
# --------------------------------------------------------------------------


def test_graphical_stability_reproducible() -> None:
    x = _xy_glasso()
    base = skein.GraphicalLasso(alpha=0.1, max_iter=200, tol=1e-6)
    lambdas = np.geomspace(0.5, 0.05, 5)

    factory = lambda rs: skein.GraphicalStabilitySelection(  # noqa: E731
        base_estimator=base, lambdas=lambdas, n_bootstraps=10,
        sample_fraction=0.5, threshold=0.6, random_state=rs,
    )

    a = factory(0).fit(x)
    b = factory(0).fit(x)
    c = factory(1).fit(x)

    _assert_arrays_equal(
        a.selection_probabilities_, b.selection_probabilities_,
        "selection_probabilities_",
    )
    _assert_arrays_differ(
        a.selection_probabilities_, c.selection_probabilities_,
        "selection_probabilities_", atol=1e-3,
    )


def test_graphical_bootstrap_reproducible() -> None:
    x = _xy_glasso()
    base = skein.GraphicalLasso(alpha=0.1, max_iter=200, tol=1e-6)

    factory = lambda rs: skein.GraphicalBootstrap(  # noqa: E731
        base_estimator=base, n_bootstraps=10, alpha=0.05, random_state=rs,
    )

    a = factory(0).fit(x)
    b = factory(0).fit(x)
    c = factory(1).fit(x)

    _assert_arrays_equal(
        a.edge_selection_probabilities_, b.edge_selection_probabilities_,
        "edge_selection_probabilities_",
    )
    _assert_arrays_equal(a.ci_lower_, b.ci_lower_, "ci_lower_")
    _assert_arrays_equal(a.ci_upper_, b.ci_upper_, "ci_upper_")
    _assert_arrays_differ(
        a.edge_selection_probabilities_, c.edge_selection_probabilities_,
        "edge_selection_probabilities_", atol=1e-3,
    )


# --------------------------------------------------------------------------
# Adaptive CV — nests CV inside an adaptive-weights pilot+refit. The
# random_state seeds the KFold shuffle in the *final* CV pass.
# --------------------------------------------------------------------------


def test_adaptive_lasso_cv_reproducible() -> None:
    x, y = _xy()
    factory = lambda rs: skein.AdaptiveLassoPathCV(  # noqa: E731
        eta=1.0, n_pilot_lambdas=6, cv=4, n_lambdas=8,
        lambda_min_ratio=1e-2, max_iter=200, tol=1e-7,
        random_state=rs,
    )
    a = factory(0).fit(x, y)
    b = factory(0).fit(x, y)
    c = factory(1).fit(x, y)

    _assert_arrays_equal(a.coef_, b.coef_, "coef_")
    _assert_arrays_equal(a.cv_scores_, b.cv_scores_, "cv_scores_")
    _assert_arrays_differ(a.cv_scores_, c.cv_scores_, "cv_scores_")


# --------------------------------------------------------------------------
# Multinomial CV — separate code path; KFold shuffle is driven by
# ``random_state`` in ``_MultinomialPathCVBase``.
# --------------------------------------------------------------------------


def test_multinomial_lasso_cv_reproducible() -> None:
    x, y = _xy_multiclass()
    factory = lambda rs: skein.MultinomialLassoPathCV(  # noqa: E731
        n_lambdas=6, lambda_min_ratio=1e-2, cv=3,
        max_iter=200, tol=1e-7, max_outer=15, outer_tol=1e-6,
        random_state=rs,
    )
    a = factory(0).fit(x, y)
    b = factory(0).fit(x, y)
    c = factory(1).fit(x, y)

    _assert_arrays_equal(a.coef_, b.coef_, "coef_")
    _assert_arrays_equal(a.cv_scores_, b.cv_scores_, "cv_scores_")
    _assert_arrays_differ(a.cv_scores_, c.cv_scores_, "cv_scores_")
