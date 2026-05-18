"""Per-block (group) orthonormalization (grpreg's `orthogonalize` trick).

Verifies that:

* The Rust-backed orthonormalization produces blocks with exact
  identity Gram (within float precision).
* The back-transform inverts the orthonormalization: predictions are
  identical whether computed via `X @ coefs_orig` or
  `X_orth @ coefs_orth`.
* The high-level :func:`fit_with_orthonormalization` wrapper composes
  with any *PathRegressor and returns coefficients that match a
  reference fit on the original X up to solver precision (recovery of
  the true active set, prediction agreement).
* Singular / rank-deficient blocks are rejected with a clear error.
"""
from __future__ import annotations

import numpy as np
import pytest

import skein_glm


@pytest.fixture
def grouped_problem():
    rng = np.random.default_rng(20)
    n, p = 120, 12
    x = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[0] = 2.0
    beta_true[1] = -1.5
    beta_true[5] = 1.0
    y = x @ beta_true + 0.1 * rng.standard_normal(n)
    labels = np.repeat(np.arange(4), 3).astype(np.int64)
    return x, y, labels


def test_orthonormalize_produces_identity_gram_per_block(grouped_problem):
    x, _, labels = grouped_problem
    x_orth, bt = skein_glm.orthonormalize_groups(x, labels)
    n = x.shape[0]
    for g in range(bt.n_groups):
        cols = np.where(labels == g)[0]
        gram_norm = (x_orth[:, cols].T @ x_orth[:, cols]) / n
        np.testing.assert_allclose(gram_norm, np.eye(len(cols)), atol=1e-10)


def test_back_transform_recovers_predictions(grouped_problem):
    x, _, labels = grouped_problem
    x_orth, bt = skein_glm.orthonormalize_groups(x, labels)
    rng = np.random.default_rng(0)
    beta_orth = rng.standard_normal(x.shape[1])
    beta_orig = bt.apply_to_coefs(beta_orth)
    np.testing.assert_allclose(x @ beta_orig, x_orth @ beta_orth, atol=1e-10)


def test_back_transform_path_matches_per_row(grouped_problem):
    x, _, labels = grouped_problem
    _, bt = skein_glm.orthonormalize_groups(x, labels)
    rng = np.random.default_rng(1)
    betas = rng.standard_normal((4, x.shape[1]))
    path_back = bt.apply_to_coefs_path(betas)
    for k in range(4):
        single = bt.apply_to_coefs(betas[k])
        np.testing.assert_allclose(path_back[k], single, atol=1e-12)


def test_orthonormalize_rejects_collinear_block():
    # 4x2 matrix where col 1 = col 0 within a single 2-feature group.
    x = np.array([[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [0.5, 0.5]])
    labels = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="rank-deficient"):
        skein_glm.orthonormalize_groups(x, labels)


def test_orthonormalize_groups_length_mismatch_raises():
    x = np.zeros((4, 3))
    with pytest.raises(ValueError, match="groups length"):
        skein_glm.orthonormalize_groups(x, np.array([0, 0], dtype=np.int64))


def test_fit_with_orthonormalization_matches_reference_intercept(grouped_problem):
    """At λ_max, both fits give β=0 and the intercept equals ȳ."""
    x, y, labels = grouped_problem
    ref = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=20,
        fit_intercept=True, standardize=False,
    ).fit(x, y)

    model = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=20,
        fit_intercept=True, standardize=False,
    )
    coefs_orig, intercepts, _, _ = skein_glm.fit_with_orthonormalization(
        model, x, y, labels, fit_intercept=True,
    )
    # At λ_max both β = 0 → intercept = ȳ.
    np.testing.assert_allclose(coefs_orig[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(intercepts[0], ref.intercepts_[0], atol=1e-10)


def test_fit_with_orthonormalization_recovers_same_active_set(grouped_problem):
    x, y, labels = grouped_problem
    ref = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3,
        fit_intercept=True, standardize=False,
    ).fit(x, y)

    model = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3,
        fit_intercept=True, standardize=False,
    )
    coefs_orig, _, _, _ = skein_glm.fit_with_orthonormalization(
        model, x, y, labels, fit_intercept=True,
    )
    ref_top3 = sorted(np.argsort(np.abs(ref.coefs_[-1]))[::-1][:3].tolist())
    orth_top3 = sorted(np.argsort(np.abs(coefs_orig[-1]))[::-1][:3].tolist())
    assert ref_top3 == orth_top3 == [0, 1, 5], (
        f"both fits should pick features 0,1,5 — ref={ref_top3}, orth={orth_top3}"
    )


def test_fit_with_orthonormalization_predictions_close_to_reference(grouped_problem):
    x, y, labels = grouped_problem
    ref = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3,
        fit_intercept=True, standardize=False,
    ).fit(x, y)

    model = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3,
        fit_intercept=True, standardize=False,
    )
    coefs_orig, intercepts, _, _ = skein_glm.fit_with_orthonormalization(
        model, x, y, labels, fit_intercept=True,
    )
    y_ref = x @ ref.coefs_[-1] + ref.intercepts_[-1]
    y_orth = x @ coefs_orig[-1] + intercepts[-1]
    # Solver tolerance is 1e-6 by default; predictions agree to ~1e-5.
    np.testing.assert_allclose(y_ref, y_orth, atol=1e-4)


def test_fit_with_orthonormalization_no_intercept(grouped_problem):
    """fit_intercept=False skips the centering step."""
    x, y, labels = grouped_problem
    # Center y manually so the no-intercept model still has hope.
    y_c = y - y.mean()
    model = skein_glm.GroupMCPPathRegressor(
        groups=labels, gamma=3.0, n_lambdas=15, fit_intercept=False, standardize=False,
    )
    coefs_orig, intercepts, _, _ = skein_glm.fit_with_orthonormalization(
        model, x, y_c, labels, fit_intercept=False,
    )
    np.testing.assert_allclose(intercepts, 0.0)


def test_block_back_transform_n_groups_attribute(grouped_problem):
    x, _, labels = grouped_problem
    _, bt = skein_glm.orthonormalize_groups(x, labels)
    assert bt.n_groups == 4
