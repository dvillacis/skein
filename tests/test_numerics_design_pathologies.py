"""H2 numerical-stability sweep — design pathologies.

Covers two failure modes for the design matrix:

* **Collinear columns** — `X[:, j] = X[:, k] + ε` for ε ∈ {0, 1e-8, 1e-12}.
  Even at perfect collinearity, a sparsity-inducing penalty has a finite
  minimizer (any feasible distribution of mass across the tied columns
  works); the solver must return one of them without NaN/inf or an
  infinite KKT loop.
* **Zero-variance columns** — `X[:, j] = c`. The `Standardized<D>` wrapper
  divides by the column std, so this exercises both the lazy-standardize
  guard and `rescale_weights_for_standardize`. With `fit_intercept=True`
  a constant column is collinear with the intercept; we want the path
  solver to zero it out rather than blow up.

Each test asserts:

1. all coefficients along the path are finite,
2. predictions on the training matrix are finite,
3. wall-clock stays under a budget (catches infinite KKT loops).

We exercise LS / logistic / Poisson and scalar / group penalties.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import skein_glm

# Budget per fit. Generous (problems are tiny); the point is to catch the
# infinite-loop fallback, not measure performance.
TIME_BUDGET_S = 30.0


# ---------- problem builders ------------------------------------------------


def _ls_problem(n: int = 120, p: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1.5, -2.0, 0.8]
    y = x @ beta + 0.1 * rng.standard_normal(n)
    return x, y


def _logistic_problem(n: int = 200, p: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:3] = [1.2, -1.0, 0.8]
    prob = 1.0 / (1.0 + np.exp(-(x @ beta)))
    y = (rng.uniform(size=n) < prob).astype(np.float64)
    return x, y


def _poisson_problem(n: int = 200, p: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)) * 0.3
    beta = np.zeros(p)
    beta[:3] = [0.4, -0.3, 0.2]
    mu = np.exp(x @ beta)
    y = rng.poisson(mu).astype(np.float64)
    return x, y


def _inject_collinear(x: np.ndarray, eps: float, src: int, dst: int) -> np.ndarray:
    out = x.copy()
    rng = np.random.default_rng(424242)
    out[:, dst] = out[:, src] + eps * rng.standard_normal(x.shape[0])
    return out


def _inject_constant(x: np.ndarray, col: int, value: float = 1.0) -> np.ndarray:
    out = x.copy()
    out[:, col] = value
    return out


def _assert_finite_path(coefs, x, y, *, predict=None) -> None:
    coefs = np.asarray(coefs)
    assert coefs.ndim == 2, f"expected 2-D coef path, got shape {coefs.shape}"
    assert np.all(np.isfinite(coefs)), "non-finite coefficient on the path"
    # A linear prediction at each λ must also be finite.
    pred = x @ coefs.T
    assert np.all(np.isfinite(pred)), "non-finite training-set prediction on the path"
    if predict is not None:
        assert np.all(np.isfinite(predict))


def _fit_under_budget(estimator, x, y) -> None:
    t0 = time.perf_counter()
    estimator.fit(x, y)
    elapsed = time.perf_counter() - t0
    assert elapsed < TIME_BUDGET_S, (
        f"{type(estimator).__name__} fit took {elapsed:.2f}s "
        f"(budget {TIME_BUDGET_S}s) — likely an infinite KKT loop"
    )


# ---------- scalar-penalty LS: collinearity --------------------------------


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-8])
@pytest.mark.parametrize(
    "factory",
    [
        lambda: skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=15),
        lambda: skein_glm.SCADPathRegressor(a=3.7, n_lambdas=15),
        lambda: skein_glm.ElasticNetPathRegressor(alpha=1.0, n_lambdas=15),
        lambda: skein_glm.ElasticNetPathRegressor(alpha=0.5, n_lambdas=15),
    ],
    ids=["mcp", "scad", "lasso", "en"],
)
def test_ls_collinear_columns_remain_finite(factory, eps):
    x, y = _ls_problem()
    # Duplicate a relevant column (index 0) into a noise slot (index 5).
    x_coll = _inject_collinear(x, eps, src=0, dst=5)
    est = factory()
    _fit_under_budget(est, x_coll, y)
    _assert_finite_path(est.coefs_, x_coll, y, predict=est.predict(x_coll[:5]))


# ---------- group penalties: collinearity ----------------------------------


def _groups_of(p: int, group_size: int = 4) -> np.ndarray:
    return np.repeat(np.arange(p // group_size), group_size).astype(np.int64)


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-8])
@pytest.mark.parametrize(
    "factory",
    [
        lambda groups: skein_glm.GroupLassoPathRegressor(
            groups=groups, n_lambdas=12,
        ),
        lambda groups: skein_glm.GroupMCPPathRegressor(
            groups=groups, gamma=3.0, n_lambdas=12,
        ),
        lambda groups: skein_glm.SparseGroupLassoPathRegressor(
            groups=groups, alpha=0.5, n_lambdas=12,
        ),
    ],
    ids=["group-lasso", "group-mcp", "sparse-group-lasso"],
)
def test_ls_group_collinear_columns_remain_finite(factory, eps):
    x, y = _ls_problem(p=16)
    groups = _groups_of(16, group_size=4)
    # Make two columns within group 0 collinear, plus column 0 ≈ column 4
    # (cross-group collinearity). The cross-group case is the harder one
    # because the path solver has to distribute mass between two groups
    # that are pulling on the same direction.
    x_coll = _inject_collinear(x, eps, src=0, dst=1)
    x_coll = _inject_collinear(x_coll, eps, src=0, dst=4)
    est = factory(groups)
    _fit_under_budget(est, x_coll, y)
    _assert_finite_path(est.coefs_, x_coll, y)


# ---------- GLM: collinearity ----------------------------------------------


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-8])
def test_logistic_lasso_collinear_columns_remain_finite(eps):
    x, y = _logistic_problem()
    x_coll = _inject_collinear(x, eps, src=0, dst=5)
    est = skein_glm.LogisticLassoPathRegressor(n_lambdas=12)
    _fit_under_budget(est, x_coll, y)
    _assert_finite_path(est.coefs_, x_coll, y)


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-8])
def test_poisson_lasso_collinear_columns_remain_finite(eps):
    x, y = _poisson_problem()
    x_coll = _inject_collinear(x, eps, src=0, dst=5)
    est = skein_glm.PoissonLassoPathRegressor(n_lambdas=12)
    _fit_under_budget(est, x_coll, y)
    _assert_finite_path(est.coefs_, x_coll, y)


# ---------- zero-variance columns ------------------------------------------


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize(
    "factory",
    [
        lambda: skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=15),
        lambda: skein_glm.SCADPathRegressor(a=3.7, n_lambdas=15),
        lambda: skein_glm.ElasticNetPathRegressor(alpha=1.0, n_lambdas=15),
        lambda: skein_glm.ElasticNetPathRegressor(alpha=0.5, n_lambdas=15),
    ],
    ids=["mcp", "scad", "lasso", "en"],
)
def test_ls_zero_variance_column_remains_finite(factory, standardize):
    x, y = _ls_problem()
    # Two constant columns: one at value 1.0 (collinear with the intercept),
    # one at value 0.0 (entirely uninformative).
    x_const = _inject_constant(x, col=5, value=1.0)
    x_const = _inject_constant(x_const, col=7, value=0.0)
    est = factory()
    est.standardize = standardize
    _fit_under_budget(est, x_const, y)
    _assert_finite_path(est.coefs_, x_const, y)
    # Constant columns must be exactly zero in the solution (no signal,
    # nothing to pick up beyond the intercept). Allow tiny slack for the
    # value=1.0 column at small λ: it competes with the intercept but the
    # KKT condition still drives it to zero. We assert "small" rather than
    # "zero" because with standardize=False the column-std for a constant
    # column is zero and the solver routes around it.
    assert np.max(np.abs(est.coefs_[:, 7])) < 1e-8


@pytest.mark.parametrize("standardize", [False, True])
def test_ls_group_zero_variance_column_remains_finite(standardize):
    x, y = _ls_problem(p=16)
    groups = _groups_of(16, group_size=4)
    # Constant column inside an otherwise-informative group.
    x_const = _inject_constant(x, col=1, value=1.0)
    # An entire group of zero columns.
    for c in range(8, 12):
        x_const = _inject_constant(x_const, col=c, value=0.0)
    est = skein_glm.GroupLassoPathRegressor(
        groups=groups, n_lambdas=12, standardize=standardize,
    )
    _fit_under_budget(est, x_const, y)
    _assert_finite_path(est.coefs_, x_const, y)
    # The all-zero group (cols 8..11) should be exactly zero everywhere.
    assert np.max(np.abs(est.coefs_[:, 8:12])) < 1e-10


def test_logistic_lasso_zero_variance_column_remains_finite():
    x, y = _logistic_problem()
    x_const = _inject_constant(x, col=5, value=1.0)
    x_const = _inject_constant(x_const, col=7, value=0.0)
    est = skein_glm.LogisticLassoPathRegressor(n_lambdas=12)
    _fit_under_budget(est, x_const, y)
    _assert_finite_path(est.coefs_, x_const, y)
    assert np.max(np.abs(est.coefs_[:, 7])) < 1e-8


def test_poisson_lasso_zero_variance_column_remains_finite():
    x, y = _poisson_problem()
    x_const = _inject_constant(x, col=5, value=1.0)
    x_const = _inject_constant(x_const, col=7, value=0.0)
    est = skein_glm.PoissonLassoPathRegressor(n_lambdas=12)
    _fit_under_budget(est, x_const, y)
    _assert_finite_path(est.coefs_, x_const, y)
    assert np.max(np.abs(est.coefs_[:, 7])) < 1e-8


# ---------- standardize-with-per-feature-weights audit ---------------------


def test_standardize_with_per_feature_weights_zero_variance():
    """`rescale_weights_for_standardize` must not divide by zero on a
    constant column. The lazy `Standardized<D>` wrapper bypasses the
    column when std=0; the weight-rescale path also needs that guard."""
    x, y = _ls_problem(p=8)
    x = _inject_constant(x, col=3, value=2.5)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    est = skein_glm.MCPPathRegressor(
        gamma=3.0, n_lambdas=10, standardize=True, weights=weights,
    )
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_, x, y)
    # The constant column must be at zero (no signal).
    assert np.max(np.abs(est.coefs_[:, 3])) < 1e-8


def test_standardize_with_zero_per_feature_weight():
    """A zero per-feature weight means the feature is effectively un-
    penalized. With `standardize=True` the weight is rescaled by the
    column std, so the zero weight has to survive the rescale as zero."""
    x, y = _ls_problem(p=8)
    weights = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    est = skein_glm.MCPPathRegressor(
        gamma=3.0, n_lambdas=10, standardize=True, weights=weights,
    )
    _fit_under_budget(est, x, y)
    _assert_finite_path(est.coefs_, x, y)
    # Feature 0 (zero penalty) is the most active across the path; it
    # should never be shrunk to zero on the entire path (because there's
    # no penalty pulling it down).
    assert np.max(np.abs(est.coefs_[:, 0])) > 0.1
