"""Tests for threaded CV fold parallelism.

The fold loop in `_PathCVMixin` and `_CoxPathCVMixin` dispatches K folds
across threads via joblib's "threads" backend. The Rust path solvers
release the GIL (via `py.allow_threads` in `crates/skein-py/src/lib.rs`),
so the threads run concurrently rather than serializing on the GIL.

These tests verify:
1. n_jobs=1 (serial) and n_jobs=-1 (all cores) produce **bitwise**
   identical results — the fold loop is deterministic regardless of
   thread interleaving.
2. The n_jobs constructor parameter is wired through every user-facing
   CV class.
"""
from __future__ import annotations

import numpy as np
import pytest

from skein_glm import (
    CoxGroupLassoPathCV,
    CoxMCPPathCV,
    CoxSCADPathCV,
    ElasticNetPathCV,
    GroupElasticNetPathCV,
    GroupLassoPathCV,
    GroupMCPPathCV,
    LogisticElasticNetPathCV,
    LogisticGroupLassoPathCV,
    LogisticGroupMCPPathCV,
    LogisticLassoPathCV,
    LogisticMCPPathCV,
    LogisticSCADPathCV,
    MCPPathCV,
    PoissonElasticNetPathCV,
    PoissonGroupLassoPathCV,
    PoissonGroupMCPPathCV,
    PoissonLassoPathCV,
    PoissonMCPPathCV,
    PoissonSCADPathCV,
    SCADPathCV,
    SparseGroupLassoPathCV,
)


def _make_ls(n=300, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:5] = [1.0, -0.8, 0.6, -0.4, 0.2]
    y = X @ beta + 0.3 * rng.standard_normal(n)
    return X, y


def _make_logistic(n=300, p=40, seed=0):
    X, _ = _make_ls(n, p, seed)
    rng = np.random.default_rng(seed + 100)
    beta = np.zeros(p); beta[:5] = [1.0, -0.8, 0.6, -0.4, 0.2]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-X @ beta))).astype(float)
    return X, y


def _make_poisson(n=300, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:5] = [0.4, -0.3, 0.25, -0.2, 0.1]
    y = rng.poisson(np.exp(X @ beta)).astype(np.float64)
    return X, y


def _make_cox(n=300, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:5] = [0.5, -0.5, 0.3, -0.3, 0.2]
    time_obs = rng.exponential(1.0 / np.exp(X @ beta))
    event = (rng.uniform(size=n) < 0.7).astype(float)
    return X, time_obs, event


# --- LS family parity -----------------------------------------------


@pytest.mark.parametrize("cls", [MCPPathCV, SCADPathCV, ElasticNetPathCV])
def test_ls_serial_parallel_parity(cls) -> None:
    X, y = _make_ls(seed=0)
    a = cls(n_lambdas=15, cv=5, n_jobs=1, random_state=0).fit(X, y)
    b = cls(n_lambdas=15, cv=5, n_jobs=-1, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.cv_scores_, b.cv_scores_)
    np.testing.assert_array_equal(a.lambdas_, b.lambdas_)
    assert a.lambda_best_ == b.lambda_best_


# --- Logistic family parity -----------------------------------------


@pytest.mark.parametrize(
    "cls",
    [LogisticLassoPathCV, LogisticElasticNetPathCV, LogisticMCPPathCV, LogisticSCADPathCV],
)
def test_logistic_serial_parallel_parity(cls) -> None:
    X, y = _make_logistic(seed=1)
    a = cls(n_lambdas=15, cv=5, n_jobs=1, random_state=0).fit(X, y)
    b = cls(n_lambdas=15, cv=5, n_jobs=-1, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.cv_scores_, b.cv_scores_)


# --- Poisson family parity ------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [PoissonLassoPathCV, PoissonElasticNetPathCV, PoissonMCPPathCV, PoissonSCADPathCV],
)
def test_poisson_serial_parallel_parity(cls) -> None:
    X, y = _make_poisson(seed=2)
    a = cls(n_lambdas=15, cv=5, n_jobs=1, random_state=0).fit(X, y)
    b = cls(n_lambdas=15, cv=5, n_jobs=-1, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.cv_scores_, b.cv_scores_)


# --- Cox family parity ----------------------------------------------


@pytest.mark.parametrize("cls", [CoxMCPPathCV, CoxSCADPathCV])
def test_cox_serial_parallel_parity(cls) -> None:
    X, time_obs, event = _make_cox(seed=3)
    a = cls(n_lambdas=10, cv=5, n_jobs=1, random_state=0).fit(X, time_obs, event)
    b = cls(n_lambdas=10, cv=5, n_jobs=-1, random_state=0).fit(X, time_obs, event)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    # Cox cv_scores_ uses nanmean/nanstd; compare with NaN-aware
    # equality. Bitwise equality should hold here since the same seed
    # gives the same folds.
    np.testing.assert_array_equal(a.cv_scores_, b.cv_scores_)


# --- group / sparse-group CV parity ---------------------------------


def _make_grouped_ls(n=300, p=40, group_size=4, seed=0):
    """Synthetic with `p / group_size` groups of `group_size` consecutive
    features. First two groups are active."""
    X, _ = _make_ls(n=n, p=p, seed=seed)
    rng = np.random.default_rng(seed + 100)
    n_groups = p // group_size
    groups = np.repeat(np.arange(n_groups), group_size)
    beta = np.zeros(p)
    beta[: 2 * group_size] = rng.uniform(0.4, 0.8, size=2 * group_size) * rng.choice([-1, 1], size=2 * group_size)
    y = X @ beta + 0.3 * rng.standard_normal(n)
    return X, y, groups


@pytest.mark.parametrize(
    "cls",
    [GroupLassoPathCV, GroupMCPPathCV, GroupElasticNetPathCV, SparseGroupLassoPathCV],
)
def test_grouped_ls_serial_parallel_parity(cls) -> None:
    X, y, groups = _make_grouped_ls(seed=4)
    a = cls(groups=groups, n_lambdas=10, cv=5, n_jobs=1, random_state=0).fit(X, y)
    b = cls(groups=groups, n_lambdas=10, cv=5, n_jobs=-1, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(a.cv_scores_, b.cv_scores_)


@pytest.mark.parametrize("cls", [LogisticGroupLassoPathCV, LogisticGroupMCPPathCV])
def test_grouped_logistic_serial_parallel_parity(cls) -> None:
    X, y_ls, groups = _make_grouped_ls(seed=5)
    rng = np.random.default_rng(5)
    beta = np.zeros(X.shape[1])
    beta[:8] = rng.uniform(0.4, 0.8, size=8) * rng.choice([-1, 1], size=8)
    y = (rng.uniform(size=X.shape[0]) < 1.0 / (1.0 + np.exp(-X @ beta))).astype(float)
    a = cls(groups=groups, n_lambdas=8, cv=5, n_jobs=1, random_state=0).fit(X, y)
    b = cls(groups=groups, n_lambdas=8, cv=5, n_jobs=-1, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)


@pytest.mark.parametrize("cls", [PoissonGroupLassoPathCV, PoissonGroupMCPPathCV])
def test_grouped_poisson_serial_parallel_parity(cls) -> None:
    rng = np.random.default_rng(6)
    n, p = 300, 40
    X = rng.standard_normal((n, p))
    groups = np.repeat(np.arange(10), 4)
    beta = np.zeros(p); beta[:8] = rng.uniform(0.2, 0.4, size=8) * rng.choice([-1, 1], size=8)
    y = rng.poisson(np.exp(X @ beta)).astype(np.float64)
    a = cls(groups=groups, n_lambdas=8, cv=5, n_jobs=1, random_state=0).fit(X, y)
    b = cls(groups=groups, n_lambdas=8, cv=5, n_jobs=-1, random_state=0).fit(X, y)
    np.testing.assert_array_equal(a.coef_, b.coef_)


def test_grouped_cox_serial_parallel_parity() -> None:
    rng = np.random.default_rng(7)
    n, p = 300, 40
    X = rng.standard_normal((n, p))
    groups = np.repeat(np.arange(10), 4)
    beta = np.zeros(p); beta[:8] = rng.uniform(0.3, 0.5, size=8) * rng.choice([-1, 1], size=8)
    time_obs = rng.exponential(1.0 / np.exp(X @ beta))
    event = (rng.uniform(size=n) < 0.7).astype(float)
    a = CoxGroupLassoPathCV(
        groups=groups, n_lambdas=6, cv=5, n_jobs=1, random_state=0,
    ).fit(X, time_obs, event)
    b = CoxGroupLassoPathCV(
        groups=groups, n_lambdas=6, cv=5, n_jobs=-1, random_state=0,
    ).fit(X, time_obs, event)
    np.testing.assert_array_equal(a.coef_, b.coef_)


# --- n_jobs sanity --------------------------------------------------


def test_n_jobs_present_in_init_signatures() -> None:
    """Every user-facing CV class accepts n_jobs."""
    import inspect
    classes = [
        MCPPathCV, SCADPathCV, ElasticNetPathCV,
        LogisticLassoPathCV, LogisticElasticNetPathCV,
        LogisticMCPPathCV, LogisticSCADPathCV,
        PoissonLassoPathCV, PoissonElasticNetPathCV,
        PoissonMCPPathCV, PoissonSCADPathCV,
        CoxMCPPathCV, CoxSCADPathCV,
        GroupLassoPathCV, GroupMCPPathCV, GroupElasticNetPathCV,
        SparseGroupLassoPathCV,
        LogisticGroupLassoPathCV, LogisticGroupMCPPathCV,
        PoissonGroupLassoPathCV, PoissonGroupMCPPathCV,
        CoxGroupLassoPathCV,
    ]
    for cls in classes:
        sig = inspect.signature(cls)
        assert "n_jobs" in sig.parameters, f"{cls.__name__} missing n_jobs"
        assert sig.parameters["n_jobs"].default is None
