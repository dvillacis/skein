"""O6: per-λ timing surface on ``info_``.

The Rust path solvers track per-λ wall-clock time and surface it through
the ``info_`` dict so users can profile a fit without rebuilding skein with
``SKEIN_PROFILE_PATH=1``. This test pins the schema.
"""
from __future__ import annotations

import numpy as np
import pytest

skein = pytest.importorskip("skein_glm")


def _toy_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 100, 20
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    y = x @ true_beta + 0.1 * rng.standard_normal(n)
    return x, y


def _toy_logistic(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 120, 15
    x = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = [1.5, -2.0, 0.8]
    logits = x @ true_beta
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    return x, y


def _assert_per_lambda_keys(info, n_lambdas, *keys):
    for k in keys:
        assert k in info, f"missing key {k!r} in info_"
        v = info[k]
        assert len(v) == n_lambdas, (
            f"{k!r}: expected len {n_lambdas}, got {len(v)}"
        )


def test_cd_path_info_carries_times_ns():
    x, y = _toy_problem()
    model = skein.MCPPathRegressor(n_lambdas=12, gamma=3.0).fit(x, y)

    info = model.info_
    n_lams = len(model.lambdas_)
    _assert_per_lambda_keys(
        info, n_lams,
        "iters", "converged", "final_objs", "working_set_sizes",
        "kkt_passes", "times_ns",
    )

    times = info["times_ns"]
    # Timings are non-negative integers; a 100x20 fit should be well under
    # a second per λ on any sane host.
    assert all(isinstance(t, int) for t in times)
    assert all(t >= 0 for t in times)
    assert sum(times) > 0, "expected at least one λ to take > 0 ns"


def test_prox_newton_path_info_carries_times_ns():
    x, y = _toy_logistic()
    model = skein.LogisticMCPPathRegressor(n_lambdas=10, gamma=3.0).fit(x, y)

    info = model.info_
    n_lams = len(model.lambdas_)
    _assert_per_lambda_keys(
        info, n_lams,
        "outer_iters", "outer_converged", "inner_iters", "final_losses",
        "times_ns",
    )

    times = info["times_ns"]
    assert all(isinstance(t, int) and t >= 0 for t in times)
    assert sum(times) > 0


def test_block_path_info_carries_times_ns():
    x, y = _toy_problem()
    groups = np.repeat(np.arange(5), 4)  # 5 groups of 4 features
    model = skein.GroupLassoPathRegressor(
        groups=groups, n_lambdas=10
    ).fit(x, y)

    info = model.info_
    n_lams = len(model.lambdas_)
    _assert_per_lambda_keys(
        info, n_lams,
        "iters", "converged", "final_objs", "working_set_sizes",
        "kkt_passes", "times_ns",
    )
    assert sum(info["times_ns"]) > 0
