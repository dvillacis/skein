"""Per-feature penalty-weights correctness check (Gaussian LS).

Each scalar regressor in skein takes a `weights=` kwarg of shape
`(n_features,)` that multiplies the per-feature penalty contribution:

    Σ_j w_j · ρ(β_j; λ)

This script exercises four invariants that any honest implementation
must satisfy, *without* relying on cross-package comparison — the
ground truth is a mathematical identity, not a reference solver:

  (1) Uniform weights `w = 1` ≡ no weights argument. *All penalties.*

  (2) X-scaling identity: for the L1 penalty specifically, weights
      `w` is equivalent to fitting the unweighted problem on a scaled
      design `X·diag(1/w)`, then rescaling β back by `1/w`. Proof:
      with `γ_j = w_j · β_j`, the penalty becomes `Σ_j |γ_j|`
      (uniform) and `X β = X·diag(1/w) γ`. *Lasso only* — the
      identity requires 1-homogeneity in β, which MCP/SCAD break at
      their `γλ` knee.

  (3) Entry-λ scales as `1/w_j`. The smallest λ at which feature j
      becomes active is `|grad_j(β=0)| / w_j` (KKT at β=0 reads
      `|grad_j| ≤ w_j · λ`). So if we trace where feature j first
      becomes non-zero as λ decreases, that entry-λ should shift by
      a factor of `1/w_j` between the unweighted and the weighted
      fit. *All penalties.*

  (4) `w_j → ∞` pins feature j to zero, *regardless of λ*. *All
      penalties.*

We run the suite across lasso (ElasticNet @ α=1), MCP, and SCAD.

Run as a script:

    python benches/correctness/penalty_weights_ls.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benches.correctness import _common as cc  # noqa: E402,F401 — sys.path tweak above; cc used for write_results

logger = logging.getLogger("benches.correctness.penalty_weights_ls")


def _build_problem(n: int = 1000, p: int = 30, k_active: int = 5, snr: float = 5.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    idx = rng.choice(p, size=k_active, replace=False)
    beta[idx] = rng.choice([-1.0, 1.0], size=k_active) * rng.uniform(0.8, 2.0, size=k_active)
    signal = x @ beta
    noise = (float(np.std(signal)) / snr) * rng.standard_normal(n)
    y = signal + noise
    return x, y, beta, idx


def _fit(estimator_cls, x, y, *, lambdas, weights=None, **kw):
    """Tiny adapter — pass `weights=` only if provided, so we can also
    test the `weights=None` default path."""
    if weights is None:
        est = estimator_cls(lambdas=lambdas, **kw)
    else:
        est = estimator_cls(lambdas=lambdas, weights=weights, **kw)
    est.fit(x, y)
    return np.asarray(est.coefs_)


def _make_lambda_grid(x: np.ndarray, y: np.ndarray, n_lambdas: int = 30) -> np.ndarray:
    lam_max = float(np.max(np.abs(x.T @ y)) / x.shape[0])
    return np.geomspace(lam_max, lam_max * 1e-2, n_lambdas)


# ---------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------

def check_uniform_weights_equivalent(estimator_cls, x, y, lambdas, tol=1e-10, **kw) -> dict:
    """(1) weights=ones(p) must produce bit-identical coefs to no-weights."""
    p = x.shape[1]
    coefs_default = _fit(estimator_cls, x, y, lambdas=lambdas, weights=None, **kw)
    coefs_uniform = _fit(estimator_cls, x, y, lambdas=lambdas, weights=np.ones(p), **kw)
    max_diff = float(np.max(np.abs(coefs_default - coefs_uniform)))
    return {
        "name": "uniform_weights_equivalent",
        "max_diff": max_diff,
        "passed": max_diff < tol,
        "tol": tol,
    }


def check_x_scaling_identity(estimator_cls, x, y, lambdas, tol=1e-6, **kw) -> dict:
    """(2) Fit with `weights=w` vs. fit on `X·diag(1/w)` with weights=1,
    then rescale β back. The two coef paths should match up to inner-
    solver tolerance.

    `tol=1e-6` is conservative — both fits use the package default tol
    of `1e-6` so per-coordinate KKT residuals can be on that order.
    """
    rng = np.random.default_rng(123)
    p = x.shape[1]
    # Mix of light and heavy penalty weights; bounded away from zero so
    # the rescaled design stays well-conditioned for the identity test.
    w = rng.uniform(0.5, 3.0, size=p)

    coefs_w = _fit(estimator_cls, x, y, lambdas=lambdas, weights=w, **kw)
    x_scaled = x / w[None, :]  # X·diag(1/w)
    coefs_scaled = _fit(estimator_cls, x_scaled, y, lambdas=lambdas, weights=None, **kw)
    # Rescale back: β_j = (β-on-X')_j / w_j
    coefs_recovered = coefs_scaled / w[None, :]

    max_diff = float(np.max(np.abs(coefs_w - coefs_recovered)))
    return {
        "name": "x_scaling_identity",
        "max_diff": max_diff,
        "passed": max_diff < tol,
        "tol": tol,
        "w_min": float(np.min(w)),
        "w_max": float(np.max(w)),
    }


def _entry_lambda(coef_path: np.ndarray, lambdas: np.ndarray, j: int, atol: float = 1e-8) -> float:
    """λ at which feature j first becomes non-zero, walking from λ_max
    (idx 0) down toward λ_min. Returns NaN if it never becomes active.

    The path is row-indexed by descending λ; we report the largest λ
    where `|β_j| > atol`. If the very first row already has β_j ≠ 0,
    we return `lambdas[0]` (best we can do — the active set was already
    open at the top of the grid)."""
    mask = np.abs(coef_path[:, j]) > atol
    if not np.any(mask):
        return float("nan")
    return float(lambdas[np.argmax(mask)])


def check_entry_lambda_scales_with_weight(
    estimator_cls, x, y, lambdas, true_idx, **kw
) -> dict:
    """(3) Weighting feature j by `w_j` should shift the entry-λ
    (largest λ at which the feature becomes active) by `1/w_j`. This
    is exact at the KKT condition `|grad_j(β=0)| = w_j · λ` and holds
    for every penalty whose subdifferential at zero is `[−1, 1]`
    (lasso, MCP, SCAD all satisfy this).

    We pick an *inactive* truly-zero feature near a strong active one,
    so the entry-λ lies inside the grid for both weighted and
    unweighted runs. Tolerance is relative — grids are geometric so
    the entry-λ ratio is measured on log scale and rounding to the
    nearest grid index typically gives a factor-of-2 imprecision; we
    pick `w = 2.0` to make the expected shift comfortably resolvable.
    """
    p = x.shape[1]
    # Pick a truly-inactive feature so its entry-λ sits cleanly inside
    # the path (active features enter at the top of the grid where the
    # shift is not measurable). Find one by elimination.
    inactive = [j for j in range(p) if j not in set(true_idx)]
    j = int(inactive[0])
    coefs_default = _fit(estimator_cls, x, y, lambdas=lambdas, weights=None, **kw)
    w = 2.0
    w_vec = np.ones(p)
    w_vec[j] = w
    coefs_weighted = _fit(estimator_cls, x, y, lambdas=lambdas, weights=w_vec, **kw)

    lam_unweighted = _entry_lambda(coefs_default, lambdas, j)
    lam_weighted = _entry_lambda(coefs_weighted, lambdas, j)

    if np.isnan(lam_unweighted) or np.isnan(lam_weighted):
        # Feature never entered under one of the fits. Either nothing
        # to compare, or w pushed it past the grid.
        return {
            "name": "entry_lambda_scales_with_weight",
            "feature_idx": j,
            "weight": w,
            "entry_lambda_unweighted": lam_unweighted,
            "entry_lambda_weighted": lam_weighted,
            "passed": False,
            "note": "feature never entered the active set under one of the fits",
        }

    ratio = lam_unweighted / lam_weighted  # expected ≈ w
    # Grids are geometric with 30 steps over 2 decades, so adjacent λ
    # differ by a factor of `10^(2/29) ≈ 1.17`. A factor-of-2 expected
    # shift therefore resolves to within ±20 %.
    rel_err = abs(ratio - w) / w
    return {
        "name": "entry_lambda_scales_with_weight",
        "feature_idx": j,
        "weight": w,
        "entry_lambda_unweighted": lam_unweighted,
        "entry_lambda_weighted": lam_weighted,
        "ratio": ratio,
        "expected_ratio": w,
        "rel_err": rel_err,
        "passed": rel_err < 0.25,
    }


def check_infinite_weight_pins_to_zero(estimator_cls, x, y, lambdas, true_idx, tol=1e-10, **kw) -> dict:
    """(4) `w_j` very large should pin β_j = 0 *for every λ in the grid*,
    even when feature j is truly active in the underlying signal.

    We use `1e8` rather than `np.inf` to avoid arithmetic surprises in
    weighted-prox arithmetic — the effective penalty is then orders of
    magnitude larger than the largest reasonable λ in any path."""
    p = x.shape[1]
    j = int(true_idx[0])
    w_inf = np.ones(p)
    w_inf[j] = 1e8
    coefs = _fit(estimator_cls, x, y, lambdas=lambdas, weights=w_inf, **kw)
    max_abs_j = float(np.max(np.abs(coefs[:, j])))
    return {
        "name": "infinite_weight_pins_to_zero",
        "feature_idx": j,
        "max_abs_coef_j_across_path": max_abs_j,
        "passed": max_abs_j < tol,
        "tol": tol,
    }


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run() -> dict:
    from skein_glm import ElasticNetPathRegressor, MCPPathRegressor, SCADPathRegressor

    x, y, beta, true_idx = _build_problem()
    lambdas = _make_lambda_grid(x, y, n_lambdas=30)

    logger.info(
        "problem: n=%d p=%d k_active=%d  λ_max=%.4f  λ_min=%.4f",
        x.shape[0], x.shape[1], len(true_idx), lambdas[0], lambdas[-1],
    )
    logger.info("true active features: %s", sorted(true_idx.tolist()))

    penalties: list[tuple[str, type, dict, bool]] = [
        # (label, class, ctor kwargs, supports_x_scaling_identity)
        ("lasso (EN α=1)", ElasticNetPathRegressor, {"alpha": 1.0}, True),
        ("MCP γ=3",        MCPPathRegressor,        {"gamma": 3.0},  False),
        ("SCAD a=3.7",     SCADPathRegressor,       {"a": 3.7},      False),
    ]

    results: dict[str, list[dict]] = {}
    for label, cls, kw, supports_xs in penalties:
        logger.info("=== %s ===", label)
        rows = [
            check_uniform_weights_equivalent(cls, x, y, lambdas, **kw),
            check_entry_lambda_scales_with_weight(cls, x, y, lambdas, true_idx, **kw),
            check_infinite_weight_pins_to_zero(cls, x, y, lambdas, true_idx, **kw),
        ]
        if supports_xs:
            rows.insert(1, check_x_scaling_identity(cls, x, y, lambdas, **kw))
        results[label] = rows
        for row in rows:
            mark = "PASS" if row["passed"] else "FAIL"
            extras = {k: v for k, v in row.items() if k not in {"name", "passed", "tol"}}
            logger.info("  [%s] %s  %s", mark, row["name"], extras)

    # Headline
    n_pass = sum(r["passed"] for rows in results.values() for r in rows)
    n_total = sum(len(rows) for rows in results.values())
    logger.info("summary: %d / %d invariants pass", n_pass, n_total)
    print()
    print(f"{'penalty':<18}  {'invariant':<42}  status")
    print("-" * 76)
    for label, rows in results.items():
        for row in rows:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"{label:<18}  {row['name']:<42}  {mark}")

    return {
        "scenario": "penalty_weights_ls",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": int(x.shape[0]),
        "p": int(x.shape[1]),
        "n_lambdas": int(len(lambdas)),
        "true_active": sorted(int(i) for i in true_idx),
        "penalties": {label: rows for label, rows in results.items()},
        "n_pass": n_pass,
        "n_total": n_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    payload = run()
    out = cc.write_results("penalty_weights_ls", payload)
    logger.info("wrote %s", out)
    # Exit code 1 if any invariant failed — useful for CI / quick sanity.
    return 0 if payload["n_pass"] == payload["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
