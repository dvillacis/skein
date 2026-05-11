"""Per-sample weights correctness check (Gaussian LS).

Verifies the `sample_weights=` kwarg newly exposed on `MCPPathRegressor`,
`SCADPathRegressor`, `ElasticNetPathRegressor` (and the GLM siblings).
The Rust core has had per-sample weights since M3.1; this script
confirms the Python plumbing fixes the "I couldn't see the effect"
symptom — the API now accepts and honors them.

**Normalisation convention (important)**: skein computes the loss as
`(1/2n) · Σ_i sw_i · (X_iβ - y_i)²` where `n` is the original sample
count. `Σ sw_i` is *not* used as the denominator (glmnet does that;
skein does not). One consequence: passing `sw = c · ones` is
*equivalent* to fitting unweighted at `λ' = λ/c` (the loss scales
by c, so the penalty's relative weight scales by 1/c). The
invariants below are stated in terms of this convention.

Invariants:

  (1) Uniform weights `sw = 1` ≡ no weights argument. Compared on a
      *shared* λ-grid so the two paths see the same problem at each
      index. (The auto-grid would differ — λ_max is computed against
      the augmented design when sw is set, vs. the centered design
      when it isn't; different problems, different grids.)

  (2) λ-scale identity (**lasso only**): `sw = c · ones at λ` ≡
      `no-sw at λ/c`. Stems from the (1/2n)-normalisation: scaling
      weights uniformly scales the loss, which is equivalent to
      inverse-scaling the L1 penalty. MCP / SCAD don't satisfy this
      because their penalty has a `γλ` knee whose shape changes when
      λ rescales; matching would require also rescaling γ.

  (3) Zero-weight ≡ dropped sample, **lasso** (with correct λ
      rescaling): setting `sw_i = 0` for `n_drop` rows and fitting
      at `λ` is equivalent to fitting unweighted on the kept rows at
      `λ' = λ · (n / n_kept)`. The factor `n / n_kept > 1` arises
      because dropping rows tightens the loss normalisation
      (from `1/n` to `1/n_kept`), so the penalty must scale up
      to keep the loss-vs-penalty ratio constant.

  (4) Effect is visible: a *biased* contamination problem (move a few
      rows' y far from the truth) recovers cleaner coefficients when
      those rows are down-weighted than when treated uniformly. This
      is the qualitative "I can see the effect" smoke test.

Run as a script:

    python benches/correctness/sample_weights_ls.py
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

logger = logging.getLogger("benches.correctness.sample_weights_ls")


def _build_problem(n: int = 500, p: int = 30, k_active: int = 5, snr: float = 5.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta = np.zeros(p)
    idx = rng.choice(p, size=k_active, replace=False)
    beta[idx] = rng.choice([-1.0, 1.0], size=k_active) * rng.uniform(0.8, 2.0, size=k_active)
    signal = x @ beta
    noise = (float(np.std(signal)) / snr) * rng.standard_normal(n)
    y = signal + noise
    return x, y, beta, idx


# ---------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------

def check_ones_equiv_no_weights(estimator_cls, x, y, lambdas, tol=1e-5, **kw) -> dict:
    """(1) sw = 1·ones must produce ≈ identical coefs to no sw on the
    *same* explicit λ-grid (tolerance accommodates convergence noise
    + the structural difference that the sw branch uses augmented-X
    while the no-sw branch standardize-centers — the two converge to
    the same optimum but via different intermediate representations).
    """
    n = x.shape[0]
    coefs_default = np.asarray(
        estimator_cls(lambdas=lambdas, **kw).fit(x, y).coefs_
    )
    coefs_sw = np.asarray(
        estimator_cls(lambdas=lambdas, sample_weights=np.ones(n), **kw).fit(x, y).coefs_
    )
    max_diff = float(np.max(np.abs(coefs_default - coefs_sw)))
    return {
        "name": "ones_equivalent_to_no_weights",
        "max_diff": max_diff,
        "passed": max_diff < tol,
        "tol": tol,
    }


def check_lambda_scale_identity(estimator_cls, x, y, lambdas, tol=1e-4, **kw) -> dict:
    """(2) sw = c · ones at λ ≡ no-sw at λ/c. The reason: skein
    normalises by `n`, so uniform sw = c scales the loss by c, and
    minimising `c·L(β) + λ·pen(β)` is equivalent to minimising
    `L(β) + (λ/c)·pen(β)`.
    """
    n = x.shape[0]
    c = 3.0
    coefs_sw = np.asarray(
        estimator_cls(lambdas=lambdas, sample_weights=c*np.ones(n), **kw).fit(x, y).coefs_
    )
    coefs_no = np.asarray(
        estimator_cls(lambdas=lambdas / c, **kw).fit(x, y).coefs_
    )
    max_diff = float(np.max(np.abs(coefs_sw - coefs_no)))
    return {
        "name": "lambda_scale_identity_uniform_weight",
        "scale_factor": c,
        "max_diff": max_diff,
        "passed": max_diff < tol,
        "tol": tol,
    }


def check_zero_weight_equiv_dropped(estimator_cls, x, y, lambdas, tol=1e-4, **kw) -> dict:
    """(3) sw_i = 0 for a subset of rows at λ ≡ unweighted fit on the
    kept rows at `λ · (n / n_kept)`. Skein normalises by `n` so the
    masked fit has effective loss `(1/2n) Σ_{kept} r²`; the dropped
    fit has `(1/2 n_kept) Σ_{kept} r²`. Equal-minimiser condition
    requires the penalty ratio to match — i.e. `λ_dropped = λ · n / n_kept`.
    """
    rng = np.random.default_rng(0)
    n = x.shape[0]
    n_drop = n // 5
    drop_idx = rng.choice(n, n_drop, replace=False)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[drop_idx] = False
    n_kept = int(keep_mask.sum())
    scale = n / n_kept

    sw = np.where(keep_mask, 1.0, 0.0)
    coefs_sw = np.asarray(
        estimator_cls(lambdas=lambdas, sample_weights=sw, **kw).fit(x, y).coefs_
    )
    coefs_dropped = np.asarray(
        estimator_cls(lambdas=lambdas * scale, **kw)
        .fit(x[keep_mask], y[keep_mask]).coefs_
    )
    max_diff = float(np.max(np.abs(coefs_sw - coefs_dropped)))
    return {
        "name": "zero_weight_equivalent_to_dropped_rows",
        "n_dropped": int(n_drop),
        "n_kept": n_kept,
        "lambda_scale": scale,
        "max_diff": max_diff,
        "passed": max_diff < tol,
        "tol": tol,
    }


def check_downweighting_improves_recovery(estimator_cls, x, y, true_beta, **kw) -> dict:
    """(4) Qualitative: a small set of rows with biased y (added bias
    of magnitude 5σ) should produce closer-to-truth β under sw=0 on
    those rows than under uniform sw. The "effect is visible" test —
    this is what the user actually wants to see when they pass
    sample_weights.
    """
    rng = np.random.default_rng(42)
    n = x.shape[0]
    n_biased = n // 20
    bias_idx = rng.choice(n, n_biased, replace=False)
    # Contaminate y on bias rows.
    bias = 5.0 * np.std(y)
    y_contam = y.copy()
    y_contam[bias_idx] += bias

    lambdas = np.geomspace(0.5, 0.005, 10)
    m_unweighted = estimator_cls(lambdas=lambdas, **kw).fit(x, y_contam)
    sw = np.ones(n)
    sw[bias_idx] = 0.0
    m_weighted = estimator_cls(lambdas=lambdas, sample_weights=sw, **kw).fit(x, y_contam)

    # Compare at the smallest λ (last row).
    err_unweighted = float(np.linalg.norm(m_unweighted.coefs_[-1] - true_beta))
    err_weighted = float(np.linalg.norm(m_weighted.coefs_[-1] - true_beta))
    return {
        "name": "downweighting_biased_rows_improves_recovery",
        "n_biased": int(n_biased),
        "bias_magnitude": float(bias),
        "err_to_truth_unweighted": err_unweighted,
        "err_to_truth_sw0_on_biased": err_weighted,
        "passed": err_weighted < err_unweighted * 0.9,  # 10 % improvement min
        "improvement_ratio": err_weighted / max(err_unweighted, 1e-12),
    }


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run() -> dict:
    from skein_glm import ElasticNetPathRegressor, MCPPathRegressor, SCADPathRegressor

    x, y, beta, true_idx = _build_problem()
    # Shared explicit λ-grid so weighted/unweighted/replicated fits all
    # see the same problem at each index.
    lambdas = np.geomspace(1.0, 0.01, 20)

    logger.info(
        "problem: n=%d p=%d k_active=%d  shared λ-grid=[%.3g..%.3g]×%d",
        x.shape[0], x.shape[1], len(true_idx),
        lambdas[0], lambdas[-1], len(lambdas),
    )

    penalties: list[tuple[str, type, dict, bool]] = [
        # (label, class, ctor kwargs, supports_lambda_scale_identity)
        ("lasso (EN α=1)", ElasticNetPathRegressor, {"alpha": 1.0}, True),
        ("MCP γ=3",        MCPPathRegressor,        {"gamma": 3.0},  False),
        ("SCAD a=3.7",     SCADPathRegressor,       {"a": 3.7},      False),
    ]

    results: dict[str, list[dict]] = {}
    for label, cls, kw, supports_lsi in penalties:
        logger.info("=== %s ===", label)
        rows = [
            check_ones_equiv_no_weights(cls, x, y, lambdas, **kw),
            check_downweighting_improves_recovery(cls, x, y, beta, **kw),
        ]
        if supports_lsi:
            # λ-scale identity (and zero-weight ≡ dropped at scaled λ)
            # are lasso-only: MCP/SCAD have a γλ knee that does not
            # rescale cleanly under loss scaling without simultaneously
            # adjusting γ — out of scope for this correctness check.
            rows.append(check_lambda_scale_identity(cls, x, y, lambdas, **kw))
            rows.append(check_zero_weight_equiv_dropped(cls, x, y, lambdas, **kw))
        results[label] = rows
        for row in rows:
            mark = "PASS" if row["passed"] else "FAIL"
            extras = {k: v for k, v in row.items() if k not in {"name", "passed", "tol"}}
            logger.info("  [%s] %s  %s", mark, row["name"], extras)

    n_pass = sum(r["passed"] for rows in results.values() for r in rows)
    n_total = sum(len(rows) for rows in results.values())
    logger.info("summary: %d / %d invariants pass", n_pass, n_total)
    print()
    print(f"{'penalty':<18}  {'invariant':<50}  status")
    print("-" * 84)
    for label, rows in results.items():
        for row in rows:
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"{label:<18}  {row['name']:<50}  {mark}")

    return {
        "scenario": "sample_weights_ls",
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
    out = cc.write_results("sample_weights_ls", payload)
    logger.info("wrote %s", out)
    return 0 if payload["n_pass"] == payload["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
