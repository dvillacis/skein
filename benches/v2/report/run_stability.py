"""F8 — Stability-selection FDR/power vs Meinshausen-Bühlmann bound.

For each (seed, threshold) pair, run StabilitySelection on synthetic
data with known support; record empirical FDR and power; compare
against the MB nominal upper-bound:

    E[V] ≤ q² / ((2τ - 1) · p)
    FDR ≤ E[V] / E[S] ≈ q² / ((2τ - 1) · p · q) = q / ((2τ - 1) · p)

where q is the *average* number of selected features per bootstrap and
S is the selected stable set at threshold τ.

Output: one JSONL row per (scenario, snr, seed) with per-threshold
empirical FDR + power and the nominal bound.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _run_one(*, n: int, p: int, snr: float, seed: int,
             n_bootstraps: int, sample_fraction: float,
             tol: float, n_lambdas: int) -> dict:
    from benches.v2.simulators import linear_truth
    from skein_glm.stability import StabilitySelection
    from skein_glm.estimators import MCPPathRegressor

    problem = linear_truth.make(n=n, p=p, seed=seed, snr=snr)
    true_supp = problem.beta_true != 0

    t0 = time.perf_counter()
    ss = StabilitySelection(
        base_estimator=MCPPathRegressor(
            n_lambdas=n_lambdas, lambda_min_ratio=1e-2, tol=tol,
        ),
        n_bootstraps=n_bootstraps, sample_fraction=sample_fraction,
        random_state=seed,
    )
    ss.fit(problem.x, problem.y)
    elapsed = time.perf_counter() - t0

    max_probs = np.asarray(ss.max_probabilities_)
    # Average number of features selected per bootstrap, averaged across λ.
    # selection_probabilities_ is (n_lambdas, p) — per-λ selection freq.
    sel_probs = np.asarray(ss.selection_probabilities_)
    q_per_lambda = sel_probs.sum(axis=1)
    q_bar = float(q_per_lambda.mean())   # mean active-set size across λ

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    rows = []
    for tau in thresholds:
        selected = max_probs >= tau
        tp = int(np.sum(selected & true_supp))
        fp = int(np.sum(selected & ~true_supp))
        fn = int(np.sum(~selected & true_supp))
        fdr = fp / (tp + fp) if (tp + fp) else 0.0
        power = tp / (tp + fn) if (tp + fn) else 0.0
        # Meinshausen-Bühlmann bound on E[V] (number of false selections).
        # E[V] ≤ q_bar² / ((2τ − 1) · p)  when τ > 0.5.
        if tau > 0.5:
            ev_bound = (q_bar ** 2) / ((2 * tau - 1) * p)
            fdr_bound = ev_bound / max(int(np.sum(selected)), 1)
        else:
            ev_bound = float("inf")
            fdr_bound = 1.0
        rows.append({
            "threshold": tau, "fdr": fdr, "power": power,
            "fp": fp, "tp": tp, "fn": fn,
            "ev_bound": ev_bound, "fdr_bound": fdr_bound,
        })
    return {
        "n": n, "p": p, "snr": snr, "seed": seed,
        "true_support_size": int(true_supp.sum()),
        "n_bootstraps": n_bootstraps,
        "sample_fraction": sample_fraction,
        "q_bar": q_bar,
        "fit_time_s": elapsed,
        "by_threshold": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--p", type=int, default=50)
    ap.add_argument("--snrs", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-bootstraps", type=int, default=50)
    ap.add_argument("--sample-fraction", type=float, default=0.5)
    ap.add_argument("--n-lambdas", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    out = a.out_dir / "stability.jsonl"
    with out.open("w") as f:
        for snr in a.snrs:
            for seed in a.seeds:
                row = _run_one(
                    n=a.n, p=a.p, snr=snr, seed=seed,
                    n_bootstraps=a.n_bootstraps,
                    sample_fraction=a.sample_fraction,
                    tol=a.tol, n_lambdas=a.n_lambdas,
                )
                f.write(json.dumps(row) + "\n")
                print(f"  snr={snr:.1f} seed={seed}: "
                      f"q_bar={row['q_bar']:.1f}  "
                      f"τ=0.6 FDR={row['by_threshold'][2]['fdr']:.3f} "
                      f"power={row['by_threshold'][2]['power']:.3f}  "
                      f"({row['fit_time_s']:.1f}s)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
