"""Collapse per-cell JSONL rows into a per-scenario aggregate JSON.

Aggregation does three things:
  1. Pool fit time / recovery / IC selection across seeds, grouped by
     (size, regime, package).
  2. Compute *cross-package agreement* (Jaccard / sign / rel-L2 per λ)
     by pairing each non-skein cell with the matching skein cell at
     the same (size, regime, seed) — requires re-fitting both runners
     so we can compare coefficient paths. Phase D handles this by
     persisting the coef_path on disk alongside the JSONL row when the
     `--store-coefs` flag is set.
  3. Refuse to mix host_ids in one aggregate (foot-gun guard).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from benches.v2.metrics import agreement as agree


def _load_cells(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coef_path_path(cell_jsonl: Path) -> Path:
    """The .npy companion file where the cell driver may have stashed
    a coefficient path (one float64 array of shape (n_lambdas, p))."""
    return cell_jsonl.with_suffix(".coefs.npy")


def _compute_agreement(rows: list[dict],
                       cell_paths: list[Path]) -> dict[str, dict[str, dict]]:
    """For each (scenario, size, regime, seed), pair skein vs each
    comparator and compute per-λ agreement metrics.

    Requires both cells to have persisted their coef_paths to disk.
    Returns nested dict: package -> metric -> list[per-λ values, mean across seeds].
    """
    # Build a lookup of (size, regime, seed, package) -> path on disk.
    by_key: dict[tuple, Path] = {}
    for row, p in zip(rows, cell_paths):
        if row.get("status") != "ok":
            continue
        coef_p = _coef_path_path(p)
        if coef_p.exists():
            by_key[(row["size"], row["regime"], row["seed"], row["package"])] = coef_p

    # For each (size, regime, seed), find the skein cell and pair it
    # with every other package in that triple.
    per_pkg: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    triples = sorted({(s, r, sd) for (s, r, sd, _) in by_key})
    for (size, regime, seed) in triples:
        skein_key = (size, regime, seed, "skein")
        if skein_key not in by_key:
            continue
        skein_coefs = np.load(by_key[skein_key])
        for (s, r, sd, pkg), path in by_key.items():
            if (s, r, sd) != (size, regime, seed) or pkg == "skein":
                continue
            try:
                other = np.load(path)
                if other.shape != skein_coefs.shape:
                    continue
                metrics = agree.per_lambda(skein_coefs, other)
                per_pkg[(size, regime, pkg)]["jaccard"].append(metrics["jaccard"])
                per_pkg[(size, regime, pkg)]["sign"].append(metrics["sign"])
                per_pkg[(size, regime, pkg)]["rel_l2"].append(metrics["rel_l2"])
            except Exception:
                continue

    # Average each metric across seeds.
    pooled: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for key, metric_dict in per_pkg.items():
        out_metric: dict[str, list[float]] = {}
        for m, runs in metric_dict.items():
            if not runs:
                continue
            n_lams = min(len(r) for r in runs)
            stacked = np.array([r[:n_lams] for r in runs])
            out_metric[m] = stacked.mean(axis=0).tolist()
        pooled[key] = out_metric
    return pooled


def aggregate(rows: list[dict]) -> dict:
    """Aggregate by (size, regime, package), pooling across seeds."""
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["size"], r["regime"], r["package"])
        groups[key].append(r)

    out: dict[str, list[dict]] = {"cells": []}
    for (size, regime, package), grp in sorted(groups.items()):
        times = [r["fit_time_s"] for r in grp]
        active = [r["active_set_size"] for r in grp]
        host_ids = sorted({r["host_id"] for r in grp})
        if len(host_ids) > 1:
            # Mixing host_ids in one aggregate is a foot-gun — flag loudly.
            raise RuntimeError(
                f"aggregate {(size, regime, package)} mixes host_ids: {host_ids}. "
                "Re-run all seeds on one host."
            )
        cell = {
            "size": size, "regime": regime, "package": package,
            "n_seeds": len(grp),
            "n": grp[0]["n"], "p": grp[0]["p"],
            "fit_time_s_median": statistics.median(times),
            "fit_time_s_min":    min(times),
            "fit_time_s_max":    max(times),
            "fit_time_s_iqr":    (statistics.quantiles(times, n=4)[2] -
                                  statistics.quantiles(times, n=4)[0])
                                 if len(times) >= 4 else 0.0,
            "active_set_size_median": statistics.median(active),
            "version": grp[0]["version"],
            "ladder_level": grp[0]["ladder_level"],
            "host_id": host_ids[0],
        }
        # Pool recovery metrics across seeds (mean of per-λ medians).
        rec = [r["recovery"] for r in grp if r.get("recovery")]
        if rec:
            # Each rec is {metric: [per-λ values]}. Take per-λ mean across seeds.
            metric_keys = set(rec[0].keys())
            pooled: dict[str, list[float]] = {}
            for k in metric_keys:
                arrs = [r[k] for r in rec if k in r]
                if arrs and len(set(len(a) for a in arrs)) == 1:
                    pooled[k] = [statistics.mean(vals) for vals in zip(*arrs)]
            cell["recovery_per_lambda_mean"] = pooled
            # And take a single headline: best F1 along the path.
            if "support_f1" in pooled and pooled["support_f1"]:
                cell["recovery_best_f1"] = max(pooled["support_f1"])

        # Pool IC selection across seeds: mean F1 / RMSE / size at each criterion.
        sel = [r["selection"] for r in grp
               if r.get("selection") and isinstance(r["selection"], dict)
                  and "aic" in r["selection"]]
        if sel:
            sel_agg: dict[str, dict[str, float]] = {}
            for crit in ("aic", "bic", "ebic"):
                rows = [s[crit] for s in sel if crit in s]
                if not rows:
                    continue
                sel_agg[crit] = {
                    "support_f1_mean":  statistics.mean(r["support_f1"]   for r in rows),
                    "beta_rmse_mean":   statistics.mean(r["beta_rmse"]    for r in rows),
                    "hat_support_size_mean": statistics.mean(r["hat_support_size"] for r in rows),
                    "lambda_index_mean": statistics.mean(r["lambda_index"] for r in rows),
                    "n_seeds":          len(rows),
                }
            cell["selection_mean"] = sel_agg
        out["cells"].append(cell)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = _load_cells(a.inputs)
    agg = aggregate(rows)
    agg["scenario"] = a.scenario

    # Cross-package agreement: pair each non-skein cell with its
    # matching skein cell and compute per-λ Jaccard / sign / rel-L2.
    agreement = _compute_agreement(rows, a.inputs)
    if agreement:
        for cell in agg["cells"]:
            key = (cell["size"], cell["regime"], cell["package"])
            if key in agreement:
                cell["agreement_vs_skein_mean"] = agreement[key]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(agg, indent=2))
    n_agree = sum(1 for c in agg["cells"] if "agreement_vs_skein_mean" in c)
    print(f"{a.scenario}: aggregated {len(rows)} cells -> "
          f"{len(agg['cells'])} groups ({n_agree} with cross-pkg agreement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
