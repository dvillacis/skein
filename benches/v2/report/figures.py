"""Figure builders. Each subcommand consumes aggregate JSON files and
emits one PDF.

Style is locked here (single font, single palette) so every figure in
the paper looks consistent. Don't override mpl rcparams from the
builders themselves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,           # editable text in the PDF
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PALETTE = {
    "skein":   "#1b4f72",
    "sklearn": "#7d3c98",
    "skglm":   "#196f3d",
    "celer":   "#7e5109",
    "glmnet":  "#922b21",
    "ncvreg":  "#b03a2e",
    "grpreg":  "#c0392b",
}


def _load(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        out.append(json.loads(p.read_text()))
    return out


def headline_timing(inputs: list[Path], out: Path) -> None:
    """F2 — grouped bars per scenario; bars = packages; size = medium."""
    aggs = _load(inputs)
    rows = []
    for agg in aggs:
        for cell in agg["cells"]:
            if cell["size"] != "medium" or cell["regime"] != "deep":
                continue
            rows.append({
                "scenario": agg["scenario"],
                "package":  cell["package"],
                "median":   cell["fit_time_s_median"],
                "min":      cell["fit_time_s_min"],
                "max":      cell["fit_time_s_max"],
            })

    if not rows:
        # Phase A: data may be empty; still produce a placeholder so the
        # DAG completes.
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no headline medium / dense cells available",
                ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    scenarios = sorted({r["scenario"] for r in rows})
    packages  = sorted({r["package"]  for r in rows})
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(scenarios)), 3.5))
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(packages))
    for i, pkg in enumerate(packages):
        ys = [next((r["median"] for r in rows
                    if r["scenario"] == s and r["package"] == pkg), np.nan)
              for s in scenarios]
        ymins = [next((r["min"] for r in rows
                       if r["scenario"] == s and r["package"] == pkg), np.nan)
                 for s in scenarios]
        ymaxs = [next((r["max"] for r in rows
                       if r["scenario"] == s and r["package"] == pkg), np.nan)
                 for s in scenarios]
        ys = np.asarray(ys, float)
        err_lo = np.maximum(ys - np.asarray(ymins, float), 0)
        err_hi = np.maximum(np.asarray(ymaxs, float) - ys, 0)
        bars = ax.bar(x + i * width - 0.4 + width / 2, ys, width,
                      yerr=[err_lo, err_hi], capsize=2,
                      color=PALETTE.get(pkg, "#444"), label=pkg)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.set_ylabel("Median fit time (s)")
    ax.set_title("F2 — Headline timings, medium size, dense regime")
    ax.legend(ncol=min(4, len(packages)), frameon=False, loc="upper left")
    ax.set_yscale("log")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def scaling_curves(inputs: list[Path], out: Path) -> None:
    """F3 — log-log wall-clock vs n (p fixed at medium p), per scenario."""
    aggs = _load(inputs)
    fig, ax = plt.subplots(figsize=(6, 4))
    for agg in aggs:
        # Pull cells where regime is 'deep' and group by package.
        by_pkg: dict[str, list[tuple[int, float]]] = {}
        for cell in agg["cells"]:
            if cell["regime"] != "deep":
                continue
            by_pkg.setdefault(cell["package"], []).append(
                (cell["n"], cell["fit_time_s_median"]))
        for pkg, pts in by_pkg.items():
            if len(pts) < 2:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o",
                    color=PALETTE.get(pkg, "#444"),
                    label=f"{agg['scenario']}/{pkg}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("Median fit time (s)")
    ax.set_title("F3 — Scaling with n, dense regime")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(ncol=2, frameon=False, fontsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def stability_fdr_power(inputs: list[Path], out: Path) -> None:
    """F8 — empirical FDR + power vs threshold τ, with the MB nominal
    error-control bound overlaid as a dotted line.

    Two panels (one per SNR by default). Each panel:
      - x-axis: threshold τ ∈ [0.5, 0.95]
      - left y: empirical FDR (mean across seeds, shaded ± 1 s.d.)
      - right y: power (mean across seeds, dashed)
      - dotted: MB nominal FDR bound
    """
    rows: list[dict] = []
    for p in inputs:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no stability data", ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    snrs = sorted({r["snr"] for r in rows})
    fig, axes = plt.subplots(1, len(snrs),
                             figsize=(3.8 * len(snrs), 3.0),
                             squeeze=False, sharey=True)
    for i, snr in enumerate(snrs):
        ax = axes[0][i]
        sub = [r for r in rows if r["snr"] == snr]
        thresholds = np.array([t["threshold"] for t in sub[0]["by_threshold"]])
        # Per-threshold matrix: rows = seeds, cols = thresholds.
        fdr_mat = np.array([[t["fdr"] for t in r["by_threshold"]] for r in sub])
        power_mat = np.array([[t["power"] for t in r["by_threshold"]] for r in sub])
        bound_mat = np.array(
            [[min(t["fdr_bound"], 1.0) for t in r["by_threshold"]] for r in sub])
        fdr_mean, fdr_std = fdr_mat.mean(axis=0), fdr_mat.std(axis=0)
        power_mean = power_mat.mean(axis=0)
        bound_mean = bound_mat.mean(axis=0)

        ax.fill_between(thresholds, fdr_mean - fdr_std, fdr_mean + fdr_std,
                        color="#b03a2e", alpha=0.18)
        ax.plot(thresholds, fdr_mean, color="#b03a2e", marker="o",
                label="empirical FDR")
        ax.plot(thresholds, bound_mean, color="#922b21", linestyle=":",
                label="MB bound")
        ax.plot(thresholds, power_mean, color="#1b4f72", linestyle="--",
                marker="s", label="power")
        ax.set_xlabel(r"threshold $\tau$")
        if i == 0:
            ax.set_ylabel("rate")
            ax.legend(fontsize=7, frameon=False, loc="center right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"SNR = {snr}", fontsize=9)
    fig.suptitle("F8 — Stability selection: FDR vs MB bound, and power",
                 fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def cv_parallel_speedup(inputs: list[Path], out: Path) -> None:
    """F10 — CV wall-clock & speedup vs serial, as a function of n_jobs.

    One panel per (n_folds). Left: log-scale wall-clock; right: speedup.
    """
    rows: list[dict] = []
    for p in inputs:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no CV-parallel data", ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    folds_list = sorted({r["n_folds"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    ax_t, ax_s = axes
    for n_folds in folds_list:
        sub = sorted([r for r in rows if r["n_folds"] == n_folds],
                     key=lambda r: r["n_jobs"])
        jobs = np.array([r["n_jobs"] for r in sub])
        times = np.array([r["fit_time_s_median"] for r in sub])
        serial = times[jobs == 1][0] if any(jobs == 1) else times[0]
        speedup = serial / times
        ax_t.plot(jobs, times, marker="o", label=f"{n_folds} folds")
        ax_s.plot(jobs, speedup, marker="o", label=f"{n_folds} folds")
    # Ideal-speedup reference.
    max_jobs = max(rows, key=lambda r: r["n_jobs"])["n_jobs"]
    xs = np.arange(1, max_jobs + 1)
    ax_s.plot(xs, xs, color="#999", linestyle=":", label="ideal y = x")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xlabel("n_jobs")
    ax_t.set_ylabel("wall-clock (s, log)")
    ax_t.set_title("CV wall-clock", fontsize=9)
    ax_t.legend(fontsize=7, frameon=False)
    ax_s.set_xscale("log", base=2)
    ax_s.set_xlabel("n_jobs")
    ax_s.set_ylabel(r"speedup vs n\_jobs=1")
    ax_s.set_title("CV speedup", fontsize=9)
    ax_s.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("F10 — Threaded CV wall-clock and speedup", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def screening_parallel(inputs: list[Path], out: Path) -> None:
    """F9 — Screening + Jacobi-parallel ablation from criterion JSON.

    Expects `report/ingest_criterion.py` to have produced a tidy JSONL
    with rows {scenario, group, mode, median_ns}. Renders a bar chart.
    """
    rows: list[dict] = []
    for p in inputs:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no criterion data available\n"
                          "(run: cargo bench -p skein-core then\n"
                          "python -m benches.v2.report.ingest_criterion)",
                ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    groups = sorted({r["group"] for r in rows})
    fig, axes = plt.subplots(1, len(groups),
                             figsize=(3.5 * len(groups), 3.0),
                             squeeze=False)
    for i, group in enumerate(groups):
        ax = axes[0][i]
        sub = sorted([r for r in rows if r["group"] == group],
                     key=lambda r: r["mode"])
        modes = [r["mode"] for r in sub]
        times_ms = [r["median_ns"] / 1e6 for r in sub]
        colors = ["#1b4f72", "#196f3d", "#7e5109", "#922b21"][:len(modes)]
        ax.bar(modes, times_ms, color=colors, edgecolor="white")
        ax.set_yscale("log")
        ax.set_title(group, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
        if i == 0:
            ax.set_ylabel("median time (ms, log)")
    fig.suptitle("F9 — Screening + Jacobi-parallel ablation (criterion microbench)",
                 fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def realdata_boxplots(inputs: list[Path], out: Path) -> None:
    """F6 — per-dataset CV-deviance boxplots, one panel per dataset,
    bars grouped by estimator. The cv-driver writes one JSONL row per
    (dataset, estimator) with per-fold deviance at λ_min.
    """
    rows = []
    for p in inputs:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows
            if r.get("fold_deviance_at_lambda_min") is not None]
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no real-dataset CV results", ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    datasets = sorted({r["dataset"] for r in rows})
    n_panels = len(datasets)
    cols = min(2, n_panels)
    rws = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(4.0 * cols, 2.7 * rws),
                             squeeze=False)
    for k, ds in enumerate(datasets):
        ax = axes[k // cols][k % cols]
        sub = sorted([r for r in rows if r["dataset"] == ds],
                     key=lambda r: r["estimator"])
        labels = [r["estimator"] for r in sub]
        data = [r["fold_deviance_at_lambda_min"] for r in sub]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        widths=0.55, medianprops=dict(color="white"))
        for patch, est in zip(bp["boxes"], labels):
            patch.set_facecolor("#1b4f72" if "mcp" in est or "scad" in est
                                else "#196f3d")
            patch.set_alpha(0.85)
        ax.set_title(f"{ds} (n={sub[0]['n']}, p={sub[0]['p']}, family={sub[0]['family']})",
                     fontsize=8)
        ax.set_ylabel("CV deviance at $\\lambda_{\\min}$")
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    for k in range(n_panels, rws * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.suptitle("F6 — Real-dataset CV deviance, per fold", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def recovery_curves(inputs: list[Path], out: Path) -> None:
    """F5 — support F1 / β-RMSE vs λ-index, per scenario × package.

    Pulls the aggregated per-λ recovery (mean across seeds) from each
    aggregate. One row per scenario, two columns: F1, RMSE.
    """
    aggs = [a for a in _load(inputs) if any(
        c.get("recovery_per_lambda_mean") for c in a["cells"])]
    if not aggs:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no recovery data available",
                ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    n = len(aggs)
    fig, axes = plt.subplots(n, 2, figsize=(7, 1.7 * n + 0.5), squeeze=False)
    for i, agg in enumerate(aggs):
        ax_f1, ax_rmse = axes[i]
        for cell in agg["cells"]:
            if cell["regime"] != "deep":
                continue
            rec = cell.get("recovery_per_lambda_mean") or {}
            f1 = rec.get("support_f1") or []
            rmse = rec.get("beta_rmse") or []
            if not f1 or not rmse:
                continue
            label = f"{cell['package']} (n={cell['n']})"
            color = PALETTE.get(cell["package"], "#444")
            ax_f1.plot(range(len(f1)), f1, color=color, alpha=0.6, label=label)
            ax_rmse.plot(range(len(rmse)), rmse, color=color, alpha=0.6, label=label)
        ax_f1.set_ylim(-0.05, 1.05)
        ax_f1.set_ylabel("support F1")
        ax_rmse.set_yscale("log")
        ax_rmse.set_ylabel(r"$\|\hat\beta - \beta^*\|_{\mathrm{RMSE}}$")
        ax_f1.set_title(agg["scenario"])
        if i == n - 1:
            ax_f1.set_xlabel(r"$\lambda$-index (dense → sparse)")
            ax_rmse.set_xlabel(r"$\lambda$-index")
        if i == 0:
            ax_f1.legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("F5 — Support-recovery curves (mean across seeds)", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def ic_selection(inputs: list[Path], out: Path) -> None:
    """F7 — IC selection accuracy: F1 at AIC/BIC/EBIC-selected λ, per scenario × n."""
    rows = []
    for agg in _load(inputs):
        for cell in agg["cells"]:
            sm = cell.get("selection_mean") or {}
            for crit, vals in sm.items():
                rows.append({
                    "scenario": agg["scenario"],
                    "package":  cell["package"],
                    "regime":   cell["regime"],
                    "n":        cell["n"],
                    "criterion": crit,
                    "support_f1": vals["support_f1_mean"],
                    "rmse": vals["beta_rmse_mean"],
                })
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no IC selection data available",
                ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    # One panel per (scenario, regime); bars grouped by criterion.
    scenarios = sorted({(r["scenario"], r["regime"]) for r in rows})
    n_panels = len(scenarios)
    cols = min(3, n_panels)
    rws = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(3.0 * cols, 2.4 * rws),
                             squeeze=False)
    crit_colors = {"aic": "#7d3c98", "bic": "#196f3d", "ebic": "#196f3d"}
    crit_hatch = {"aic": "", "bic": "", "ebic": "//"}
    for k, (scen, regime) in enumerate(scenarios):
        ax = axes[k // cols][k % cols]
        sub = [r for r in rows if r["scenario"] == scen and r["regime"] == regime]
        packages = sorted({r["package"] for r in sub})
        x = np.arange(len(packages))
        width = 0.27
        for i, crit in enumerate(("aic", "bic", "ebic")):
            ys = [next((r["support_f1"] for r in sub
                        if r["package"] == p and r["criterion"] == crit), 0)
                  for p in packages]
            ax.bar(x + (i - 1) * width, ys, width,
                   color=crit_colors[crit], hatch=crit_hatch[crit],
                   edgecolor="white", label=crit.upper() if k == 0 else None)
        ax.set_title(f"{scen} / {regime}", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(packages, fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        if k % cols == 0:
            ax.set_ylabel("support F1 @ selected λ")
    # Hide any leftover empty panels.
    for k in range(n_panels, rws * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=8)
    fig.suptitle("F7 — Support-F1 at IC-selected λ", fontsize=10, y=1.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def agreement(inputs: list[Path], out: Path) -> None:
    """F4 — per-λ cross-package agreement (Jaccard / sign / rel-L2).

    Reads `agreement_per_lambda` (mean across seeds) from each
    aggregate, which the cross-package aggregator computes by pairing
    skein vs each available comparator at the same (scenario, size,
    regime, seed).
    """
    rows = []
    for agg in _load(inputs):
        for cell in agg["cells"]:
            comp = cell.get("agreement_vs_skein_mean") or {}
            if not comp:
                continue
            for metric, vals in comp.items():
                rows.append({
                    "scenario": agg["scenario"],
                    "comparator": cell["package"],
                    "regime":     cell["regime"],
                    "size":       cell["size"],
                    "metric":     metric,
                    "values":     vals,
                })
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                "no cross-package agreement available\n"
                "(needs at least one external runner per scenario)",
                ha="center", va="center")
        ax.set_axis_off()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        return

    scenarios = sorted({r["scenario"] for r in rows})
    fig, axes = plt.subplots(len(scenarios), 3,
                             figsize=(8, 1.7 * len(scenarios) + 0.5),
                             squeeze=False)
    metric_order = ("jaccard", "sign", "rel_l2")
    metric_titles = {"jaccard": "Jaccard ↑", "sign": "Sign agreement ↑",
                     "rel_l2": "Rel L2 ↓"}
    for i, scen in enumerate(scenarios):
        scen_rows = [r for r in rows if r["scenario"] == scen]
        comparators = sorted({r["comparator"] for r in scen_rows})
        for j, metric in enumerate(metric_order):
            ax = axes[i][j]
            for comp in comparators:
                ys = next((r["values"] for r in scen_rows
                           if r["comparator"] == comp
                           and r["metric"] == metric
                           and r["regime"] == "deep"), None)
                if ys is None:
                    continue
                ax.plot(range(len(ys)), ys,
                        color=PALETTE.get(comp, "#444"),
                        alpha=0.7, label=comp)
            ax.set_title(f"{scen} — {metric_titles[metric]}", fontsize=8)
            if metric != "rel_l2":
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_yscale("log")
            if i == len(scenarios) - 1:
                ax.set_xlabel(r"$\lambda$-index")
            if i == 0 and j == 0:
                ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle("F4 — Cross-package agreement vs skein (dense regime)", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("headline_timing", "scaling_curves", "recovery_curves",
                 "ic_selection", "agreement", "realdata_boxplots",
                 "stability_fdr_power", "cv_parallel_speedup",
                 "screening_parallel"):
        s = sub.add_parser(name)
        s.add_argument("--inputs", nargs="+", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"headline_timing":     headline_timing,
     "scaling_curves":      scaling_curves,
     "recovery_curves":     recovery_curves,
     "ic_selection":        ic_selection,
     "agreement":           agreement,
     "realdata_boxplots":   realdata_boxplots,
     "stability_fdr_power": stability_fdr_power,
     "cv_parallel_speedup": cv_parallel_speedup,
     "screening_parallel":  screening_parallel}[a.cmd](a.inputs, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
