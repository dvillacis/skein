"""Plot per-λ agreement for the nonconvex correctness checks.

Reads `benches/correctness/results/<scenario>.json` and produces a
side-by-side {jaccard, sign, rel-L2} plot vs λ-index. Writes PNG to
`benches/correctness/results/<scenario>_agreement.png`.

Run as a script:

    python benches/correctness/plot_agreement.py mcp_ls scad_ls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def plot_one(scenario: str, focus: str | None = None) -> Path:
    payload = json.loads((RESULTS_DIR / f"{scenario}.json").read_text())
    n_lambdas = payload["n_lambdas"]
    pairs = payload["pairs"]
    if focus is not None:
        pairs = [p for p in pairs if focus in (p["a"], p["b"])]
    if not pairs:
        pairs = payload["pairs"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for p in pairs:
        label = f"{p['a']} vs {p['b']}"
        axes[0].plot(np.arange(n_lambdas), p["per_lambda"]["jaccard"], marker="o", ms=3, label=label)
        axes[1].plot(np.arange(n_lambdas), p["per_lambda"]["sign_agreement"], marker="o", ms=3, label=label)
        axes[2].plot(np.arange(n_lambdas), p["per_lambda"]["rel_l2"], marker="o", ms=3, label=label)

    axes[0].set_title("Active-set Jaccard")
    axes[1].set_title("Sign agreement")
    axes[2].set_title("Relative L2 distance")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("λ index (0 = λ_max, low λ on the right)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    gamma_str = f"γ={payload['gamma']}, " if "gamma" in payload else ""
    fig.suptitle(
        f"{scenario}: per-λ cross-package agreement "
        f"({gamma_str}n_lambdas={n_lambdas}, "
        f"λ_min/λ_max={payload['lambda_min_ratio']})"
    )
    fig.tight_layout()
    out = RESULTS_DIR / f"{scenario}_agreement.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="+", help="scenario name(s), e.g. mcp_ls scad_ls")
    parser.add_argument(
        "--focus",
        default=None,
        help="if set, only plot pairs that include this package (e.g. 'skein' or 'ncvreg')",
    )
    args = parser.parse_args()

    for scenario in args.scenarios:
        try:
            out = plot_one(scenario, focus=args.focus)
        except FileNotFoundError:
            print(f"skip {scenario}: no results JSON", file=sys.stderr)
            continue
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
