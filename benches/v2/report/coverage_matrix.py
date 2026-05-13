"""F1 — coverage heatmap: rows = (datafit × penalty), columns = packages,
cell color = direct / surrogate / none.

Phase A renders the matrix from `runners/registry.py` (the source of
truth) plus the skein column (always direct). Phase D will replace the
hand-curated registry with introspection over the public estimator
surface.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from benches.v2.runners.registry import LADDER

LEVEL_COLORS = {
    "direct":    "#196f3d",
    "surrogate": "#b7950b",
    "none":      "#e5e7e9",
}
LEVEL_INT = {"direct": 2, "surrogate": 1, "none": 0}


def build(out: Path) -> None:
    cells = sorted(LADDER.keys())                # (datafit, penalty)
    packages = sorted({pkg for v in LADDER.values() for pkg in v.keys()})
    packages = ["skein"] + packages              # skein column first

    matrix = np.zeros((len(cells), len(packages)), dtype=int)
    for i, (df_, pen) in enumerate(cells):
        for j, pkg in enumerate(packages):
            if pkg == "skein":
                matrix[i, j] = LEVEL_INT["direct"]
            else:
                level = LADDER[(df_, pen)].get(pkg, "none")
                matrix[i, j] = LEVEL_INT[level]

    cmap = mpl.colors.ListedColormap(
        [LEVEL_COLORS["none"], LEVEL_COLORS["surrogate"], LEVEL_COLORS["direct"]])
    fig, ax = plt.subplots(figsize=(0.6 * len(packages) + 1.5,
                                    0.32 * len(cells) + 1.0))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(packages)))
    ax.set_xticklabels(packages, rotation=45, ha="right")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"{df_}/{pen}" for df_, pen in cells], fontsize=8)
    ax.set_title("F1 — Estimator support across packages", fontsize=10)

    handles = [
        mpl.patches.Patch(color=LEVEL_COLORS["direct"],    label="direct"),
        mpl.patches.Patch(color=LEVEL_COLORS["surrogate"], label="surrogate"),
        mpl.patches.Patch(color=LEVEL_COLORS["none"],      label="not supported"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
