"""Bardet/grpreg::Birthwt — group-LS benchmark dataset.

Real load uses `Rscript` + CSV transport (no `arrow` R dep required).
Falls back to synthetic when grpreg is absent or by request.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benches.problems import Problem
from benches.v2.datasets._cache import (
    dataset_dir, sha256, verify_or_drop, write_manifest,
)
from benches.v2.datasets._r_csv import (
    load_csv_pair, required_pkgs_present, run_r_dump,
)


DATASET_ID = "bardet"
GROUP_SIZE = 4


_R_SCRIPT = r"""
suppressPackageStartupMessages({ library(grpreg) })
args <- commandArgs(trailingOnly = TRUE); out <- args[[1]]
data("Birthwt", package = "grpreg")
X <- Birthwt$X
y <- as.numeric(Birthwt$bwt)
write.table(X, file.path(out, "X.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.table(y, file.path(out, "y.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
"""


def _fetch_from_r(d: Path) -> bool:
    if not required_pkgs_present(["grpreg"]):
        return False
    script = d / "_dump.R"
    script.write_text(_R_SCRIPT)
    try:
        run_r_dump(script, d)
    except Exception as e:
        d.joinpath("_fetch_error.log").write_text(repr(e))
        return False
    finally:
        script.unlink(missing_ok=True)
    write_manifest(d, {
        "source": "grpreg::Birthwt (group-LS proxy)",
        "y_label": "birth weight",
        "files": {
            "X.csv": sha256(d / "X.csv"),
            "y.csv": sha256(d / "y.csv"),
        },
    })
    return True


def _load_cached(d: Path) -> Problem:
    X, y = load_csv_pair(d)
    p = X.shape[1]
    groups = np.repeat(
        np.arange((p + GROUP_SIZE - 1) // GROUP_SIZE, dtype=np.int64),
        GROUP_SIZE,
    )[:p]
    return Problem(
        x=X, y=y, beta_true=np.zeros(X.shape[1]),
        groups=groups, family="gaussian",
        meta={"dataset": "bardet", "n": X.shape[0], "p": X.shape[1],
              "group_size": GROUP_SIZE, "n_groups": int(groups.max()) + 1,
              "source": "grpreg::Birthwt"},
    )


def synthetic_bardet(seed: int = 0) -> Problem:
    from benches.v2.simulators import group_truth
    return group_truth.make(
        n=120, p=200, seed=seed, group_size=GROUP_SIZE,
        k_active_groups=8, rho_within=0.7, rho_between=0.1,
    )


def load(use_synthetic: bool = False) -> Problem:
    if use_synthetic:
        return synthetic_bardet()
    d = dataset_dir(DATASET_ID)
    if verify_or_drop(d):
        return _load_cached(d)
    if not _fetch_from_r(d):
        raise RuntimeError(
            "bardet not cached and R::grpreg unavailable. Install with:\n"
            "  R -e \"install.packages('grpreg',repos='https://cloud.r-project.org/')\"\n"
            "or pass use_synthetic=True."
        )
    return _load_cached(d)
