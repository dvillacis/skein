"""Leukemia / Golub dataset (Golub et al., *Science* 1999).

n = 72 patients (47 ALL, 25 AML)
p = 7129 gene-expression measurements (Affymetrix Hu6800)
y ∈ {0, 1}: AML = 1, ALL = 0

Loader strategy:
  1. Cached feather (preferred).
  2. R `Biobase::golubEsets` if installed.
  3. Synthetic fallback with matching (n=72, p=7129) sparse-logistic shape.

The Golub data is freely redistributable but lives behind Bioconductor —
fetching it programmatically inside an arbitrary Python venv is fragile,
so the synthetic fallback is the realistic default outside of users who
have already loaded it locally.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from benches.problems import Problem
from benches.v2.datasets._cache import (
    dataset_dir, sha256, verify_or_drop, write_manifest,
)


DATASET_ID = "leukemia"


def _r_dump_script() -> str:
    return r"""
suppressPackageStartupMessages({
  library(arrow)
  library(Biobase)
  library(golubEsets)
})
args <- commandArgs(trailingOnly = TRUE)
out <- args[[1]]
data("Golub_Merge", package = "golubEsets")
X <- t(exprs(Golub_Merge))
y_factor <- pData(Golub_Merge)$ALL.AML
y <- as.numeric(y_factor == "AML")
colnames(X) <- sprintf("x%d", seq.int(0L, ncol(X) - 1L))
write_feather(as.data.frame(X), file.path(out, "X.feather"),
              compression = "uncompressed")
write_feather(data.frame(y = y), file.path(out, "y.feather"),
              compression = "uncompressed")
"""


def _fetch_from_r(d: Path) -> bool:
    if not shutil.which("Rscript"):
        return False
    probe = subprocess.run(
        ["Rscript", "-e",
         "q(status = as.integer(!all(c('arrow','Biobase','golubEsets') %in% installed.packages()[,1])))"],
        check=False, capture_output=True, timeout=15,
    )
    if probe.returncode != 0:
        return False
    script = d / "_dump.R"
    script.write_text(_r_dump_script())
    try:
        subprocess.run(
            ["Rscript", str(script), str(d)],
            check=True, capture_output=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        d.joinpath("_fetch_error.log").write_text(
            (e.stderr or b"").decode(errors="replace"))
        return False
    finally:
        script.unlink(missing_ok=True)
    write_manifest(d, {
        "source": "golubEsets::Golub_Merge (Golub 1999)",
        "n": 72, "p": 7129, "y_label": "AML (1) vs ALL (0)",
        "files": {
            "X.feather": sha256(d / "X.feather"),
            "y.feather": sha256(d / "y.feather"),
        },
    })
    return True


def _load_cached(d: Path) -> Problem:
    import pyarrow.feather as feather
    x_tbl = feather.read_table(d / "X.feather")
    y_tbl = feather.read_table(d / "y.feather")
    cols = sorted(x_tbl.column_names,
                  key=lambda c: int(c.lstrip("x")) if c.startswith("x") else 0)
    X = np.column_stack(
        [x_tbl.column(c).to_numpy(zero_copy_only=False) for c in cols]
    )
    y = y_tbl.column("y").to_numpy(zero_copy_only=False).astype(float)
    return Problem(
        x=X, y=y, beta_true=np.zeros(X.shape[1]),
        family="logistic",
        meta={"dataset": "leukemia", "n": X.shape[0], "p": X.shape[1],
              "source": "golubEsets::Golub_Merge",
              "class_balance": float(y.mean())},
    )


def synthetic_leukemia(seed: int = 0) -> Problem:
    """Offline-safe stand-in for benchmarking pipeline shakedowns."""
    from benches.v2.simulators import logistic_truth
    return logistic_truth.make(
        n=72, p=7129, seed=seed, signal_scale=1.5, sparsity_k=2.0,
        corr_kind="toeplitz", corr_rho=0.4,
    )


def load(use_synthetic: bool = False) -> Problem:
    if use_synthetic:
        return synthetic_leukemia()
    d = dataset_dir(DATASET_ID)
    if verify_or_drop(d):
        return _load_cached(d)
    if not _fetch_from_r(d):
        raise RuntimeError(
            "leukemia not cached and R::golubEsets unavailable. Install with:\n"
            "  R -e \"BiocManager::install(c('Biobase','golubEsets'))\"\n"
            "or pass use_synthetic=True for a (n=72, p=7129) stand-in."
        )
    return _load_cached(d)
