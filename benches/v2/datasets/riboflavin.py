"""Riboflavin dataset (Bühlmann et al., *Annals of Stats* 2014; ships
in the `hdi` R package).

n = 71 industrial Bacillus subtilis production samples
p = 4088 gene-expression log-intensities
y = log-transformed riboflavin production rate

This is the canonical high-dimensional regression dataset for sparse-
modeling papers (used in Bühlmann–Van de Geer for debiased lasso, in
glmnet documentation, etc.). The skein paper uses it to demonstrate
LS Lasso / MCP / SCAD + adaptive lasso on real n ≪ p data.

Loader strategy:
  1. Check `~/.cache/skein-bench/riboflavin/` for cached `X.feather` and
     `y.feather`. If valid (sha256 match), return them.
  2. If missing, try `Rscript -e 'data(riboflavin, package="hdi"); ...'`
     to pull from a locally-installed R `hdi` package. Cache and return.
  3. If R + hdi aren't available, raise with a clear install hint.

The synthetic fallback (use `synthetic_riboflavin()` directly) generates
a problem with matching (n=71, p=4088, sparse β*) so the pipeline can be
exercised offline.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from benches.problems import Problem
from benches.v2.datasets._cache import (
    dataset_dir, read_manifest, sha256, verify_or_drop, write_manifest,
)


DATASET_ID = "riboflavin"


def _r_dump_script() -> str:
    return r"""
suppressPackageStartupMessages({
  library(arrow)
  library(hdi)
})
args <- commandArgs(trailingOnly = TRUE)
out <- args[[1]]
data("riboflavin", package = "hdi")
df <- riboflavin
y <- as.numeric(df[, "y"])
X <- as.matrix(df[, "x"])
colnames(X) <- sprintf("x%d", seq.int(0L, ncol(X) - 1L))
write_feather(as.data.frame(X), file.path(out, "X.feather"),
              compression = "uncompressed")
write_feather(data.frame(y = y), file.path(out, "y.feather"),
              compression = "uncompressed")
"""


def _fetch_from_r(d: Path) -> bool:
    """Pull riboflavin from a locally-installed `hdi` R package via Rscript.
    Returns True on success, False if Rscript / hdi / arrow aren't available."""
    if not shutil.which("Rscript"):
        return False
    # Probe deps.
    probe = subprocess.run(
        ["Rscript", "-e",
         "q(status = as.integer(!all(c('arrow','hdi') %in% installed.packages()[,1])))"],
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
        "source": "hdi::riboflavin (Bühlmann 2014)",
        "n": 71, "p": 4088, "y_label": "log-riboflavin production",
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
    p = x_tbl.num_columns
    cols = sorted(x_tbl.column_names,
                  key=lambda c: int(c.lstrip("x")) if c.startswith("x") else 0)
    X = np.column_stack(
        [x_tbl.column(c).to_numpy(zero_copy_only=False) for c in cols]
    )
    y = y_tbl.column("y").to_numpy(zero_copy_only=False)
    return Problem(
        x=X, y=y, beta_true=np.zeros(X.shape[1]),  # truth unknown for real data
        family="gaussian",
        meta={"dataset": "riboflavin", "n": X.shape[0], "p": X.shape[1],
              "source": "hdi::riboflavin"},
    )


def synthetic_riboflavin(seed: int = 0) -> Problem:
    """Offline-safe stand-in: n=71, p=4088, sparse β*, correlated gene-
    expression-like design. Use only when the real loader is unavailable."""
    from benches.v2.simulators import linear_truth
    return linear_truth.make(
        n=71, p=4088, seed=seed, snr=3.0, sparsity_k=2.0,
        corr_kind="toeplitz", corr_rho=0.4,
    )


def load(use_synthetic: bool = False) -> Problem:
    if use_synthetic:
        return synthetic_riboflavin()
    d = dataset_dir(DATASET_ID)
    if verify_or_drop(d):
        return _load_cached(d)
    if not _fetch_from_r(d):
        raise RuntimeError(
            "riboflavin not cached and R::hdi unavailable. Install with:\n"
            "  R -e \"install.packages(c('arrow','hdi'),repos='https://cloud.r-project.org/')\"\n"
            "or pass use_synthetic=True to get a (n=71, p=4088) stand-in."
        )
    return _load_cached(d)
