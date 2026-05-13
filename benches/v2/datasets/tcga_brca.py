"""survival::pbc — Cox proxy for TCGA-BRCA (CSV transport)."""
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


DATASET_ID = "pbc"


_R_SCRIPT = r"""
suppressPackageStartupMessages({ library(survival) })
args <- commandArgs(trailingOnly = TRUE); out <- args[[1]]
data(pbc)
df <- pbc[complete.cases(pbc), ]
event <- as.integer(df$status > 0)
time <- as.numeric(df$time)
features <- df[, !(names(df) %in% c("id", "time", "status"))]
features$sex <- as.integer(features$sex == "f")
features$ascites <- as.integer(features$ascites > 0)
features$hepato <- as.integer(features$hepato > 0)
features$spiders <- as.integer(features$spiders > 0)
features$edema <- as.numeric(features$edema)
features$stage <- as.integer(features$stage)
X <- as.matrix(features)
y <- cbind(time, event)
write.table(X, file.path(out, "X.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
write.table(y, file.path(out, "y.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
"""


def _fetch_from_r(d: Path) -> bool:
    if not required_pkgs_present(["survival"]):
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
        "source": "survival::pbc (Cox proxy)",
        "y_label": "Surv(time, event)",
        "files": {
            "X.csv": sha256(d / "X.csv"),
            "y.csv": sha256(d / "y.csv"),
        },
    })
    return True


def _load_cached(d: Path) -> Problem:
    X, y = load_csv_pair(d)
    time = y[:, 0]
    event = y[:, 1].astype(np.int64)
    return Problem(
        x=X, y=time, beta_true=np.zeros(X.shape[1]),
        family="cox",
        meta={"dataset": "pbc", "n": X.shape[0], "p": X.shape[1],
              "event": event,
              "censoring_rate": float(1.0 - event.mean()),
              "source": "survival::pbc"},
    )


def synthetic_pbc(seed: int = 0) -> Problem:
    from benches.v2.simulators import cox_truth
    return cox_truth.make(
        n=276, p=17, seed=seed, signal_scale=0.6,
        sparsity_k=1.0, target_censoring=0.6,
    )


def load(use_synthetic: bool = False) -> Problem:
    if use_synthetic:
        return synthetic_pbc()
    d = dataset_dir(DATASET_ID)
    if verify_or_drop(d):
        return _load_cached(d)
    if not _fetch_from_r(d):
        raise RuntimeError(
            "pbc not cached and R::survival unavailable. Install with:\n"
            "  R -e \"install.packages('survival',repos='https://cloud.r-project.org/')\"\n"
            "or pass use_synthetic=True."
        )
    return _load_cached(d)
