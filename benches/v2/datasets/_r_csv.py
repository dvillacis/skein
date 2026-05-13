"""Helper: run an R script that writes (X.csv, y.csv) and load them.

Used for small real datasets where the feather transport's `arrow` R
package isn't worth the install. CSV is fine here — sizes are kBs to
single-digit MBs.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


def rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def required_pkgs_present(pkgs: list[str], timeout: int = 15) -> bool:
    if not rscript_available():
        return False
    script = ("q(status = as.integer(!all(c("
              + ",".join(f"'{p}'" for p in pkgs)
              + ") %in% installed.packages()[,1])))")
    r = subprocess.run(["Rscript", "-e", script], check=False,
                       capture_output=True, timeout=timeout)
    return r.returncode == 0


def run_r_dump(script_path: Path, workdir: Path, timeout: int = 120) -> None:
    """Run an Rscript that writes X.csv (no header, comma-separated) and
    y.csv (single column, may include extra cols for Cox)."""
    subprocess.run(
        ["Rscript", str(script_path), str(workdir)],
        check=True, capture_output=True, timeout=timeout,
    )


def load_csv_pair(workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    X = np.loadtxt(workdir / "X.csv", delimiter=",")
    y_raw = np.loadtxt(workdir / "y.csv", delimiter=",")
    return X, y_raw
