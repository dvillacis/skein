"""Arrow-IPC (feather) transport for the R subprocess runners.

Replaces the legacy JSON transport in `benches/runners/r_runner.R`,
which OOM'd at n=100k, p=10k (the JSON string of X alone was ~8 GB).

Design (deliberately boring):
    <tmpdir>/
        config.feather    # one row of scalars (package, penalty, family, tol, ...)
        X.feather         # n × p, one column per feature, f64
        y.feather         # n × 1
        lambda.feather    # n_lambdas × 1
        groups.feather    # optional, n_features × 1, int64
        --- R writes ---
        result.feather    # coef_path: n_lambdas × p
        result_meta.feather  # fit_time_s, version, n_iter, ...

Each file is a single Arrow table; reading is a single
`arrow::read_feather()` call on the R side, `pyarrow.feather.read_table`
on the Python side. No multi-stream stitching, no schema gymnastics.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

R_RUNNER = Path(__file__).resolve().parent / "r_runner.R"


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.feather as feather
        return pa, feather
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is required for the v2 R transport — "
            "install via `pip install -e '.[bench]'`"
        ) from e


def _scalars_table(d: dict[str, Any]):
    pa, _ = _import_pyarrow()
    cleaned = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            cleaned[k] = pa.array([bool(v)], type=pa.bool_())
        elif isinstance(v, (int, np.integer)):
            cleaned[k] = pa.array([int(v)], type=pa.int64())
        elif isinstance(v, (float, np.floating)):
            cleaned[k] = pa.array([float(v)], type=pa.float64())
        elif isinstance(v, str):
            cleaned[k] = pa.array([v], type=pa.string())
        else:
            raise TypeError(f"unsupported scalar type for {k!r}: {type(v).__name__}")
    return pa.table(cleaned)


def write_request(
    workdir: Path,
    *,
    package: str,
    penalty: str,
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    tol: float,
    groups: np.ndarray | None = None,
    gamma: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    pa, feather = _import_pyarrow()

    workdir.mkdir(parents=True, exist_ok=True)

    cfg = _scalars_table({
        "package": package, "penalty": penalty, "family": family,
        "tol": float(tol),
        "gamma": None if gamma is None else float(gamma),
        "n": int(x.shape[0]), "p": int(x.shape[1]),
        "has_groups": groups is not None,
        **(extra or {}),
    })
    feather.write_feather(cfg, workdir / "config.feather", compression="uncompressed")

    x = np.ascontiguousarray(x, dtype=np.float64)
    x_tbl = pa.table({f"x{j}": pa.array(x[:, j]) for j in range(x.shape[1])})
    feather.write_feather(x_tbl, workdir / "X.feather", compression="uncompressed")

    y_tbl = pa.table({"y": pa.array(np.ascontiguousarray(y, dtype=np.float64))})
    feather.write_feather(y_tbl, workdir / "y.feather", compression="uncompressed")

    lam_tbl = pa.table({"lambda": pa.array(
        np.ascontiguousarray(lambdas, dtype=np.float64))})
    feather.write_feather(lam_tbl, workdir / "lambda.feather", compression="uncompressed")

    if groups is not None:
        grp_tbl = pa.table({"group": pa.array(
            np.ascontiguousarray(groups, dtype=np.int64))})
        feather.write_feather(grp_tbl, workdir / "groups.feather",
                              compression="uncompressed")


def read_response(workdir: Path) -> dict[str, Any]:
    _, feather = _import_pyarrow()
    out: dict[str, Any] = {}

    coefs_path = workdir / "result.feather"
    if coefs_path.exists():
        tbl = feather.read_table(coefs_path)
        # result.feather has one column per feature: x0..x{p-1}, n_lambdas rows.
        cols = tbl.column_names
        # Preserve column order (x0..x{p-1}) explicitly.
        idx = sorted(range(len(cols)),
                     key=lambda i: int(cols[i].lstrip("x")) if cols[i].startswith("x")
                     else len(cols))
        out["coef_path"] = np.column_stack(
            [tbl.column(i).to_numpy(zero_copy_only=False) for i in idx]
        )

    meta_path = workdir / "result_meta.feather"
    if meta_path.exists():
        tbl = feather.read_table(meta_path)
        for col in tbl.column_names:
            val = tbl.column(col)[0].as_py()
            out[col] = val
    return out


def run_r(
    *,
    package: str,
    penalty: str,
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    tol: float,
    groups: np.ndarray | None = None,
    gamma: float | None = None,
    extra: dict[str, Any] | None = None,
    timeout_s: int = 1800,
    keep_workdir: Path | None = None,
) -> dict[str, Any]:
    """Round-trip one fit through the R subprocess via feather IPC."""
    if keep_workdir is not None:
        workdir = keep_workdir
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workdir = Path(tempfile.mkdtemp(prefix="skein-bench-r-"))
        cleanup = True

    try:
        write_request(workdir, package=package, penalty=penalty, family=family,
                      x=x, y=y, lambdas=lambdas, tol=tol, groups=groups,
                      gamma=gamma, extra=extra)
        try:
            subprocess.run(
                ["Rscript", str(R_RUNNER), str(workdir)],
                check=True, capture_output=True, timeout=timeout_s,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace")
            stdout = (e.stdout or b"").decode(errors="replace")
            raise RuntimeError(
                f"r_runner.R failed (package={package}, penalty={penalty}):\n"
                f"--- stderr ---\n{stderr[:4000]}\n--- stdout ---\n{stdout[:1000]}"
            ) from e
        return read_response(workdir)
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
