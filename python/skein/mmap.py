"""Memory-mapped design-matrix wrappers.

Two flavors backed by column-major (Fortran-order) raw files on disk:

- `MmapDesignF64`: 8 bytes per element, no precision loss vs. an
  in-memory `np.ndarray(dtype=np.float64)`.
- `MmapDesignF32`: 4 bytes per element — half the disk footprint and
  half the page-cache pressure. Solver arithmetic stays f64; each
  column is cast f32→f64 on read. Useful when the bulk of `(n × p)`
  is the bottleneck and the f32 truncation error (~1e-7 relative) is
  acceptable for the path; refining at the active set in f64 is a
  separate "true mixed precision" milestone.

Estimators that support memory-mapped input sniff `isinstance(x,
(MmapDesignF64, MmapDesignF32))` and route through the matching
`_mmap[_f32]` PyO3 entry points instead of copying the matrix into RAM.

Producing a compatible file from numpy. Note that `ndarray.tofile()`
always writes in C (row-major) order regardless of array layout, so
the column-major layout we need has to be written via `tobytes(
order='F')`:

    >>> import numpy as np
    >>> x = np.random.standard_normal((100_000, 1_000))
    >>> # f64:
    >>> buf = np.ascontiguousarray(x, dtype=np.float64).tobytes(order="F")
    >>> with open("x.bin", "wb") as f: f.write(buf)
    >>> # f32:
    >>> buf32 = np.ascontiguousarray(x, dtype=np.float32).tobytes(order="F")
    >>> with open("x32.bin", "wb") as f: f.write(buf32)

Then construct the wrapper and pass it to a supporting estimator:

    >>> from skein import MmapDesignF64, MCPPathRegressor
    >>> design = MmapDesignF64("x.bin", n_rows=100_000, n_cols=1_000)
    >>> model = MCPPathRegressor(gamma=3.0, n_lambdas=50).fit(design, y)

v1 estimator coverage: `MCPPathRegressor`, `LogisticMCPPathRegressor`.
Other estimators raise `TypeError` if handed an mmap design —
expanding coverage is mechanical and waits on user demand.
"""
from __future__ import annotations

import os
from pathlib import Path


def _validate_mmap_args(
    cls_name: str,
    path: str | os.PathLike,
    n_rows: int,
    n_cols: int,
    bytes_per_elem: int,
) -> str:
    path = os.fspath(Path(path).resolve())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{cls_name}: {path} does not exist")
    actual = os.path.getsize(path)
    expected = n_rows * n_cols * bytes_per_elem
    if actual != expected:
        raise ValueError(
            f"{cls_name}: file {path} is {actual} bytes; "
            f"expected {expected} bytes for shape ({n_rows}, {n_cols}) "
            f"with {bytes_per_elem}-byte elements"
        )
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(
            f"{cls_name}: n_rows and n_cols must be > 0 "
            f"(got {n_rows}, {n_cols})"
        )
    return path


class MmapDesignF64:
    """Reference to an on-disk column-major `f64` matrix.

    The constructor validates dimensions against file size; it does
    not open the mapping (the Rust side mmaps lazily inside each
    `_mmap` solve).
    """

    dtype: str = "f64"

    def __init__(
        self,
        path: str | os.PathLike,
        n_rows: int,
        n_cols: int,
    ) -> None:
        self.path = _validate_mmap_args("MmapDesignF64", path, n_rows, n_cols, 8)
        self.n_rows = n_rows
        self.n_cols = n_cols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    def __repr__(self) -> str:
        return (
            f"MmapDesignF64(path={self.path!r}, "
            f"n_rows={self.n_rows}, n_cols={self.n_cols})"
        )


class MmapDesignF32:
    """Reference to an on-disk column-major `f32` matrix.

    Half the bytes per element vs. `MmapDesignF64`, with f32→f64
    conversion done on each column read inside the solver. Equivalent
    to the f64 path up to f32 truncation error (~1e-7 relative).
    """

    dtype: str = "f32"

    def __init__(
        self,
        path: str | os.PathLike,
        n_rows: int,
        n_cols: int,
    ) -> None:
        self.path = _validate_mmap_args("MmapDesignF32", path, n_rows, n_cols, 4)
        self.n_rows = n_rows
        self.n_cols = n_cols

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    def __repr__(self) -> str:
        return (
            f"MmapDesignF32(path={self.path!r}, "
            f"n_rows={self.n_rows}, n_cols={self.n_cols})"
        )
