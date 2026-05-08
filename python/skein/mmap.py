"""Memory-mapped design-matrix wrapper.

Backed by a column-major (Fortran-order) raw `f64` file on disk.
Estimators that support memory-mapped input sniff `isinstance(x,
MmapDesignF64)` and route through the `_mmap` PyO3 entry points
instead of copying the matrix into RAM.

Producing a compatible file from numpy. Note that `ndarray.tofile()`
always writes in C (row-major) order regardless of array layout, so
the column-major layout we need has to be written via `tobytes(
order='F')`:

    >>> import numpy as np
    >>> x = np.random.standard_normal((100_000, 1_000))
    >>> buf = np.ascontiguousarray(x, dtype=np.float64).tobytes(order="F")
    >>> with open("x.bin", "wb") as f: f.write(buf)

Then construct the wrapper and pass it to a supporting estimator:

    >>> from skein import MmapDesignF64, MCPPathRegressor
    >>> design = MmapDesignF64("x.bin", n_rows=100_000, n_cols=1_000)
    >>> model = MCPPathRegressor(gamma=3.0, n_lambdas=50).fit(design, y)

v1 estimator coverage: `MCPPathRegressor`, `LogisticMCPPathRegressor`.
Other estimators raise `TypeError` if handed a `MmapDesignF64` —
expanding coverage is mechanical and waits on user demand.
"""
from __future__ import annotations

import os
from pathlib import Path


class MmapDesignF64:
    """Reference to an on-disk column-major `f64` matrix.

    The constructor validates dimensions against file size; it does
    not open the mapping (the Rust side mmaps lazily inside each
    `_mmap` solve).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        n_rows: int,
        n_cols: int,
    ) -> None:
        path = os.fspath(Path(path).resolve())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MmapDesignF64: {path} does not exist")
        actual = os.path.getsize(path)
        expected = n_rows * n_cols * 8
        if actual != expected:
            raise ValueError(
                f"MmapDesignF64: file {path} is {actual} bytes; "
                f"expected {expected} bytes for shape ({n_rows}, {n_cols}) f64"
            )
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError(
                f"MmapDesignF64: n_rows and n_cols must be > 0 "
                f"(got {n_rows}, {n_cols})"
            )
        self.path = path
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
