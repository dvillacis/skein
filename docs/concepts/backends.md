# Backends

`skein`'s `DesignMatrix` trait abstracts the storage of $X$ from the
solver. The trait has five methods; once you implement them, every
estimator works with your backend, and every wrapper composes with
it. v0.1 ships four storage backends and two structural wrappers.

## The trait

```rust
pub trait DesignMatrix: Sync + Send {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64>;       // X β
    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64>;         // Xᵀ r
    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64;       // ⟨X[:,j], v⟩
    fn col_sq_norm(&self, j: usize) -> f64;                       // ‖X[:,j]‖²
    fn columns(&self, cols: &[usize]) -> Array2<f64>;             // X[:, cols] block
}
```

`col_dot` and `col_sq_norm` are the hot path for coordinate descent;
`columns(cols)` is what group block-CD calls to materialize a group's
columns. `matvec` and `rmatvec` are used by gap-safe screening and
by the GLM working-response calculation.

The CD solver only ever touches $X$ through these five methods.
That's the leverage — every concrete backend gets every algorithm
for free.

## Storage backends

### `DenseMatrix` (in-memory, row-major f64)

The reference implementation. Backed by `ndarray::Array2<f64>`.
Used automatically when you pass a numpy 2D array to any estimator.

```python
model = skein.MCPPathRegressor(...).fit(X, y)   # X is np.ndarray
```

`col_sq_norms` is precomputed at construction. Hot path is direct
ndarray indexing — fast, no allocation per call.

### `SparseCSC` (in-memory, scipy.sparse.csc_matrix)

Compressed sparse column. The hot path (`col_dot`, `columns`) walks
each column's `data` / `indices` arrays, skipping the implicit
zero entries. `col_sq_norms` is precomputed.

```python
import scipy.sparse as sp
X_sparse = sp.csc_matrix(X)
model = skein.MCPPathRegressor(...).fit(X_sparse, y)
```

Used automatically when you pass a `scipy.sparse.csc_matrix`. CSR
inputs are converted to CSC at the boundary (one copy).

`matvec` skips entries with `β_j = 0`, which makes warm-started
paths very fast on sparse problems.

### `MmapMatrix` (out-of-RAM, column-major f64 on disk)

Memory-mapped raw `f64` files. The OS page cache transparently
materializes the columns as the solver reads them; nothing is
loaded into RAM until accessed.

```python
# Write a column-major f64 file (note: tofile() always writes in
# C order regardless of array layout, so use tobytes(order='F')).
X_disk = ...  # large numpy array
buf = np.ascontiguousarray(X_disk, dtype=np.float64).tobytes(order="F")
with open("X.bin", "wb") as f:
    f.write(buf)

design = skein.MmapDesignF64("X.bin", n_rows=n, n_cols=p)
model = skein.MCPPathRegressor(...).fit(design, y)
```

Why column-major: the CD hot path `col_dot(j, v)` reads each column
as a contiguous slice of the file. With row-major storage, each
`col_dot` would scatter $n$ non-contiguous bytes — pathological for
the page cache.

`col_sq_norms` is computed once at `open()` time (one full pass
through the file, paid by the page cache); from then on the hot
path matches `DenseMatrix` performance.

### `MmapMatrixF32` (out-of-RAM, column-major f32 on disk)

Same as `MmapMatrix` but the file is f32. Half the disk footprint
and half the page-cache pressure for the same `(n, p)`. The solver
stays f64; each column is cast f32 → f64 on read. Equivalent to
the f64 path up to f32 truncation error (~1e-7 relative), in
exchange for halving the I/O.

```python
buf32 = np.ascontiguousarray(X_disk, dtype=np.float32).tobytes(order="F")
with open("X32.bin", "wb") as f:
    f.write(buf32)

design = skein.MmapDesignF32("X32.bin", n_rows=n, n_cols=p)
```

This is "f32-on-disk" — not "true mixed precision" (which would also
do arithmetic in f32). True mixed precision is a separate roadmap
bullet that requires parameterizing the solver core over a
floating-point type parameter.

### `Chunked<C>` (out-of-RAM, list of row-block files)

For when one mmap'd file is unwieldy: multiple shards from an
upstream pipeline, files larger than your filesystem's per-file
limit, or natural per-batch chunking. Each chunk is a separate
column-major file with the same `n_cols`; the wrapper concatenates
them logically.

```python
chunks = [
    ("chunk_0.bin", 10_000_000),
    ("chunk_1.bin", 10_000_000),
    ("chunk_2.bin",  7_345_678),
]
design = skein.ChunkedDesignF64(chunks, n_cols=50_000)
# total n = 27_345_678
```

Generic over the chunk backend in Rust: `Chunked<MmapMatrix>` is the
f64 case (above), `Chunked<MmapMatrixF32>` is f32 (via
`ChunkedDesignF32` in Python).

Routes the trait methods chunk-by-chunk:

- `col_dot(j, v)` slices `v` into per-chunk segments and sums.
- `matvec(β)` calls each chunk's `matvec(β)` and concatenates.
- `rmatvec(r)` slices `r` and sums per-feature.
- `col_sq_norm(j)` is precomputed at construction by summing across
  chunks.

v0.1 is serial. Per-chunk Rayon parallelism is a one-line follow-up
once benchmarks justify it.

## Wrapper layers

Two zero-cost wrappers that compose with any storage backend.

### `Augmented<D>` — virtual intercept column

Adds a single all-ones column at index `n_features` (the augmented
intercept slot) without touching the underlying storage:

- `n_features = base.n_features + 1`
- `col_dot(p, v) = Σ v_i`
- `col_sq_norm(p) = n`
- `matvec(β) = base.matvec(β[:p]) + β[p]`

This is what `fit_intercept=True` uses on mmap and chunked backends:
appending a 1s column to a 100GB file would be insane, so we just
declare one virtually.

For dense and sparse backends, `fit_intercept=True` physically
appends a 1s column instead. Both routes converge to the same β.

### `Standardized<D>` — lazy column scaling

Wraps any backend with per-column scales $s_j$ to expose
$\tilde X = X \cdot \mathrm{diag}(1/s)$:

- `col_dot(j, v) = base.col_dot(j, v) / s_j`
- `col_sq_norm(j) = base.col_sq_norm(j) / s_j²`
- `matvec(β) = base.matvec(β / s)`

No materialization, no centering — and centering would densify a
sparse / mmap'd $X$. The intercept column (when wrapped via
`Augmented` first) gets `s_p = 1.0` so it's never scaled.

This is what `standardize=True` uses on every non-dense-LS backend.
For dense LS, the original glmnet centering+scaling is still in
play (centering $X$ is fine when $X$ already lives in RAM as
`f64`).

## How the layers compose

A typical mmap + intercept + standardize fit on logistic MCP:

```text
DesignMatrix tower at solve time:

    Standardized<Augmented<MmapMatrix>>
        x_scale = [s_1, s_2, ..., s_p, 1.0]   ← intercept unscaled
            Augmented adds a virtual 1s column
                MmapMatrix opens X.bin (col-major f64)
```

The solver doesn't see the layers — it sees a `&dyn DesignMatrix`
with `n_features = p + 1` and trait methods that route through
the entire stack on each call. After convergence, the user-facing
β is recovered as $\beta_{\text{user}} = \tilde\beta / s$ for the
non-intercept features, and the intercept is the last entry of
$\tilde\beta$ as-is.

The same composition works for sparse, chunked, etc. — the wrappers
don't know or care which backend is at the bottom.

## Choosing a backend

| `n × p` | RAM available | Choose                        |
|---------|---------------|-------------------------------|
| `n × p × 8 ≪ RAM`, dense | ≫ data | `np.ndarray` (DenseMatrix)    |
| Mostly zeros (>50%)        | ≫ data | `scipy.sparse.csc_matrix` (SparseCSC) |
| `n × p × 8 > RAM`, single file | ≪ data | `MmapDesignF64`         |
| `n × p × 4 < RAM ≤ n × p × 8` | between | `MmapDesignF32`         |
| Multi-file shards          | any   | `ChunkedDesignF64` / `F32`    |

The estimators dispatch on the input type; you don't need to pass a
flag. The same `MCPPathRegressor` works on all five.

## Adding a new backend

To add a new backend (HDF5 reader, Parquet column store, GPU
buffer, etc.):

1. Implement the five `DesignMatrix` trait methods in Rust.
2. Add a thin PyO3 wrapper if you want Python access (mirror what
   `MmapMatrix` does).
3. Add a Python helper class similar to `MmapDesignF64` that
   estimators recognize via `isinstance` dispatch.
4. The estimators that should support your backend get an `elif
   _is_yourbackend(x):` branch in their `fit` method.

The "extending skein" page (coming in commit 2) walks through this
end-to-end with a worked example.

## See also

- [Penalties](penalties.md) — what gets composed with the design.
- [Datafits](datafits.md) — what reads from the design via `matvec` /
  `rmatvec`.
