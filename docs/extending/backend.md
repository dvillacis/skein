# Extending: custom design-matrix backends

The `DesignMatrix` trait in `skein-core::design` is the cleanest
extension point in the codebase — five methods, no algorithmic
content. Any concrete implementation works with every estimator
immediately.

This page walks through writing a new backend: an HDF5 column
reader as the worked example. The same pattern applies to any
read-once / read-many storage (Parquet, Zarr, custom binary
formats, GPU buffers, etc.).

## The trait

```rust
pub trait DesignMatrix: Sync + Send {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;

    /// X β.
    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64>;

    /// Xᵀ r.
    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64>;

    /// ⟨X[:,j], v⟩. Hot path for coordinate descent.
    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64;

    /// ‖X[:,j]‖². Cached by callers when possible — precompute once
    /// at construction.
    fn col_sq_norm(&self, j: usize) -> f64;

    /// Block of columns indexed by `cols`, returned as an owned
    /// (n, |cols|) array. Used by group block-CD.
    fn columns(&self, cols: &[usize]) -> Array2<f64>;
}
```

A few invariants to honor:

- **`Sync + Send`**. The solver runs `col_dot` from multiple Rayon
  threads concurrently. If your backend has interior mutability
  (e.g. a cache), wrap it in `Mutex` or use `crossbeam`'s atomics.
- **`col_sq_norm` should be O(1)**. The hot path calls it once per
  coordinate update; if it's $O(n)$ each call, you've added a
  factor of $n$ to every fit. Precompute.
- **Column-major access is preferred**. `col_dot(j, v)` is by far
  the most-called method. If your storage is row-major, every call
  scatters $n$ accesses. The mmap and CSC backends are both
  column-major for this reason.

## Worked example: HDF5 column reader

Suppose you have a 100M × 10K dataset stored in an HDF5 file with
each column as a separate dataset (`/columns/0`, `/columns/1`, ...).
You want to fit `MCPPathRegressor` on it without loading any
column twice.

### Rust skeleton

```rust
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Mutex;
use skein_core::design::DesignMatrix;
use hdf5::{File, Dataset};

pub struct Hdf5Matrix {
    /// HDF5 file handle. The hdf5 crate's File is Sync+Send.
    file: File,
    /// Per-column dataset paths (e.g. ["/columns/0", "/columns/1", ...]).
    column_paths: Vec<String>,
    n_samples: usize,
    n_features: usize,
    /// Precomputed at open time (one full pass through the file).
    col_sq_norms: Array1<f64>,
    /// Optional bounded LRU cache so repeatedly visited columns
    /// don't pay the HDF5 read cost. Mutex for thread safety.
    cache: Mutex<lru::LruCache<usize, Array1<f64>>>,
}

impl Hdf5Matrix {
    pub fn open(
        path: impl AsRef<std::path::Path>,
        column_paths: Vec<String>,
        n_samples: usize,
        cache_size: usize,
    ) -> hdf5::Result<Self> {
        let file = File::open(path)?;
        let n_features = column_paths.len();

        // One full pass over the file at open time to precompute norms.
        let mut col_sq_norms = Array1::<f64>::zeros(n_features);
        for (j, path) in column_paths.iter().enumerate() {
            let ds: Dataset = file.dataset(path)?;
            let col: Array1<f64> = ds.read_1d()?;
            assert_eq!(col.len(), n_samples);
            col_sq_norms[j] = col.iter().map(|&v| v * v).sum();
        }

        Ok(Self {
            file,
            column_paths,
            n_samples,
            n_features,
            col_sq_norms,
            cache: Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_size).unwrap(),
            )),
        })
    }

    fn read_column(&self, j: usize) -> Array1<f64> {
        // Try the cache first.
        if let Some(col) = self.cache.lock().unwrap().get(&j) {
            return col.clone();
        }
        // Cache miss: read from HDF5.
        let col: Array1<f64> = self.file
            .dataset(&self.column_paths[j])
            .expect("column dataset")
            .read_1d()
            .expect("column read");
        self.cache.lock().unwrap().put(j, col.clone());
        col
    }
}

impl DesignMatrix for Hdf5Matrix {
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.n_samples);
        for j in 0..self.n_features {
            let bj = beta[j];
            if bj == 0.0 {
                continue;   // skip zero β — warm-start friendly
            }
            let col = self.read_column(j);
            for (oi, &cv) in out.iter_mut().zip(col.iter()) {
                *oi += bj * cv;
            }
        }
        out
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.n_features);
        for j in 0..self.n_features {
            let col = self.read_column(j);
            out[j] = col.iter().zip(r.iter()).map(|(&a, &b)| a * b).sum();
        }
        out
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        let col = self.read_column(j);
        col.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum()
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        self.col_sq_norms[j]
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n_samples, cols.len()));
        for (k, &j) in cols.iter().enumerate() {
            let col = self.read_column(j);
            for (i, &v) in col.iter().enumerate() {
                out[[i, k]] = v;
            }
        }
        out
    }
}
```

### Tests

Two equivalence tests against an in-memory `DenseMatrix` reference:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use skein_core::design::DenseMatrix;

    #[test]
    fn hdf5_matvec_matches_dense() {
        let (file, dense) = build_test_problem();   // helper below
        let hdf5 = Hdf5Matrix::open(file.path(), col_paths(), n, 64).unwrap();
        let beta = ndarray::array![0.5, -1.0, 2.0];
        let r_dense = dense.matvec(beta.view());
        let r_hdf5 = hdf5.matvec(beta.view());
        for i in 0..r_dense.len() {
            assert_abs_diff_eq!(r_hdf5[i], r_dense[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn hdf5_solver_path_matches_dense() {
        // Run solve_path on Hdf5Matrix and on the reference DenseMatrix
        // for the same data; assert β matches at every λ within 1e-7.
        // ... (see crates/skein-core/src/design/mmap.rs for the pattern)
    }
}
```

The solver-equivalence test is the load-bearing one — if `Hdf5Matrix`
produces the same β as `DenseMatrix` at every λ, the trait
implementation is correct.

### PyO3 wrapper

If you want Python access:

```rust
#[pyfunction]
fn solve_mcp_ls_path_hdf5<'py>(
    py: Python<'py>,
    file_path: &str,
    column_paths: Vec<String>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    /* ... usual MCP args ... */
) -> PyResult<PathOutput<'py>> {
    let design = Hdf5Matrix::open(file_path, column_paths, n_rows, 64)
        .map_err(|e| PyValueError::new_err(format!("Hdf5Matrix::open: {e}")))?;
    /* ... thread through to the existing path solver ... */
}
```

Then a Python helper class similar to `MmapDesignF64`:

```python
class Hdf5DesignF64:
    dtype: str = "f64"

    def __init__(self, file_path, column_paths, n_rows):
        self.file_path = file_path
        self.column_paths = column_paths
        self.n_rows = n_rows
        self.n_cols = len(column_paths)
```

And an `_is_hdf5(x)` branch in the estimator's `fit()` that
dispatches to the new PyO3 entry point.

## Design considerations

### Caching

For backends where each column read is expensive (HDF5, network
storage), an LRU cache pays off — the CD inner loop revisits each
active feature many times. Size the cache to fit the active set (a
few hundred features for most problems).

For mmap backends, the OS page cache already does this transparently.
Don't add a redundant in-process cache.

### Concurrency

If your backend's read API is not thread-safe (some HDF5 versions
aren't), serialize access via a `Mutex`. The CD solver runs
`col_dot` from one thread at a time within a single fit; the Rayon
parallelism happens at the group level (block CD), not the column
level. So a coarse `Mutex` is usually enough.

### Out-of-core arithmetic

If the design is too big to even hold one column in RAM
(streaming-sized chunks of column-shaped data, etc.), you can
split each column into chunks and stream-reduce. See the
`ChunkedMatrix` source for a row-block version of this pattern; a
column-chunk version follows the same skeleton.

### When to wrap vs implement directly

Ask: does my backend produce $X$ in column-contiguous form? If yes,
implement `DesignMatrix` directly. If your backend produces $X$ in
some other form (row-major, blocked, sparse with a non-CSC layout)
and you can convert chunk-by-chunk to column-major, consider
**wrapping it via the `Chunked<C>` pattern** — your backend supplies
each chunk in the right shape, `Chunked` handles the routing.

## When to port the prototype to Rust

A Python `DesignMatrix` ABC isn't currently in v0.1 — the trait-
objects-via-FFI bridge is non-trivial and we deferred it. So
backend extension goes directly to Rust. The compiler's type
checking is actually a help here: implementing the trait surfaces
any place where you forgot a method or got the signature wrong.

## Verification checklist

Before shipping a new backend:

- [ ] Implement all five trait methods.
- [ ] `col_sq_norms` precomputed at construction.
- [ ] `Sync + Send` (cargo will tell you if it isn't).
- [ ] 4-5 unit tests against an in-memory `DenseMatrix` reference
      (one per trait method).
- [ ] One solver-equivalence test running `solve_path` on both
      backends and asserting β matches at every λ within 1e-7.
- [ ] If you added a PyO3 entry point, wire the dispatch into the
      relevant estimator(s) and add a pytest equivalence test.

The mmap and chunked backends in `crates/skein-core/src/design/`
are both ~250 lines of Rust + tests; they're useful templates.

## See also

- [Concepts: Backends](../concepts/backends.md) — the existing
  backend zoo and how the wrapper layers (`Augmented`, `Standardized`)
  compose.
- [Extending: penalties](penalty.md) — the same prototype-then-port
  recipe for regularizers.
- [Extending: datafits](datafit.md) — same recipe for losses.
