//! Design-matrix abstraction.
//!
//! The solver only ever touches `X` through this trait, which is what keeps
//! the door open for sparse, memory-mapped, or chunked backends without
//! rewriting the optimization code.

mod augmented;
mod chunked;
mod dense;
pub mod gram;
mod mmap;
mod mmap_f32;
mod multitask;
mod sparse_csc;
mod standardized;

pub use augmented::Augmented;
pub use chunked::Chunked;
pub use dense::DenseMatrix;
pub use gram::GramDesign;
pub use mmap::MmapMatrix;
pub use mmap_f32::MmapMatrixF32;
pub use multitask::MultiTaskDesign;
pub use sparse_csc::SparseCSC;
pub use standardized::Standardized;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

pub trait DesignMatrix: Sync + Send {
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;

    /// Returns `X β`.
    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64>;

    /// Returns `Xᵀ r`.
    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64>;

    /// `⟨X[:, j], v⟩`. Hot path for coordinate descent.
    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64;

    /// `‖X[:, j]‖²`. Cached by callers when possible.
    fn col_sq_norm(&self, j: usize) -> f64;

    /// Block of columns indexed by `cols` returned as an owned `(n, |cols|)`
    /// array. Used by group-block coordinate descent.
    fn columns(&self, cols: &[usize]) -> Array2<f64>;

    /// `r += alpha * X[:, j]` in place. Hot path for coordinate descent's
    /// residual update.
    ///
    /// The default implementation materialises the column via
    /// `columns(&[j])` and is correct for any backend, but it allocates a
    /// fresh `(n, 1)` `Array2` per call. Backends should override with a
    /// zero-alloc impl — the inner CD loop hits this for every nonzero
    /// coordinate update.
    ///
    /// `r` is taken as an `ArrayViewMut1` so backends can dispatch to
    /// ndarray's stride-aware `scaled_add` (which routes to BLAS `daxpy`
    /// on contiguous data and falls back to a vectorised loop otherwise).
    fn col_axpy(&self, j: usize, alpha: f64, mut r: ArrayViewMut1<f64>) {
        let col = self.columns(&[j]);
        r.scaled_add(alpha, &col.column(0));
    }
}
