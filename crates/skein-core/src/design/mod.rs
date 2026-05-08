//! Design-matrix abstraction.
//!
//! The solver only ever touches `X` through this trait, which is what keeps
//! the door open for sparse, memory-mapped, or chunked backends without
//! rewriting the optimization code.

mod augmented;
mod dense;
mod mmap;
mod sparse_csc;
mod standardized;

pub use augmented::Augmented;
pub use dense::DenseMatrix;
pub use mmap::MmapMatrix;
pub use sparse_csc::SparseCSC;
pub use standardized::Standardized;

use ndarray::{Array1, Array2, ArrayView1};

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
}
