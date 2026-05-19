//! Design-matrix abstraction.
//!
//! The solver only ever touches `X` through this trait, which is what keeps
//! the door open for sparse, memory-mapped, or chunked backends without
//! rewriting the optimization code.

mod augmented;
mod chunked;
mod dense;
pub(crate) mod gram;
mod mmap;
mod mmap_f32;
mod multitask;
mod orthonormalize;
mod sparse_csc;
mod standardized;

pub use augmented::Augmented;
pub use chunked::Chunked;
pub use dense::DenseMatrix;
pub(crate) use gram::GramDesign;
pub use mmap::MmapMatrix;
pub use mmap_f32::MmapMatrixF32;
pub use multitask::MultiTaskDesign;
pub use orthonormalize::{orthonormalize_groups_dense, BlockBackTransform};
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

    /// `Σ w_i X[i, j] v_i`. Generalises `col_dot` (w = 1) and shows up in
    /// the weighted-LS inner CD that every GLM's prox-Newton wrapper runs:
    /// `coord_grad_j = col_dot_weighted(j, sample_weights, residual) / n`.
    ///
    /// The default impl materialises `w · v` and routes through
    /// `col_dot`, which is correct but allocates an n-sized buffer per
    /// call. Backends with cheap column access should override with a
    /// single fused loop — the prox-Newton inner hits this for every
    /// coordinate update of every CD sweep.
    fn col_dot_weighted(&self, j: usize, w: ArrayView1<f64>, v: ArrayView1<f64>) -> f64 {
        let weighted: Array1<f64> = (0..w.len()).map(|i| w[i] * v[i]).collect();
        self.col_dot(j, weighted.view())
    }

    /// `Σ w_i X[i, j]²`. The weighted analogue of `col_sq_norm`; the
    /// prox-Newton wrapper precomputes this for every feature once per
    /// inner CD call to skip the per-coord Lipschitz scan that
    /// `LeastSquares::coord_lipschitz` would otherwise repeat for every
    /// coordinate update.
    ///
    /// Default impl copies the column via `columns(&[j])` and folds it
    /// in place; backends with O(1) column access should override.
    fn col_sq_norm_weighted(&self, j: usize, w: ArrayView1<f64>) -> f64 {
        let col = self.columns(&[j]);
        let n = self.n_samples();
        let mut s = 0.0_f64;
        for i in 0..n {
            let v = col[[i, 0]];
            s += w[i] * v * v;
        }
        s
    }

    /// Batched `weighted_col_sq_norms[j] = Σ w_i X[i, j]²` for every
    /// feature. The prox-Newton wrapper calls this once per outer iter
    /// to seed its coord-Lipschitz cache; with p manual per-column
    /// folds this was a measurable hot spot (~15 ms per outer iter on
    /// the medium Poisson bench). Override on backends where the
    /// computation collapses to a single batched matvec — e.g.,
    /// `DenseMatrix` materialises element-wise `X²` and routes
    /// `X²ᵀ w` through BLAS gemv.
    fn weighted_col_sq_norms(&self, w: ArrayView1<f64>) -> Array1<f64> {
        let p = self.n_features();
        Array1::from_iter((0..p).map(|j| self.col_sq_norm_weighted(j, w)))
    }

    /// `target += alpha * w * X[:, j]` in place. Used by the prox-Newton
    /// inner CD to maintain `wr = w · r` alongside `r` so coord gradients
    /// can be read as a plain (BLAS) `col_dot(j, wr) / n` instead of the
    /// manual weighted triple-product `col_dot_weighted` per coordinate.
    ///
    /// Default impl materialises `α · w · X[:, j]` and delegates the axpy
    /// to `col_axpy` — correct but allocates an n-sized buffer per call.
    /// Backends with cheap column access should override.
    fn col_axpy_weighted(
        &self,
        j: usize,
        alpha: f64,
        w: ArrayView1<f64>,
        target: ArrayViewMut1<f64>,
    ) {
        let col = self.columns(&[j]);
        let n = self.n_samples();
        let mut scaled = Array1::<f64>::zeros(n);
        for i in 0..n {
            scaled[i] = alpha * w[i] * col[[i, 0]];
        }
        let mut t = target;
        t += &scaled;
    }
}
