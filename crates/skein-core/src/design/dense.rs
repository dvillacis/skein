//! Dense `f64` design matrix backed by `ndarray::Array2`.

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, ShapeBuilder};

pub struct DenseMatrix {
    x: Array2<f64>,
    col_sq_norms: Array1<f64>,
}

impl DenseMatrix {
    /// Constructs a backend over `x`, forcing column-major (Fortran)
    /// layout if the input is row-major.
    ///
    /// **Why F-order**: `col_dot`, `col_sq_norm`, and especially
    /// `col_axpy` (the hot path of CD's residual update) walk a single
    /// column repeatedly. In row-major (C) layout each column has stride
    /// `n_features` so successive elements live in different cache lines
    /// — every element costs an L1 miss, and BLAS daxpy/ddot don't
    /// dispatch on strided 1-D views. Forcing F-order makes
    /// `column(j)` contiguous; `scaled_add` then routes to BLAS daxpy
    /// and the inner loop runs at memory bandwidth.
    ///
    /// The one-shot copy is `n × p × 8 B` (~80 MB on the 10k × 1k
    /// medium bench), amortised over an entire path solve. Inputs that
    /// are already F-order are kept in place.
    pub fn new(x: Array2<f64>) -> Self {
        let x = if x.is_standard_layout() {
            // C-order input → physically copy into F-order so columns
            // become contiguous (stride 1).
            let (rows, cols) = x.dim();
            let mut x_f = Array2::<f64>::zeros((rows, cols).f());
            x_f.assign(&x);
            x_f
        } else {
            x
        };
        let col_sq_norms = x.map_axis(ndarray::Axis(0), |c| c.dot(&c));
        Self { x, col_sq_norms }
    }

    pub fn view(&self) -> ndarray::ArrayView2<'_, f64> {
        self.x.view()
    }
}

impl DesignMatrix for DenseMatrix {
    fn n_samples(&self) -> usize {
        self.x.nrows()
    }

    fn n_features(&self) -> usize {
        self.x.ncols()
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        self.x.dot(&beta)
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        self.x.t().dot(&r)
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        self.x.column(j).dot(&v)
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        self.col_sq_norms[j]
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let n = self.n_samples();
        let mut out = Array2::<f64>::zeros((n, cols.len()));
        for (k, &j) in cols.iter().enumerate() {
            out.column_mut(k).assign(&self.x.column(j));
        }
        out
    }

    fn col_axpy(&self, j: usize, alpha: f64, mut r: ArrayViewMut1<f64>) {
        // Stride-aware axpy via ndarray's `scaled_add`. Since `x` is
        // F-order (forced in `new`), the column view is contiguous —
        // `scaled_add`'s vectorised loop uses Zip's autovectorisation
        // hints which beat a naive indexed slice loop in practice.
        // (Tried a manual `for i in 0..n { r[i] += α · col[i]; }` loop;
        // it was ~30% slower than this on the medium lasso/LS bench.)
        r.scaled_add(alpha, &self.x.column(j));
    }

    fn col_dot_weighted(&self, j: usize, w: ArrayView1<f64>, v: ArrayView1<f64>) -> f64 {
        // Indexed-loop variants of this fused triple-dot run at
        // ~1 GFLOPS — bounds checks on `w[i]`/`v[i]` defeat
        // auto-vectorisation, even though the column is contiguous
        // (F-order). Switching to slice access via `as_slice` removes
        // the bounds checks and lets the compiler emit FMA + SIMD;
        // measured ~12× faster on the Poisson medium bench and
        // closes the gap to BLAS ddot for the same dot product.
        let col = self.x.column(j);
        let col_s = col.as_slice().expect("F-order column must be contiguous");
        let w_s = w.as_slice().expect("sample weights must be contiguous");
        let v_s = v.as_slice().expect("residual must be contiguous");
        let n = col_s.len();
        debug_assert_eq!(w_s.len(), n);
        debug_assert_eq!(v_s.len(), n);
        let mut s = 0.0_f64;
        for i in 0..n {
            s += w_s[i] * col_s[i] * v_s[i];
        }
        s
    }

    fn col_sq_norm_weighted(&self, j: usize, w: ArrayView1<f64>) -> f64 {
        let col = self.x.column(j);
        let col_s = col.as_slice().expect("F-order column must be contiguous");
        let w_s = w.as_slice().expect("sample weights must be contiguous");
        let n = col_s.len();
        debug_assert_eq!(w_s.len(), n);
        let mut s = 0.0_f64;
        for i in 0..n {
            let xi = col_s[i];
            s += w_s[i] * xi * xi;
        }
        s
    }

    fn weighted_col_sq_norms(&self, w: ArrayView1<f64>) -> Array1<f64> {
        // (X .² )ᵀ w via BLAS gemv. The element-wise square allocates an
        // (n × p) buffer (~80 MB on the medium bench, ~10× cheaper to
        // alloc than the savings from replacing a p-pass manual fold
        // with one gemv at memory bandwidth).
        let x_sq = self.x.mapv(|v| v * v);
        x_sq.t().dot(&w)
    }

    fn col_axpy_weighted(
        &self,
        j: usize,
        alpha: f64,
        w: ArrayView1<f64>,
        mut target: ArrayViewMut1<f64>,
    ) {
        let col = self.x.column(j);
        let col_s = col.as_slice().expect("F-order column must be contiguous");
        let w_s = w.as_slice().expect("sample weights must be contiguous");
        let t_s = target
            .as_slice_mut()
            .expect("target residual must be contiguous");
        let n = col_s.len();
        debug_assert_eq!(w_s.len(), n);
        debug_assert_eq!(t_s.len(), n);
        for i in 0..n {
            t_s[i] += alpha * w_s[i] * col_s[i];
        }
    }
}
