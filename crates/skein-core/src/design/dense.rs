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
}
