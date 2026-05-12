//! Gram-matrix-shaped "design": wraps a `(p, p)` PSD matrix `W` and lets
//! the inner CD treat it as if it were an `X` with `XᵀX = W`.
//!
//! Used as the design for the inner column-wise lasso subproblem in
//! graphical lasso (Friedman et al. 2008): the optimality conditions of
//! glasso, peeled column by column, reduce to a weighted lasso on the
//! `(p-1) × (p-1)` Schur complement of `W` — and that lasso only needs
//! `W` and a right-hand side `s`, not an underlying `X`. Implementing
//! the gram form as a `DesignMatrix` lets the existing `cd_solve` run
//! the inner solve unchanged.
//!
//! `col_axpy(j, δ, r)` and `col_dot(j, v)` use `W[:, j]` directly,
//! which is exactly what the gram-form CD update needs.

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

pub struct GramDesign {
    gram: Array2<f64>,
}

impl GramDesign {
    pub fn new(gram: Array2<f64>) -> Self {
        assert_eq!(
            gram.nrows(),
            gram.ncols(),
            "GramDesign: gram must be square"
        );
        Self { gram }
    }

    pub fn gram(&self) -> &Array2<f64> {
        &self.gram
    }
}

impl DesignMatrix for GramDesign {
    fn n_samples(&self) -> usize {
        self.gram.nrows()
    }

    fn n_features(&self) -> usize {
        self.gram.ncols()
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        self.gram.dot(&beta)
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        self.gram.t().dot(&r)
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        self.gram.column(j).dot(&v)
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        let col = self.gram.column(j);
        col.dot(&col)
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n_samples(), cols.len()));
        for (k, &j) in cols.iter().enumerate() {
            out.column_mut(k).assign(&self.gram.column(j));
        }
        out
    }

    fn col_axpy(&self, j: usize, alpha: f64, mut r: ArrayViewMut1<f64>) {
        r.scaled_add(alpha, &self.gram.column(j));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn matvec_and_col_axpy_consistent() {
        let w = array![[2.0, 0.5, 0.1], [0.5, 1.5, 0.3], [0.1, 0.3, 1.0]];
        let d = GramDesign::new(w);
        let beta = array![1.0, -0.5, 0.2];
        let direct = d.matvec(beta.view());

        let mut r = Array1::<f64>::zeros(3);
        for j in 0..3 {
            d.col_axpy(j, beta[j], r.view_mut());
        }
        for i in 0..3 {
            assert_abs_diff_eq!(r[i], direct[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn col_dot_matches_manual() {
        let w = array![[2.0, 0.5], [0.5, 1.5]];
        let d = GramDesign::new(w);
        let v = array![3.0, -1.0];
        // W[:, 0] · v = 2·3 + 0.5·(-1) = 5.5
        assert_abs_diff_eq!(d.col_dot(0, v.view()), 5.5, epsilon = 1e-12);
    }
}
