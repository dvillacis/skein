//! Dense `f64` design matrix backed by `ndarray::Array2`.

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1};

pub struct DenseMatrix {
    x: Array2<f64>,
    col_sq_norms: Array1<f64>,
}

impl DenseMatrix {
    pub fn new(x: Array2<f64>) -> Self {
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
}
