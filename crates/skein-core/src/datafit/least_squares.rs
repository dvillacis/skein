use super::Datafit;
use crate::design::DesignMatrix;
use ndarray::{Array1, ArrayView1};

/// Least-squares loss `(1/2n) Σ w_i (Xβ_i − y_i)²` (uniform `w_i = 1` by
/// default; per-sample `w` honored throughout the trait when present —
/// `value`, `coord_grad`, `full_grad`, and `coord_lipschitz` all carry
/// the weight through).
pub struct LeastSquares {
    y: Array1<f64>,
    sample_weights: Option<Array1<f64>>,
}

impl LeastSquares {
    pub fn new(y: Array1<f64>) -> Self {
        Self {
            y,
            sample_weights: None,
        }
    }

    pub fn with_sample_weights(y: Array1<f64>, w: Array1<f64>) -> Self {
        assert_eq!(
            y.len(),
            w.len(),
            "sample_weights length must equal y length"
        );
        Self {
            y,
            sample_weights: Some(w),
        }
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }
}

impl Datafit for LeastSquares {
    fn value(&self, residual: ArrayView1<'_, f64>) -> f64 {
        let n = residual.len() as f64;
        match &self.sample_weights {
            None => 0.5 * residual.dot(&residual) / n,
            Some(w) => {
                let mut s = 0.0_f64;
                for i in 0..residual.len() {
                    s += w[i] * residual[i] * residual[i];
                }
                0.5 * s / n
            }
        }
    }

    fn init_residual(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut r = design.matvec(beta);
        r -= &self.y;
        r
    }

    fn coord_grad(&self, design: &dyn DesignMatrix, j: usize, residual: ArrayView1<'_, f64>) -> f64 {
        let n = design.n_samples() as f64;
        match &self.sample_weights {
            None => design.col_dot(j, residual) / n,
            Some(w) => {
                // (1/n) Σ w_i x_ij r_i — express as a column dot with a
                // weighted residual so we still ride the design's
                // `col_dot` fast path.
                let weighted: Array1<f64> = (0..residual.len())
                    .map(|i| w[i] * residual[i])
                    .collect();
                design.col_dot(j, weighted.view()) / n
            }
        }
    }

    fn full_grad(&self, design: &dyn DesignMatrix, residual: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = design.n_samples() as f64;
        match &self.sample_weights {
            None => &design.rmatvec(residual) / n,
            Some(w) => {
                let weighted: Array1<f64> = (0..residual.len())
                    .map(|i| w[i] * residual[i])
                    .collect();
                &design.rmatvec(weighted.view()) / n
            }
        }
    }

    fn coord_lipschitz(&self, design: &dyn DesignMatrix, j: usize) -> f64 {
        let n = design.n_samples() as f64;
        match &self.sample_weights {
            None => design.col_sq_norm(j) / n,
            Some(w) => {
                // (1/n) Σ w_i x_ij² — read column j explicitly since the
                // `DesignMatrix` trait doesn't expose a weighted-norm
                // helper.
                let mut s = 0.0_f64;
                let col = design.columns(&[j]);
                for i in 0..design.n_samples() {
                    let v = col[[i, 0]];
                    s += w[i] * v * v;
                }
                s / n
            }
        }
    }

    fn sample_weights(&self) -> Option<ArrayView1<'_, f64>> {
        self.sample_weights.as_ref().map(|w| w.view())
    }
}
