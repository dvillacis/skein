//! Gram-form least squares: minimises `(1/2) βᵀ W β − sᵀ β`.
//!
//! Distinct from [`super::LeastSquares`] (which maintains the residual
//! `r = X β − y`); here we maintain `r = W β − s`, which *is* the
//! gradient. The inner CD update at coordinate `j` is therefore
//!
//! ```text
//!     β_j ← prox( β_j − r[j] / W[j,j],  1 / W[j,j] )
//! ```
//!
//! and the in-place residual update after `β_j += δ` is
//! `r += δ · W[:, j]` — the same `col_axpy` hot path the residual form
//! uses, applied to a [`super::super::design::gram::GramDesign`] that
//! returns `W[:, j]` from its `col_axpy`.
//!
//! Used as the inner subproblem in graphical lasso (Friedman et al.
//! 2008), where peeling column `k` of the working covariance reduces
//! glasso's stationarity conditions to a weighted lasso of this exact
//! shape. No `1/n` normalisation — the gram form is unnormalised by
//! construction.

use super::Datafit;
use crate::design::DesignMatrix;
use ndarray::{Array1, ArrayView1};

pub(crate) struct GramLeastSquares {
    rhs: Array1<f64>,
    diag: Array1<f64>,
}

impl GramLeastSquares {
    /// `gram_diag[j] = W[j, j]` (pre-extracted for fast per-coord Lipschitz).
    /// `rhs = s` (right-hand side of the gram-form problem).
    pub fn new(gram_diag: Array1<f64>, rhs: Array1<f64>) -> Self {
        assert_eq!(
            gram_diag.len(),
            rhs.len(),
            "GramLeastSquares: gram_diag and rhs must have the same length"
        );
        Self {
            rhs,
            diag: gram_diag,
        }
    }
}

impl Datafit for GramLeastSquares {
    /// **Not the true objective** — the gram-form quadratic
    /// `(1/2)βᵀWβ − sᵀβ` depends on β, but [`Datafit::value`] only
    /// receives the residual. We return `½ ‖r‖²` (half the gradient
    /// norm squared) as a monotone-ish proxy that's bounded and usable
    /// for report-only metrics. Disable Anderson acceleration when
    /// running gram-form CD, since its acceptance check compares
    /// `value()` across extrapolations.
    fn value(&self, residual: ArrayView1<f64>) -> f64 {
        0.5 * residual.dot(&residual)
    }

    fn init_residual(&self, design: &dyn DesignMatrix, beta: ArrayView1<f64>) -> Array1<f64> {
        let mut r = design.matvec(beta);
        r -= &self.rhs;
        r
    }

    fn coord_grad(&self, _design: &dyn DesignMatrix, j: usize, residual: ArrayView1<f64>) -> f64 {
        residual[j]
    }

    fn coord_lipschitz(&self, _design: &dyn DesignMatrix, j: usize) -> f64 {
        self.diag[j]
    }

    fn sample_weights(&self) -> Option<ArrayView1<'_, f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::gram::GramDesign;
    use crate::penalty::Mcp;
    use crate::solver::{cd_solve, CdConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn cd_minimises_gram_quadratic_no_penalty() {
        // Gram quadratic with W spd and known minimiser β* = W⁻¹ s.
        // Pick W = [[2, 0.5], [0.5, 1.5]], s = [1, 0.5] →
        // det = 2.75, β* = (1/2.75) · [[1.5, -0.5], [-0.5, 2]] · [1, 0.5]
        //              = (1/2.75) · [1.25, 0.5] ≈ [0.4545, 0.1818].
        let w = array![[2.0, 0.5], [0.5, 1.5]];
        let diag = array![2.0, 1.5];
        let s = array![1.0, 0.5];
        let design = GramDesign::new(w);
        let datafit = GramLeastSquares::new(diag, s);
        // Tiny λ ≈ no penalty.
        let penalty = Mcp::new(1e-12, 1e8, 2);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-14,
            acceleration: None,
        };
        let (beta, report) = cd_solve(&design, &datafit, &penalty, &cfg);
        assert!(report.converged);
        assert_abs_diff_eq!(beta[0], 1.25 / 2.75, epsilon = 1e-8);
        assert_abs_diff_eq!(beta[1], 0.5 / 2.75, epsilon = 1e-8);
    }

    #[test]
    fn cd_recovers_zero_under_strong_penalty() {
        let w = array![[2.0, 0.5], [0.5, 1.5]];
        let diag = array![2.0, 1.5];
        let s = array![0.1, 0.05];
        let design = GramDesign::new(w);
        let datafit = GramLeastSquares::new(diag, s);
        let penalty = Mcp::new(10.0, 3.0, 2);
        let cfg = CdConfig {
            max_iter: 100,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta, _) = cd_solve(&design, &datafit, &penalty, &cfg);
        assert_abs_diff_eq!(beta[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(beta[1], 0.0, epsilon = 1e-10);
    }
}
