//! Huber regression — robust LS via proximal-Newton on a re-weighted
//! quadratic surrogate.
//!
//! `Huber` holds the targets `y` and a robustness scale `δ > 0`. The
//! Huber loss is the standard piecewise quadratic / linear:
//!
//! ```text
//!     ρ_δ(r) = ½ r²              for |r| ≤ δ
//!     ρ_δ(r) = δ |r| − ½ δ²       for |r| > δ
//! ```
//!
//! At the current iterate `β`, with residual `r_i = (Xβ)_i − y_i`, the
//! per-sample IRLS weight is
//!
//! ```text
//!     w_i = 1                    if |r_i| ≤ δ
//!     w_i = δ / |r_i|            if |r_i| > δ
//! ```
//!
//! and the prox-Newton working response is `z_i = y_i`. That is, Huber
//! reduces to weighted least-squares with `z = y` and the iteratively
//! re-weighted scaling above. Iterating this in an outer prox-Newton
//! loop converges to the regularized Huber optimum.
//!
//! Why `z = y` (and not `η + grad/w` like the logistic case)? In the
//! GLM setting `z = η + (y − μ)/w` is the standard IRLS working
//! response after a non-identity link. Huber uses the identity link
//! and the negative log-likelihood is just `½ ρ_δ(Xβ − y)`, so the
//! quadratic surrogate is centred on `y` directly. Working through the
//! algebra: the second-order Taylor expansion of `½ ρ_δ((Xβ)_i − y_i)`
//! at the current `η_i` is `½ w_i ((Xβ)_i − y_i)²` (off by a constant
//! that doesn't matter for the minimizer), which is exactly weighted
//! LS with `z_i = y_i` and weights `w_i`.
//!
//! The `w_i` floor of `1e-6` keeps the surrogate stable when an
//! outlier residual is so large that `δ/|r_i|` would underflow.

use super::{GlmDatafit, LeastSquares};
use crate::design::DesignMatrix;
use crate::numerics::W_FLOOR;
use ndarray::{Array1, ArrayView1};

/// Huber robust regression with the standard piecewise loss.
///
/// Use [`Self::surrogate_at`] inside a prox-Newton outer loop to obtain
/// the weighted-LS approximation at the current iterate; the M1/M2
/// solvers then handle the regularized inner solve unchanged.
pub struct Huber {
    y: Array1<f64>,
    delta: f64,
    sample_weights: Option<Array1<f64>>,
}

impl Huber {
    /// Construct a Huber datafit with robustness scale `delta`.
    /// `delta` must be strictly positive; the recommended starting
    /// point is the median absolute deviation of `y` from its median,
    /// scaled by 1.345 to reach 95% asymptotic efficiency at the
    /// normal (Huber, 1981).
    pub fn new(y: Array1<f64>, delta: f64) -> Self {
        assert!(delta > 0.0, "Huber delta must be strictly positive");
        assert!(delta.is_finite(), "Huber delta must be finite");
        Self {
            y,
            delta,
            sample_weights: None,
        }
    }

    pub fn with_sample_weights(y: Array1<f64>, delta: f64, w: Array1<f64>) -> Self {
        assert_eq!(
            y.len(),
            w.len(),
            "sample_weights length must equal y length"
        );
        assert!(delta > 0.0, "Huber delta must be strictly positive");
        Self {
            y,
            delta,
            sample_weights: Some(w),
        }
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Original Huber loss `(1/n) Σ ρ_δ(η_i − y_i)`, evaluated at `β`.
    /// Per-sample weights (if set) multiply the per-sample ρ.
    pub fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        let n = design.n_samples();
        let n_f = n as f64;
        let eta = design.matvec(beta);
        let mut total = 0.0_f64;
        for i in 0..n {
            let r = eta[i] - self.y[i];
            let ar = r.abs();
            let rho = if ar <= self.delta {
                0.5 * r * r
            } else {
                self.delta * (ar - 0.5 * self.delta)
            };
            let w = self.sample_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            total += w * rho;
        }
        total / n_f
    }

    /// Build the local quadratic surrogate at `β` as a `LeastSquares`
    /// datafit with per-sample weights `w_i = 1` if `|r_i| ≤ δ`, else
    /// `δ/|r_i|` (floored at `1e-6`), and working response `z_i = y_i`.
    /// Per-sample multipliers from `with_sample_weights` (if set)
    /// multiply into the surrogate weights.
    pub fn surrogate_at(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<'_, f64>,
    ) -> LeastSquares {
        let n = design.n_samples();
        let eta = design.matvec(beta);
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = eta[i] - self.y[i];
            let ar = r.abs();
            let w_raw = if ar <= self.delta {
                1.0
            } else {
                (self.delta / ar).max(W_FLOOR)
            };
            let scale = self.sample_weights.as_ref().map(|sw| sw[i]).unwrap_or(1.0);
            w[i] = scale * w_raw;
            z[i] = self.y[i];
        }
        LeastSquares::with_sample_weights(z, w)
    }
}

impl GlmDatafit for Huber {
    fn surrogate_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> LeastSquares {
        Huber::surrogate_at(self, design, beta)
    }

    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        Huber::loss(self, design, beta)
    }

    // Intentionally inherit the `None` defaults for `glm_per_sample_loss_grad`
    // and `glm_dual_obj`. Huber's conjugate is closed-form (the standard
    // "soft-thresholding-style" dual), but porting it is a separate
    // workstream — lower priority than logistic/Poisson where the
    // perf gap to celer is largest.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::Datafit;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn huber_loss_quadratic_regime_matches_half_ssq() {
        // δ chosen larger than any residual — Huber reduces to (1/n)·½‖r‖².
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![1.0, -0.5, 0.3];
        let design = DenseMatrix::new(x);
        let beta = Array1::<f64>::zeros(2);
        let huber = Huber::new(y.clone(), 10.0);
        // r = Xβ − y = -y at β=0, so ‖r‖² = ‖y‖².
        let ssq: f64 = y.iter().map(|v| v * v).sum();
        let expected = 0.5 * ssq / (y.len() as f64);
        assert_abs_diff_eq!(huber.loss(&design, beta.view()), expected, epsilon = 1e-12);
    }

    #[test]
    fn huber_loss_linear_regime_matches_l1_minus_constant() {
        // δ much smaller than any residual: ρ_δ(r) = δ|r| − ½δ²,
        // so total = (1/n) Σ (δ|r_i| − ½δ²) = δ·(Σ|r_i|)/n − ½δ².
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![10.0, -10.0, 5.0]; // |r| = |0 − y| = {10, 10, 5}
        let design = DenseMatrix::new(x);
        let beta = Array1::<f64>::zeros(1);
        let delta = 0.1;
        let huber = Huber::new(y.clone(), delta);
        let sum_abs: f64 = y.iter().map(|v| v.abs()).sum();
        let n = y.len() as f64;
        let expected = delta * sum_abs / n - 0.5 * delta * delta;
        assert_abs_diff_eq!(huber.loss(&design, beta.view()), expected, epsilon = 1e-12);
    }

    #[test]
    fn huber_surrogate_at_zero_quadratic_regime_has_uniform_weights() {
        // δ > max|y| ⇒ every sample is in the quadratic regime ⇒ w_i = 1
        // and z_i = y_i ⇒ surrogate ≡ unweighted LS in y, i.e. r_init = -y.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![1.0, -0.5, 0.3];
        let design = DenseMatrix::new(x);
        let huber = Huber::new(y.clone(), 10.0);
        let beta = Array1::<f64>::zeros(2);
        let surr = huber.surrogate_at(&design, beta.view());
        let r = surr.init_residual(&design, beta.view());
        for i in 0..3 {
            // r_i = (Xβ − z) = −y_i at β=0 with z = y.
            assert_abs_diff_eq!(r[i], -y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn huber_surrogate_at_zero_linear_regime_downweights_outliers() {
        // δ ≪ |y| ⇒ all samples in linear regime ⇒ w_i = δ/|y_i|.
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![10.0, -10.0, 5.0];
        let design = DenseMatrix::new(x);
        let delta = 0.5;
        let huber = Huber::new(y.clone(), delta);
        let beta = Array1::<f64>::zeros(1);
        let surr = huber.surrogate_at(&design, beta.view());
        let expected_w = [delta / 10.0, delta / 10.0, delta / 5.0];
        // Coord-Lipschitz on a constant column of 1s equals (1/n) Σ w_i x_ij².
        // For x = 1, this is (1/n) Σ w_i = mean(w).
        let lip = surr.coord_lipschitz(&design, 0);
        let expected_mean_w = expected_w.iter().sum::<f64>() / (y.len() as f64);
        assert_abs_diff_eq!(lip, expected_mean_w, epsilon = 1e-12);
    }

    #[test]
    fn huber_per_sample_weights_compose_into_surrogate() {
        // Quadratic regime; the user-side sample_weights should multiply
        // into surrogate weights and into the loss linearly.
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![1.0, -0.5, 0.3];
        let user_w = array![2.0, 1.0, 0.5];
        let design = DenseMatrix::new(x);
        let huber = Huber::with_sample_weights(y.clone(), 10.0, user_w.clone());
        let beta = Array1::<f64>::zeros(1);
        let lip = huber
            .surrogate_at(&design, beta.view())
            .coord_lipschitz(&design, 0);
        // Quadratic regime: surrogate w_raw = 1, so effective weight = user_w.
        let expected_mean_w = user_w.iter().sum::<f64>() / (y.len() as f64);
        assert_abs_diff_eq!(lip, expected_mean_w, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "delta")]
    fn huber_rejects_nonpositive_delta() {
        let _ = Huber::new(array![1.0, 2.0], 0.0);
    }
}
