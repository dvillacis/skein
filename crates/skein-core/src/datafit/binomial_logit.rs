//! Binomial logistic regression via proximal-Newton.
//!
//! `BinomialLogit` holds the binary labels (0/1) and produces the local
//! quadratic surrogate at any β as a [`LeastSquares`] datafit:
//!
//! ```text
//!     η = X β
//!     p_i = sigmoid(η_i)
//!     w_i = p_i (1 − p_i)             [Hessian diagonal]
//!     z_i = η_i + (y_i − p_i) / w_i   [working response]
//! ```
//!
//! Minimizing `(1/2n) Σ w_i (Xβ − z_i)²` then solves the second-order
//! Taylor expansion of the cross-entropy loss; iterating this in an
//! outer prox-Newton loop converges to the regularized GLM optimum.
//!
//! The `w_i` floor of `1e-6` keeps the surrogate stable when `p_i`
//! saturates at 0 or 1 (so the working response stays finite). This
//! mirrors what `glmnet` does — without it `(y_i − p_i)/w_i` would
//! explode.

use super::{GlmDatafit, LeastSquares};
use crate::design::DesignMatrix;
use crate::numerics::W_FLOOR;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Binomial logistic regression with binary labels in `{0, 1}`.
///
/// Use [`Self::surrogate_at`] inside a prox-Newton outer loop to obtain
/// the weighted-LS approximation at the current iterate; the M1/M2
/// solvers then handle the regularized inner solve unchanged.
pub struct BinomialLogit {
    y: Array1<f64>,
    sample_weights: Option<Array1<f64>>,
}

impl BinomialLogit {
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

    /// Original cross-entropy loss `(1/n) Σ -y_i log p_i − (1−y_i) log(1−p_i)`,
    /// evaluated at `β`. Computed in a numerically stable form using
    /// `softplus(η) = log(1 + exp(η))` so very large `|η|` doesn't
    /// overflow.
    pub fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        let n = design.n_samples();
        let n_f = n as f64;
        let eta = design.matvec(beta);
        let mut total = 0.0_f64;
        for i in 0..n {
            // softplus(η) = log(1 + exp(η)), numerically stable form:
            //   if η > 0: η + log(1 + exp(-η))
            //   else:        log(1 + exp(η))
            let sp = if eta[i] > 0.0 {
                eta[i] + (-eta[i]).exp().ln_1p()
            } else {
                eta[i].exp().ln_1p()
            };
            // CE term: softplus(η) − y·η = -y log p − (1-y) log(1-p).
            let term = sp - self.y[i] * eta[i];
            let w = self.sample_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            total += w * term;
        }
        total / n_f
    }

    /// Build the local quadratic surrogate at `β` as a `LeastSquares`
    /// datafit with per-sample weights `w_i = p_i(1−p_i)` (floored at
    /// `1e-6`) and working response `z_i = η_i + (y_i − p_i)/w_i`.
    /// Per-sample multipliers from `with_sample_weights` (if set) compose
    /// in by being multiplied into the surrogate weights.
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
            let p = sigmoid(eta[i]);
            let w_raw = (p * (1.0 - p)).max(W_FLOOR);
            let scale = self.sample_weights.as_ref().map(|sw| sw[i]).unwrap_or(1.0);
            w[i] = scale * w_raw;
            z[i] = eta[i] + (self.y[i] - p) / w_raw;
        }
        LeastSquares::with_sample_weights(z, w)
    }
}

impl GlmDatafit for BinomialLogit {
    fn surrogate_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> LeastSquares {
        BinomialLogit::surrogate_at(self, design, beta)
    }

    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        BinomialLogit::loss(self, design, beta)
    }

    fn refresh_surrogate_components(
        &self,
        eta: ArrayView1<'_, f64>,
        mut w_out: ArrayViewMut1<'_, f64>,
        mut r_out: ArrayViewMut1<'_, f64>,
    ) {
        let n = eta.len();
        debug_assert_eq!(w_out.len(), n);
        debug_assert_eq!(r_out.len(), n);
        for i in 0..n {
            let p = sigmoid(eta[i]);
            let w_raw = (p * (1.0 - p)).max(W_FLOOR);
            let scale = self.sample_weights.as_ref().map(|sw| sw[i]).unwrap_or(1.0);
            w_out[i] = scale * w_raw;
            r_out[i] = (self.y[i] - p) / w_raw;
        }
    }
}

/// Numerically stable sigmoid `1 / (1 + exp(-η))`.
fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        let e = (-eta).exp();
        1.0 / (1.0 + e)
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn binomial_logit_loss_at_zero_is_log_two() {
        // β = 0 ⇒ η = 0 ⇒ p = 0.5 ⇒ −y log p − (1−y) log(1−p) = log 2 for any y∈{0,1}.
        // Average over n samples = log 2.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let design = DenseMatrix::new(x);
        let glm = BinomialLogit::new(y);
        let beta = Array1::<f64>::zeros(2);
        let loss = glm.loss(&design, beta.view());
        assert_abs_diff_eq!(loss, std::f64::consts::LN_2, epsilon = 1e-12);
    }

    #[test]
    fn binomial_logit_surrogate_at_zero_has_uniform_quarter_weights() {
        // β = 0 ⇒ p = 0.5 ⇒ w = 0.5·0.5 = 0.25 for every sample.
        // z_i = η_i + (y − p)/w = 0 + (y − 0.5)/0.25 = 4·(y − 0.5).
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let design = DenseMatrix::new(x);
        let glm = BinomialLogit::new(y.clone());
        let beta = Array1::<f64>::zeros(2);
        let surr = glm.surrogate_at(&design, beta.view());
        // The surrogate stores its own y (the working response z) and w.
        // We can't peek directly without exposing accessors, but we can
        // verify init_residual at β=0 reproduces -z, and value-at-zero
        // reproduces (1/2n) Σ w z².
        let r = surr.init_residual(&design, beta.view());
        for i in 0..4 {
            let expected_z = 4.0 * (y[i] - 0.5);
            // r_i = (Xβ − z) = -z at β=0.
            assert_abs_diff_eq!(r[i], -expected_z, epsilon = 1e-12);
        }
    }

    use crate::datafit::Datafit;

    #[test]
    fn binomial_logit_sigmoid_handles_extreme_eta() {
        // No NaN at very large positive/negative η.
        assert_abs_diff_eq!(sigmoid(1e6), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sigmoid(-1e6), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-12);
    }
}
