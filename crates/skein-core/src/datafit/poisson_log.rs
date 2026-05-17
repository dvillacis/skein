//! Poisson regression with the log link via proximal-Newton.
//!
//! `PoissonLog` holds non-negative count outcomes `y ∈ ℕ` (encoded as
//! `f64`) and produces the local quadratic surrogate at any β as a
//! [`LeastSquares`] datafit:
//!
//! ```text
//!     η_full = X β + offset           [offset = log-exposure if present]
//!     μ_i = exp(η_full_i)             [conditional mean = rate]
//!     w_i = μ_i                       [Hessian diagonal]
//!     z_i = (η_full_i − offset_i) + (y_i − μ_i) / w_i
//! ```
//!
//! With an offset, `μ_i = exp(X[i,:] β + offset_i)` so the linear
//! predictor `X β` is offset by a known per-sample term — common for
//! rate models where `offset_i = log(exposure_i)` (e.g., person-years
//! in epidemiology, observation-time in click-through-rate models).
//! The surrogate's working response subtracts the offset so β fits the
//! `Xβ` part directly; predictions need to add the offset back.
//!
//! Minimizing `(1/2n) Σ w_i (Xβ − z_i)²` solves the second-order Taylor
//! expansion of the Poisson negative log-likelihood at the current β;
//! iterating this in the prox-Newton outer loop converges to the
//! regularized GLM optimum.
//!
//! η is clamped to `[-30, 30]` before exponentiation to keep `μ` in a
//! finite range; outside this band the linear model has saturated and
//! further movement in η doesn't change the surrogate meaningfully.
//! `w_i` is then floored at `1e-6` so the working response stays
//! finite when the rate underflows.
//!
//! Loss reported by [`PoissonLog::loss`] is the canonical Poisson
//! negative log-likelihood (per sample, divided by `n`):
//!
//! ```text
//!     L(β) = (1/n) Σ_i (μ_i − y_i · η_i)
//! ```
//!
//! The `y_i!` constant is dropped since it doesn't depend on β.

use super::{GlmDatafit, LeastSquares};
use crate::design::DesignMatrix;
use crate::numerics::{ETA_CLAMP, W_FLOOR};
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Poisson regression with non-negative count outcomes and the canonical
/// log link.
///
/// Use [`Self::surrogate_at`] inside a prox-Newton outer loop to obtain
/// the weighted-LS approximation at the current iterate; the M1/M2
/// solvers then handle the regularized inner solve unchanged.
pub struct PoissonLog {
    y: Array1<f64>,
    offset: Option<Array1<f64>>,
    sample_weights: Option<Array1<f64>>,
}

fn validate_y_nonneg(y: ArrayView1<'_, f64>) {
    for &v in y.iter() {
        assert!(
            v >= 0.0 && v.is_finite(),
            "PoissonLog requires y ≥ 0 (got {})",
            v
        );
    }
}

impl PoissonLog {
    pub fn new(y: Array1<f64>) -> Self {
        validate_y_nonneg(y.view());
        Self {
            y,
            offset: None,
            sample_weights: None,
        }
    }

    pub fn with_offset(y: Array1<f64>, offset: Array1<f64>) -> Self {
        assert_eq!(y.len(), offset.len(), "offset length must equal y length");
        for &v in offset.iter() {
            assert!(
                v.is_finite(),
                "PoissonLog offset must be finite (got {})",
                v
            );
        }
        validate_y_nonneg(y.view());
        Self {
            y,
            offset: Some(offset),
            sample_weights: None,
        }
    }

    pub fn with_sample_weights(y: Array1<f64>, w: Array1<f64>) -> Self {
        assert_eq!(
            y.len(),
            w.len(),
            "sample_weights length must equal y length"
        );
        validate_y_nonneg(y.view());
        Self {
            y,
            offset: None,
            sample_weights: Some(w),
        }
    }

    pub fn with_sample_weights_and_offset(
        y: Array1<f64>,
        w: Array1<f64>,
        offset: Array1<f64>,
    ) -> Self {
        assert_eq!(
            y.len(),
            w.len(),
            "sample_weights length must equal y length"
        );
        assert_eq!(y.len(), offset.len(), "offset length must equal y length");
        for &v in offset.iter() {
            assert!(
                v.is_finite(),
                "PoissonLog offset must be finite (got {})",
                v
            );
        }
        validate_y_nonneg(y.view());
        Self {
            y,
            offset: Some(offset),
            sample_weights: Some(w),
        }
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }

    pub fn offset(&self) -> Option<ArrayView1<'_, f64>> {
        self.offset.as_ref().map(|o| o.view())
    }

    /// Poisson NLL `(1/n) Σ_i (μ_i − y_i · η_full_i)` evaluated at `β`,
    /// where `η_full_i = X[i,:] β + offset_i`. Per-sample weights (if
    /// set) multiply each term.
    pub fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        let n = design.n_samples();
        let n_f = n as f64;
        let mut eta_full = design.matvec(beta);
        if let Some(o) = &self.offset {
            for i in 0..n {
                eta_full[i] += o[i];
            }
        }
        let mut total = 0.0_f64;
        for i in 0..n {
            let eta_c = eta_full[i].clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu = eta_c.exp();
            let term = mu - self.y[i] * eta_c;
            let w = self.sample_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            total += w * term;
        }
        total / n_f
    }

    /// Build the local quadratic surrogate at `β` as a `LeastSquares`
    /// datafit with per-sample weights `w_i = μ_i` (floored at `1e-6`)
    /// and working response `z_i = (η_full_i − offset_i) + (y_i − μ_i)/w_i`.
    /// The surrogate is in `Xβ`-space (offset subtracted out), so the
    /// solver fits β unchanged. Per-sample multipliers from
    /// `with_sample_weights` (if set) compose by multiplying into the
    /// surrogate weights.
    pub fn surrogate_at(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<'_, f64>,
    ) -> LeastSquares {
        let n = design.n_samples();
        let mut eta_full = design.matvec(beta);
        if let Some(o) = &self.offset {
            for i in 0..n {
                eta_full[i] += o[i];
            }
        }
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta_c = eta_full[i].clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu = eta_c.exp();
            let w_raw = mu.max(W_FLOOR);
            let scale = self.sample_weights.as_ref().map(|sw| sw[i]).unwrap_or(1.0);
            w[i] = scale * w_raw;
            let offset_i = self.offset.as_ref().map(|o| o[i]).unwrap_or(0.0);
            z[i] = (eta_c - offset_i) + (self.y[i] - mu) / w_raw;
        }
        LeastSquares::with_sample_weights(z, w)
    }
}

impl GlmDatafit for PoissonLog {
    fn surrogate_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> LeastSquares {
        PoissonLog::surrogate_at(self, design, beta)
    }

    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64 {
        PoissonLog::loss(self, design, beta)
    }

    fn refresh_surrogate_components(
        &self,
        eta: ArrayView1<'_, f64>,
        mut w_out: ArrayViewMut1<'_, f64>,
        mut r_out: ArrayViewMut1<'_, f64>,
    ) {
        // The fused solver passes the un-offset `eta = X·β`; we
        // recombine with the optional offset here and apply the
        // ETA_CLAMP exactly as `surrogate_at` does.
        let n = eta.len();
        debug_assert_eq!(w_out.len(), n);
        debug_assert_eq!(r_out.len(), n);
        for i in 0..n {
            let offset_i = self.offset.as_ref().map(|o| o[i]).unwrap_or(0.0);
            let eta_c = (eta[i] + offset_i).clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu = eta_c.exp();
            let w_raw = mu.max(W_FLOOR);
            let scale = self.sample_weights.as_ref().map(|sw| sw[i]).unwrap_or(1.0);
            w_out[i] = scale * w_raw;
            r_out[i] = (self.y[i] - mu) / w_raw;
        }
    }

    fn glm_per_sample_loss_grad(&self, eta: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
        // `eta` here is the un-offset `X·β` (matches the convention used
        // by `refresh_surrogate_components`); apply the offset and the
        // same η clamp `loss` uses so the gradient and the dual obj see
        // the same μᵢ.
        let n = eta.len();
        let mut g = Array1::<f64>::zeros(n);
        for i in 0..n {
            let offset_i = self.offset.as_ref().map(|o| o[i]).unwrap_or(0.0);
            let eta_c = (eta[i] + offset_i).clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu = eta_c.exp();
            let sw = self.sample_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            g[i] = sw * (mu - self.y[i]);
        }
        Some(g)
    }

    fn glm_dual_obj(
        &self,
        design: &dyn DesignMatrix,
        eta: ArrayView1<'_, f64>,
        scale: f64,
    ) -> Option<f64> {
        // Derivation (with optional offset oᵢ; same η clamp `loss` uses):
        //   Per-sample primal loss   ℓᵢ(η_full) = exp(η_full) − yᵢ·η_full
        //                          = exp(oᵢ)·exp((Xβ)ᵢ) − yᵢ·(Xβ)ᵢ − yᵢ·oᵢ
        //   So skein's primal P_skein(β) = f((Xβ)) + C, where
        //     f(η) = (1/n) Σ wᵢ [exp(oᵢ)·exp(ηᵢ) − yᵢ·ηᵢ],
        //     C = −(1/n) Σ wᵢ·yᵢ·oᵢ.
        //   The Fenchel conjugate of `tη − exp(oᵢ)·exp(η)` is
        //     ℓᵢ*(t) = sᵢ·(log sᵢ − oᵢ − 1)  for sᵢ = t + yᵢ > 0,
        //   so f*(θ) = (1/n) Σ wᵢ·sᵢ·(log sᵢ − oᵢ − 1) with
        //   sᵢ = n·θᵢ/wᵢ + yᵢ.
        //   At θ_naive = (wᵢ/n)·(μᵢ − yᵢ) we get sᵢ = μᵢ > 0 (feasible).
        //   For θ_scaled = scale·θ_naive, sᵢ = scale·(μᵢ − yᵢ) + yᵢ ≥ 0
        //   (convex combination of two non-negatives). The composite
        //   dual is then D = −f*(θ_scaled) + C — the constant C cancels
        //   the −yᵢoᵢ contribution that the conjugate added back, so
        //   the net offset-dependent term in D is `−scale·(μᵢ − yᵢ)·oᵢ`.
        let n = eta.len();
        debug_assert_eq!(n, design.n_samples());
        let n_f = n as f64;
        let mut sum = 0.0_f64;
        for i in 0..n {
            let offset_i = self.offset.as_ref().map(|o| o[i]).unwrap_or(0.0);
            let eta_c = (eta[i] + offset_i).clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu = eta_c.exp();
            let s = self.y[i] + scale * (mu - self.y[i]);
            // sᵢ log sᵢ extended by continuity at sᵢ = 0 (no harm: yᵢ=0
            // AND μᵢ=0 would require yᵢ=0 since μᵢ > 0 always given the
            // clamp, but scale → 0 with yᵢ = 0 gives sᵢ → 0, and
            // x·log(x) → 0).
            let slogs = if s > 0.0 { s * s.ln() } else { 0.0 };
            let offset_term = scale * (mu - self.y[i]) * offset_i;
            let term = slogs - s - offset_term;
            let sw = self.sample_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            sum += sw * term;
        }
        Some(-sum / n_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::Datafit;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn poisson_loss_at_zero_is_one_minus_zero_times_y() {
        // β = 0 ⇒ η = 0 ⇒ μ = 1; loss term per sample = μ − y·η = 1 − 0 = 1.
        // Average over n samples = 1.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![3.0, 0.0, 1.0, 5.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::new(y);
        let beta = Array1::<f64>::zeros(2);
        let loss = glm.loss(&design, beta.view());
        assert_abs_diff_eq!(loss, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn poisson_surrogate_at_zero_has_unit_weights_and_residual_y_minus_one() {
        // β = 0 ⇒ μ = 1, w = 1, z = 0 + (y − 1)/1 = y − 1.
        // init_residual at β=0 in the surrogate: Xβ − z = −z = 1 − y.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![3.0, 0.0, 1.0, 5.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::new(y.clone());
        let beta = Array1::<f64>::zeros(2);
        let surr = glm.surrogate_at(&design, beta.view());
        let r = surr.init_residual(&design, beta.view());
        for i in 0..4 {
            // r_i = -(y_i - 1) = 1 - y_i.
            assert_abs_diff_eq!(r[i], 1.0 - y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn poisson_handles_extreme_eta_without_overflow() {
        // Large positive β ⇒ very large η. η is clamped to [-30, 30],
        // so μ stays in [exp(-30), exp(30)] and the loss is finite.
        let x = array![[10.0], [10.0], [10.0]];
        let y = array![1.0, 2.0, 3.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::new(y);
        let beta = array![100.0]; // η = 1000 before clamp
        let loss = glm.loss(&design, beta.view());
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        // Surrogate should also be finite.
        let surr = glm.surrogate_at(&design, beta.view());
        let r = surr.init_residual(&design, beta.view());
        for i in 0..3 {
            assert!(r[i].is_finite(), "residual must be finite at i={}", i);
        }
    }

    #[test]
    fn poisson_sample_weights_scale_loss_linearly() {
        let x = array![[1.0, 0.5], [0.5, 1.0]];
        let y = array![2.0, 4.0];
        let design = DenseMatrix::new(x);
        let beta = array![0.1, -0.2];

        let unw = PoissonLog::new(y.clone());
        let l_unw = unw.loss(&design, beta.view());

        // Doubled weights ⇒ doubled total loss. (Both numerator entries
        // double; the 1/n divisor stays the same.)
        let w = array![2.0, 2.0];
        let dbl = PoissonLog::with_sample_weights(y.clone(), w);
        let l_dbl = dbl.loss(&design, beta.view());
        assert_abs_diff_eq!(l_dbl, 2.0 * l_unw, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "y ≥ 0")]
    fn poisson_panics_on_negative_y() {
        let _ = PoissonLog::new(array![1.0, -1.0, 2.0]);
    }

    #[test]
    fn poisson_with_offset_loss_at_zero_matches_offset_only_baseline() {
        // β = 0 ⇒ η_full = offset. Per-sample term = exp(offset) − y·offset.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![3.0, 0.0, 2.0];
        let offset = array![0.5, -0.3, 1.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::with_offset(y.clone(), offset.clone());
        let beta = Array1::<f64>::zeros(2);
        let loss = glm.loss(&design, beta.view());
        let expected: f64 = (0..3)
            .map(|i| offset[i].exp() - y[i] * offset[i])
            .sum::<f64>()
            / 3.0;
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-12);
    }

    #[test]
    fn poisson_offset_zero_matches_no_offset() {
        // Offset of all-zeros must be exactly equivalent to no offset.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![3.0, 0.0, 1.0, 5.0];
        let beta = array![0.3, -0.2];
        let design = DenseMatrix::new(x);

        let no_off = PoissonLog::new(y.clone());
        let zero_off = PoissonLog::with_offset(y.clone(), Array1::<f64>::zeros(4));
        assert_abs_diff_eq!(
            no_off.loss(&design, beta.view()),
            zero_off.loss(&design, beta.view()),
            epsilon = 1e-12
        );

        let s_no = no_off.surrogate_at(&design, beta.view());
        let s_zero = zero_off.surrogate_at(&design, beta.view());
        let r_no = s_no.init_residual(&design, beta.view());
        let r_zero = s_zero.init_residual(&design, beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(r_no[i], r_zero[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn poisson_offset_constant_shift_matches_intercept_shift() {
        // Adding a constant `c` to offset is mathematically equivalent
        // to adding `c` to a unit-feature β (i.e., shifting an intercept
        // by `c`). At identical η_full, the surrogate must match.
        let x = array![[1.0, 0.5, 1.0], [0.5, 1.0, 1.0], [0.2, 0.8, 1.0]];
        // Last column is the intercept-style 1s; baseline test fixes
        // η_full identically across both formulations.
        let y = array![3.0, 0.0, 2.0];
        let design = DenseMatrix::new(x);

        // (a) constant offset c, β at last entry = 0.
        let c = 0.7_f64;
        let glm_off = PoissonLog::with_offset(y.clone(), Array1::from(vec![c; 3]));
        let beta_a = array![0.3, -0.2, 0.0];

        // (b) no offset, β at last entry = c.
        let glm_no = PoissonLog::new(y.clone());
        let beta_b = array![0.3, -0.2, c];

        assert_abs_diff_eq!(
            glm_off.loss(&design, beta_a.view()),
            glm_no.loss(&design, beta_b.view()),
            epsilon = 1e-12
        );
    }

    #[test]
    fn glm_dual_obj_collapses_to_primal_at_beta_zero_with_unit_scale() {
        // At β=0 (no offset), μ=1 ⇒ s=1 ⇒ D = (1/n) Σ wᵢ.
        // Primal = (1/n) Σ wᵢ μᵢ = (1/n) Σ wᵢ at β=0.
        let x = array![[1.0_f64, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![3.0, 0.0, 1.0, 5.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::new(y);
        let beta = Array1::<f64>::zeros(2);
        let eta = design.matvec(beta.view());
        let dual = glm
            .glm_dual_obj(&design, eta.view(), 1.0)
            .expect("poisson must return dual");
        let primal = glm.loss(&design, beta.view());
        assert_abs_diff_eq!(dual, primal, epsilon = 1e-12);
    }

    #[test]
    fn glm_dual_obj_collapses_to_primal_at_beta_zero_with_offset() {
        // Same identity with offset: β=0 is the primal optimum when
        // λ ≥ λ_max, and the dual at θ_naive must equal the primal.
        let x = array![[1.0_f64, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![3.0_f64, 0.0, 1.0, 5.0];
        let offset = array![0.5_f64, -0.3, 1.0, 0.2];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::with_offset(y, offset);
        let beta = Array1::<f64>::zeros(2);
        let eta = design.matvec(beta.view());
        let dual = glm
            .glm_dual_obj(&design, eta.view(), 1.0)
            .expect("poisson must return dual");
        let primal = glm.loss(&design, beta.view());
        assert_abs_diff_eq!(dual, primal, epsilon = 1e-12);
    }

    #[test]
    fn glm_dual_obj_is_lower_bound_at_arbitrary_beta() {
        // Same composite weak-duality test as binomial: pick λ at which
        // `scale·θ_naive` is just feasible, then dual ≤ penalized primal.
        let x = array![[1.0_f64, 0.5], [0.5, 1.0], [0.2, 0.8], [-0.3, 0.4]];
        let y = array![3.0_f64, 0.0, 1.0, 2.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::new(y);
        let beta = array![0.2_f64, -0.3];
        let eta = design.matvec(beta.view());
        let per_sample = glm
            .glm_per_sample_loss_grad(eta.view())
            .expect("poisson must return per-sample grad");
        let n_f = design.n_samples() as f64;
        let grad_beta = &design.rmatvec(per_sample.view()) / n_f;
        let lambda_bound = grad_beta.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let loss = glm.loss(&design, beta.view());
        for &scale in &[0.25_f64, 0.5, 0.75, 1.0] {
            let lam = scale * lambda_bound;
            let pen = lam * beta.iter().map(|&b| b.abs()).sum::<f64>();
            let primal = loss + pen;
            let dual = glm
                .glm_dual_obj(&design, eta.view(), scale)
                .expect("poisson must return dual");
            assert!(
                primal - dual >= -1e-12,
                "gap non-negative at scale={}, primal={}, dual={}",
                scale,
                primal,
                dual
            );
        }
    }

    #[test]
    fn glm_per_sample_loss_grad_matches_mu_minus_y() {
        let x = array![[1.0_f64, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![3.0, 0.0, 2.0];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::new(y.clone());
        let beta = array![0.2_f64, -0.3];
        let eta = design.matvec(beta.view());
        let g = glm
            .glm_per_sample_loss_grad(eta.view())
            .expect("poisson must return per-sample grad");
        for i in 0..3 {
            let mu = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp();
            assert_abs_diff_eq!(g[i], mu - y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn glm_per_sample_loss_grad_includes_offset() {
        let x = array![[1.0_f64, 0.5], [0.5, 1.0]];
        let y = array![3.0_f64, 0.0];
        let off = array![0.5_f64, -0.3];
        let design = DenseMatrix::new(x);
        let glm = PoissonLog::with_offset(y.clone(), off.clone());
        let beta = array![0.2_f64, -0.3];
        let eta = design.matvec(beta.view());
        let g = glm
            .glm_per_sample_loss_grad(eta.view())
            .expect("poisson must return per-sample grad");
        for i in 0..2 {
            let mu = (eta[i] + off[i]).clamp(-ETA_CLAMP, ETA_CLAMP).exp();
            assert_abs_diff_eq!(g[i], mu - y[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn poisson_with_offset_constructor_validates_lengths_and_finiteness() {
        let y = array![1.0, 2.0];
        let bad_len = array![0.5, -0.3, 0.1];
        let nan_offset = array![0.5, f64::NAN];
        let result = std::panic::catch_unwind(|| {
            PoissonLog::with_offset(y.clone(), bad_len);
        });
        assert!(result.is_err(), "should panic on length mismatch");
        let result = std::panic::catch_unwind(|| {
            PoissonLog::with_offset(y, nan_offset);
        });
        assert!(result.is_err(), "should panic on non-finite offset");
    }
}
