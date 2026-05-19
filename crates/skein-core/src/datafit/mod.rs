//! Datafit traits and implementations.
//!
//! The trait is built around the residual `r = Xβ - y` (or its analogue for
//! GLMs) so that coordinate descent can update `r` in place rather than
//! recomputing `Xβ` after every coordinate update.
//!
//! GLMs (logistic, Poisson, …) implement [`GlmDatafit`] instead of
//! [`Datafit`] directly. They expose a `surrogate_at(β)` that returns a
//! [`LeastSquares`] datafit — the local prox-Newton quadratic — so the
//! M1/M2 inner solvers run unchanged. The prox-Newton outer loop in
//! `solver::prox_newton` / `solver::prox_newton_block` is generic over
//! `&dyn GlmDatafit`.

mod binomial_logit;
mod cox_ph;
mod gram_least_squares;
mod huber;
mod least_squares;
mod multinomial_logit;
mod poisson_log;

pub use binomial_logit::BinomialLogit;
pub use cox_ph::{CoxPH, TieHandling};
pub(crate) use gram_least_squares::GramLeastSquares;
pub use huber::Huber;
pub use least_squares::LeastSquares;
pub use multinomial_logit::MultinomialLogit;
pub use poisson_log::PoissonLog;

use crate::design::DesignMatrix;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// GLM-shaped datafit: exposes a local quadratic surrogate `surrogate_at(β)`
/// (the prox-Newton IRLS step) and the original (non-quadratic) `loss(β)`
/// for reporting / outer-loop monitoring.
///
/// Implementors must be `Sync + Send` so the prox-Newton outer loop and
/// downstream block-CD inner can dispatch work across Rayon threads.
pub trait GlmDatafit: Sync + Send {
    /// Build the local quadratic surrogate at `β` as a weighted-LS
    /// datafit. The returned [`LeastSquares`] has working response `z`
    /// and per-sample weights `w` such that minimizing
    /// `(1/2n) Σ w_i (Xβ − z_i)²` is the second-order Taylor expansion
    /// of the GLM negative log-likelihood at the current `β`.
    fn surrogate_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> LeastSquares;

    /// Original GLM loss at `β` (negative log-likelihood, divided by `n`,
    /// in the canonical form for the link function — see each
    /// implementor's docs for the exact formula). Used by the outer loop
    /// for reporting only.
    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64;

    /// Refresh the IRLS surrogate components in-place from a given
    /// linear predictor `eta`. The fused IRLS+CD solver in
    /// `solver::prox_newton::prox_newton_fused_solve` calls this once
    /// per outer iter, with `eta` maintained incrementally across
    /// coordinate updates (no need for a fresh `X·β` matvec).
    ///
    /// Writes to two output buffers:
    /// - `w_out[i] = effective per-sample IRLS weight` (incorporates
    ///   the GLM's Hessian-diagonal floor at `W_FLOOR` and any user-
    ///   supplied `sample_weights`).
    /// - `r_out[i] = (y_i − μ_i) / w_raw_i` = the **working residual**
    ///   in classical IRLS notation. Note: divides by the unscaled
    ///   `w_raw`, not the scaled `w_out[i]`, so `Σ x_ij · w_out_i · r_i
    ///   = Σ x_ij · scale_i · (y_i − μ_i)` — the natural form of the
    ///   coordinate gradient.
    ///
    /// Default impl is `unimplemented!()`; override in GLMs that want
    /// to support the fused solver (BinomialLogit / PoissonLog /
    /// CoxPH). Out-of-scope implementors (Huber, MultinomialLogit)
    /// keep the default and are routed through the classic
    /// `prox_newton_solve` instead.
    fn refresh_surrogate_components(
        &self,
        _eta: ArrayView1<'_, f64>,
        _w_out: ArrayViewMut1<'_, f64>,
        _r_out: ArrayViewMut1<'_, f64>,
    ) {
        unimplemented!(
            "refresh_surrogate_components must be overridden to route through \
             prox_newton_fused_solve; classic prox_newton_solve does not require it"
        );
    }

    /// Per-sample loss derivative at the linear predictor `eta = X·β`,
    /// weighted by `sample_weights` when present. Returns the vector
    /// `gᵢ = wᵢ · ℓᵢ'(ηᵢ)` so the caller can form the full β-gradient
    /// via `(1/n) Xᵀ g` in one rmatvec. The full-β gradient is what the
    /// gap-safe screening loop tests for L1 feasibility.
    ///
    /// Per GLM:
    /// - logistic: `gᵢ = wᵢ · (sigmoid(ηᵢ) − yᵢ)`
    /// - Poisson:  `gᵢ = wᵢ · (μᵢ − yᵢ)` with `μᵢ = exp(clamp(ηᵢ + oᵢ))`
    ///
    /// `None` (default) signals the GLM has no closed-form dual screening
    /// support; the path solver then falls back to KKT-only termination
    /// for that GLM (Cox / Huber / Multinomial). Implementors that
    /// override this must also override [`Self::glm_dual_obj`] so the
    /// feasibility scaling and dual obj are mutually consistent.
    fn glm_per_sample_loss_grad(&self, _eta: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
        None
    }

    /// Closed-form dual objective evaluated at `θ_scaled = scale · θ_naive`,
    /// where `θ_naive = ∇f(η)` is the natural dual point implied by the
    /// composite primal `min_β f(Xβ) + λR(β)`. Mirrors the role of
    /// [`Datafit::lasso_dual_obj`] but for GLMs whose loss is not LS.
    ///
    /// `scale ∈ (0, 1]` is the feasibility shrinkage chosen by the caller
    /// so that `‖Xᵀθ_scaled‖_∞ ≤ λ · w_j`. The dual is a valid lower
    /// bound on the primal optimum at every feasible θ; the closer to
    /// the saddle, the tighter the bound.
    ///
    /// Returns `None` (default) for GLMs without a closed-form dual
    /// (Cox / Huber / Multinomial). Caller (the path solver / prox-Newton
    /// outer loop) then skips gap-safe screening for that GLM.
    fn glm_dual_obj(
        &self,
        _design: &dyn DesignMatrix,
        _eta: ArrayView1<'_, f64>,
        _scale: f64,
    ) -> Option<f64> {
        None
    }
}

pub trait Datafit: Sync + Send {
    /// Loss value given the current residual.
    fn value(&self, residual: ArrayView1<'_, f64>) -> f64;

    /// Initialize working state (residual + any cached quantities) for a
    /// fresh `β`. For least squares this is `Xβ − y`. For weighted LS or a
    /// proximal-Newton GLM surrogate it's `Xβ − z` where `z` is the
    /// working response (the per-sample weights enter `coord_grad` /
    /// `coord_lipschitz` / `value`, not the residual itself).
    fn init_residual(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> Array1<f64>;

    /// Per-coordinate gradient `∂L/∂β_j` at the given residual.
    ///
    /// - Unweighted LS: `(1/n) X_jᵀ r`.
    /// - Weighted LS / GLM surrogate: `(1/n) Σ w_i x_ij r_i`.
    ///
    /// Required (no default) so every implementor must think about the
    /// formula — silent default-LS gradients are a foot-gun for GLMs.
    fn coord_grad(&self, design: &dyn DesignMatrix, j: usize, residual: ArrayView1<'_, f64>)
        -> f64;

    /// Full gradient `∂L/∂β` (length = n_features). Default impl loops
    /// over `coord_grad`; LS-shaped datafits should override with a
    /// single matvec for efficiency.
    fn full_grad(&self, design: &dyn DesignMatrix, residual: ArrayView1<'_, f64>) -> Array1<f64> {
        let p = design.n_features();
        Array1::from_iter((0..p).map(|j| self.coord_grad(design, j, residual)))
    }

    /// Lipschitz constant of the per-coordinate gradient. For unweighted
    /// LS: `‖X_j‖² / n`. For weighted LS: `(1/n) Σ w_i x_ij²`.
    fn coord_lipschitz(&self, design: &dyn DesignMatrix, j: usize) -> f64;

    /// Per-sample weights (length = n_samples). `None` = uniform 1.
    fn sample_weights(&self) -> Option<ArrayView1<'_, f64>>;

    /// Lasso-form dual objective at `θ_scaled = scale · θ_naive`, where
    /// `θ_naive` is the datafit's natural unscaled dual point (`-r/n`
    /// for LS). The "lasso form" name reflects that the formula
    /// assumes the constraint set has the shape `{θ : ‖Xᵀ θ‖_∞ ≤ λ · w_j}`
    /// — i.e., an L1-ball dual. For LS it works out to
    ///
    /// ```text
    ///     D(θ_scaled) = ‖r‖²/n · scale · (1 − scale/2) − scale · βᵀ grad
    /// ```
    ///
    /// (after eliminating `y` via `‖y‖² − ‖Xβ‖² = ‖r‖² − 2 nβᵀ grad`).
    /// `grad` must be the loss gradient `Xᵀ r / n`, supplied by the
    /// caller to avoid a duplicate `rmatvec`.
    ///
    /// Returns `None` for datafits that don't have a closed-form
    /// lasso-style dual (logistic, Poisson, Cox via prox-Newton):
    /// the path solver then falls back to the prox-gradient
    /// stationarity criterion for outer convergence and skips dual
    /// extrapolation. LS overrides; weighted-LS (`sample_weights`
    /// set) currently also returns `None` because the formula needs
    /// adjustment for the diagonal weight.
    fn lasso_dual_obj(
        &self,
        _design: &dyn DesignMatrix,
        _beta: ArrayView1<'_, f64>,
        _residual: ArrayView1<'_, f64>,
        _grad: ArrayView1<'_, f64>,
        _scale: f64,
    ) -> Option<f64> {
        None
    }
}
