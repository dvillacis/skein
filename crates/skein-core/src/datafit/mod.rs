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
mod least_squares;
mod poisson_log;

pub use binomial_logit::BinomialLogit;
pub use cox_ph::CoxPH;
pub use least_squares::LeastSquares;
pub use poisson_log::PoissonLog;

use crate::design::DesignMatrix;
use ndarray::{Array1, ArrayView1};

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
    fn surrogate_at(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<'_, f64>,
    ) -> LeastSquares;

    /// Original GLM loss at `β` (negative log-likelihood, divided by `n`,
    /// in the canonical form for the link function — see each
    /// implementor's docs for the exact formula). Used by the outer loop
    /// for reporting only.
    fn loss(&self, design: &dyn DesignMatrix, beta: ArrayView1<'_, f64>) -> f64;
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
    fn coord_grad(&self, design: &dyn DesignMatrix, j: usize, residual: ArrayView1<'_, f64>) -> f64;

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
}
