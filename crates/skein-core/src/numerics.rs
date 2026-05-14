//! Cross-cutting numerical guards shared by GLM datafits.
//!
//! Centralized so a future bump (e.g. tightening the η clamp for a
//! more aggressive Newton step, or relaxing the weight floor) is one
//! edit instead of grepping per-datafit `const` definitions.
//!
//! - [`W_FLOOR`] is a lower bound on the IRLS / prox-Newton diagonal
//!   weight `w_i`. The working response `z_i = η_i + (y_i − μ_i)/w_i`
//!   (or its Huber / Cox analogue) divides by `w_i`, so an unfloored
//!   `w_i ↓ 0` would explode `z_i` and destabilize the inner LS solve.
//!   `glmnet` / `ncvreg` use the same guard.
//!
//! - [`ETA_CLAMP`] bounds the absolute value of the linear predictor
//!   `η = Xβ` before exponentiation in canonical-link GLMs (Poisson,
//!   Cox). `exp(±30)` ≈ `[9.4e-14, 1.07e13]` brackets every numerically
//!   meaningful rate; outside this band the model has saturated and
//!   further movement in η wouldn't change the surrogate.
//!
//! Logistic regression doesn't need the η clamp because `sigmoid(η)`
//! is naturally bounded in `(0, 1)` and the stable `softplus`
//! formulation handles large `|η|`.

/// Lower bound on the IRLS / prox-Newton diagonal weight to keep the
/// working response finite when the Hessian collapses.
pub const W_FLOOR: f64 = 1e-6;

/// Bound on `|η|` before `exp(η)` to keep the conditional mean / risk
/// score in a finite range. Applied by [`PoissonLog`](crate::datafit::PoissonLog)
/// and [`CoxPH`](crate::datafit::CoxPH).
pub const ETA_CLAMP: f64 = 30.0;
