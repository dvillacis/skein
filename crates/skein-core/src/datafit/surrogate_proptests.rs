//! Randomized identity tests for the GLM surrogates (H3).
//!
//! Each GLM has a hand-written suite that pins specific `(X, y, β)`
//! triples. The properties below complement that by quantifying the
//! *exact-at-β* identities the prox-Newton outer loop relies on,
//! evaluated over a randomized input domain:
//!
//! 1. **Gradient match.** The surrogate's per-coordinate gradient at the
//!    surrogate's initial residual must equal the GLM loss gradient at
//!    `β`. Verified by central finite differences against `loss(β ± ε·eⱼ)`
//!    — this catches the failure mode the property tier exists for
//!    (the working response `z` and surrogate weights `w` must compose
//!    so that `w · r = g`, the per-sample score), without duplicating
//!    the closed-form score formulas in the test.
//!
//! 2. **Hessian-diagonal match.** The surrogate's `coord_lipschitz(j)`
//!    must equal the analytical Fisher Hessian diagonal `(1/n) Σᵢ
//!    scaleᵢ · ℓ''(ηᵢ) · X[i, j]²`. Tested for BinomialLogit /
//!    PoissonLog (canonical links where the diagonal is exact); skipped
//!    for Cox because diagonal-IRLS Cox is intentionally approximate.
//!
//! Inputs are bounded (`X, β ∈ [-1, 1]`, `n ≤ 8`, `p ≤ 4`) so the
//! linear predictor stays well inside `ETA_CLAMP` and the per-sample
//! Hessian stays well above `W_FLOOR`. The identities are then exact up
//! to round-off + the central-difference truncation error.

use super::{BinomialLogit, CoxPH, Datafit, PoissonLog, TieHandling};
use crate::design::{DenseMatrix, DesignMatrix};
use ndarray::{Array1, Array2};
use proptest::collection::vec as prop_vec;
use proptest::prelude::*;

const N_SAMPLES: usize = 8;
const N_FEATURES: usize = 4;

fn arb_x_flat() -> impl Strategy<Value = Vec<f64>> {
    prop_vec(-1.0_f64..1.0, N_SAMPLES * N_FEATURES)
}

fn arb_beta() -> impl Strategy<Value = Vec<f64>> {
    prop_vec(-1.0_f64..1.0, N_FEATURES)
}

fn arb_sample_weights() -> impl Strategy<Value = Vec<f64>> {
    // ∈ [0.5, 2.0]: away from zero (no-op coord_grad branch) and away
    // from anything large enough to dominate the gradient.
    prop_vec(0.5_f64..2.0, N_SAMPLES)
}

fn arb_binary_y() -> impl Strategy<Value = Vec<f64>> {
    prop_vec(0u8..=1, N_SAMPLES).prop_map(|v| v.into_iter().map(|b| b as f64).collect())
}

fn arb_count_y() -> impl Strategy<Value = Vec<f64>> {
    // Small counts (0..5) match the regime tested in the Poisson
    // fixtures and keep `μ − y` well-behaved.
    prop_vec(0u32..=5, N_SAMPLES).prop_map(|v| v.into_iter().map(|c| c as f64).collect())
}

fn arb_cox_time() -> impl Strategy<Value = Vec<f64>> {
    prop_vec(0.1_f64..10.0, N_SAMPLES)
}

fn arb_cox_event() -> impl Strategy<Value = Vec<f64>> {
    prop_vec(0u8..=1, N_SAMPLES).prop_map(|v| v.into_iter().map(|b| b as f64).collect())
}

fn build_x(flat: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((N_SAMPLES, N_FEATURES), flat.to_vec()).expect("shape ok")
}

/// Central-difference estimate of `∂L/∂βⱼ` at `β`. Truncation error is
/// `O(ε²) · L'''(β)`; with ε = 1e-5 and the bounded inputs we use
/// (smooth `L`), the residual is `< 1e-10`.
fn fd_grad<F>(loss: F, beta: &Array1<f64>, j: usize, eps: f64) -> f64
where
    F: Fn(&Array1<f64>) -> f64,
{
    let mut bp = beta.clone();
    bp[j] += eps;
    let lp = loss(&bp);
    let mut bm = beta.clone();
    bm[j] -= eps;
    let lm = loss(&bm);
    (lp - lm) / (2.0 * eps)
}

/// `(1/n) Σᵢ scaleᵢ · ℓ''(ηᵢ) · X[i, j]²`. Analytical Fisher Hessian
/// diagonal; matches `coord_lipschitz` for BinomialLogit / PoissonLog.
fn analytical_hess_diag(
    design: &dyn DesignMatrix,
    j: usize,
    h: &Array1<f64>,
    sample_weights: Option<&[f64]>,
) -> f64 {
    let n = design.n_samples();
    let col = design.columns(&[j]);
    let mut s = 0.0_f64;
    for i in 0..n {
        let sw = sample_weights.map(|w| w[i]).unwrap_or(1.0);
        s += sw * h[i] * col[[i, 0]].powi(2);
    }
    s / n as f64
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

proptest! {
    /// Logistic surrogate gradient at β matches central-difference of
    /// the cross-entropy loss at β, to within FD truncation + round-off.
    #[test]
    fn binomial_surrogate_gradient_matches_loss_fd(
        x_flat in arb_x_flat(),
        beta_vec in arb_beta(),
        y_vec in arb_binary_y(),
        sw_vec in prop::option::weighted(0.5, arb_sample_weights()),
    ) {
        let x = build_x(&x_flat);
        let design = DenseMatrix::new(x);
        let y = Array1::from(y_vec);
        let beta = Array1::from(beta_vec);

        let glm = match sw_vec.clone() {
            None => BinomialLogit::new(y.clone()),
            Some(ref sw) => BinomialLogit::with_sample_weights(y.clone(), Array1::from(sw.clone())),
        };
        let surrogate = glm.surrogate_at(&design, beta.view());
        let r = surrogate.init_residual(&design, beta.view());

        let loss = |b: &Array1<f64>| glm.loss(&design, b.view());
        for j in 0..N_FEATURES {
            let actual = surrogate.coord_grad(&design, j, r.view());
            let expected = fd_grad(loss, &beta, j, 1e-5);
            prop_assert!(
                (actual - expected).abs() < 1e-7,
                "j={}, fd={}, surrogate={}",
                j, expected, actual,
            );
        }
    }

    /// Logistic Hessian diagonal: ℓ''(η) = p(1−p). With the bounded
    /// inputs `|η| ≤ 4`, sigmoid stays in [0.018, 0.982], so p(1−p) ≥
    /// 0.018 — well above `W_FLOOR = 1e-4`. Flooring is inactive and
    /// the identity is exact.
    #[test]
    fn binomial_surrogate_lipschitz_matches_analytical(
        x_flat in arb_x_flat(),
        beta_vec in arb_beta(),
        y_vec in arb_binary_y(),
        sw_vec in prop::option::weighted(0.5, arb_sample_weights()),
    ) {
        let x = build_x(&x_flat);
        let design = DenseMatrix::new(x);
        let y = Array1::from(y_vec);
        let beta = Array1::from(beta_vec);

        let glm = match sw_vec.clone() {
            None => BinomialLogit::new(y.clone()),
            Some(ref sw) => BinomialLogit::with_sample_weights(y.clone(), Array1::from(sw.clone())),
        };
        let surrogate = glm.surrogate_at(&design, beta.view());

        let eta = design.matvec(beta.view());
        let h: Array1<f64> = (0..N_SAMPLES).map(|i| {
            let p = sigmoid(eta[i]);
            p * (1.0 - p)
        }).collect();

        for j in 0..N_FEATURES {
            let actual = surrogate.coord_lipschitz(&design, j);
            let expected = analytical_hess_diag(&design, j, &h, sw_vec.as_deref());
            prop_assert!(
                (actual - expected).abs() < 1e-10,
                "j={}, expected={}, actual={}",
                j, expected, actual,
            );
        }
    }

    /// Poisson surrogate gradient matches loss FD with an optional
    /// offset (exercises the `(η_full − offset)` recentering in `z`).
    #[test]
    fn poisson_surrogate_gradient_matches_loss_fd(
        x_flat in arb_x_flat(),
        beta_vec in arb_beta(),
        y_vec in arb_count_y(),
        offset_vec in prop::option::weighted(0.5, prop_vec(-0.5_f64..0.5, N_SAMPLES)),
        sw_vec in prop::option::weighted(0.5, arb_sample_weights()),
    ) {
        let x = build_x(&x_flat);
        let design = DenseMatrix::new(x);
        let y = Array1::from(y_vec);
        let beta = Array1::from(beta_vec);

        let glm = match (offset_vec.clone(), sw_vec.clone()) {
            (None, None) => PoissonLog::new(y.clone()),
            (Some(ref o), None) => PoissonLog::with_offset(y.clone(), Array1::from(o.clone())),
            (None, Some(ref sw)) => PoissonLog::with_sample_weights(y.clone(), Array1::from(sw.clone())),
            (Some(ref o), Some(ref sw)) => PoissonLog::with_sample_weights_and_offset(
                y.clone(), Array1::from(sw.clone()), Array1::from(o.clone()),
            ),
        };
        let surrogate = glm.surrogate_at(&design, beta.view());
        let r = surrogate.init_residual(&design, beta.view());

        let loss = |b: &Array1<f64>| glm.loss(&design, b.view());
        for j in 0..N_FEATURES {
            let actual = surrogate.coord_grad(&design, j, r.view());
            let expected = fd_grad(loss, &beta, j, 1e-5);
            prop_assert!(
                (actual - expected).abs() < 1e-7,
                "j={}, fd={}, surrogate={}",
                j, expected, actual,
            );
        }
    }

    /// Poisson Hessian diagonal: ℓ''(η_full) = μ. With `|η_full| ≤ 4.5`,
    /// `μ ≥ exp(-4.5) ≈ 0.011 > W_FLOOR`, so flooring is inactive.
    #[test]
    fn poisson_surrogate_lipschitz_matches_analytical(
        x_flat in arb_x_flat(),
        beta_vec in arb_beta(),
        y_vec in arb_count_y(),
        offset_vec in prop::option::weighted(0.5, prop_vec(-0.5_f64..0.5, N_SAMPLES)),
        sw_vec in prop::option::weighted(0.5, arb_sample_weights()),
    ) {
        let x = build_x(&x_flat);
        let design = DenseMatrix::new(x);
        let y = Array1::from(y_vec);
        let beta = Array1::from(beta_vec);

        let glm = match (offset_vec.clone(), sw_vec.clone()) {
            (None, None) => PoissonLog::new(y.clone()),
            (Some(ref o), None) => PoissonLog::with_offset(y.clone(), Array1::from(o.clone())),
            (None, Some(ref sw)) => PoissonLog::with_sample_weights(y.clone(), Array1::from(sw.clone())),
            (Some(ref o), Some(ref sw)) => PoissonLog::with_sample_weights_and_offset(
                y.clone(), Array1::from(sw.clone()), Array1::from(o.clone()),
            ),
        };
        let surrogate = glm.surrogate_at(&design, beta.view());

        let eta = design.matvec(beta.view());
        let h: Array1<f64> = (0..N_SAMPLES).map(|i| {
            let o = offset_vec.as_ref().map(|o| o[i]).unwrap_or(0.0);
            (eta[i] + o).exp()
        }).collect();

        for j in 0..N_FEATURES {
            let actual = surrogate.coord_lipschitz(&design, j);
            let expected = analytical_hess_diag(&design, j, &h, sw_vec.as_deref());
            prop_assert!(
                (actual - expected).abs() < 1e-9,
                "j={}, expected={}, actual={}",
                j, expected, actual,
            );
        }
    }

    /// Cox PH score identity for both tie-handlers. The diagonal-IRLS
    /// surrogate's per-coordinate gradient at β matches the
    /// central-difference of the Breslow / Efron partial NLL. (The
    /// diagonal Hessian is intentionally an approximation, so the
    /// Lipschitz identity is not tested here.)
    #[test]
    fn cox_surrogate_gradient_matches_loss_fd(
        x_flat in arb_x_flat(),
        beta_vec in arb_beta(),
        time_vec in arb_cox_time(),
        event_vec in arb_cox_event(),
        ties in prop_oneof![Just(TieHandling::Breslow), Just(TieHandling::Efron)],
    ) {
        // Cox requires at least one observed event.
        prop_assume!(event_vec.iter().any(|&e| e > 0.5));

        let x = build_x(&x_flat);
        let design = DenseMatrix::new(x);
        let beta = Array1::from(beta_vec);
        let time = Array1::from(time_vec);
        let event = Array1::from(event_vec);

        let glm = CoxPH::with_ties(time, event, ties);
        let surrogate = glm.surrogate_at(&design, beta.view());
        let r = surrogate.init_residual(&design, beta.view());

        let loss = |b: &Array1<f64>| glm.loss(&design, b.view());
        for j in 0..N_FEATURES {
            let actual = surrogate.coord_grad(&design, j, r.view());
            let expected = fd_grad(loss, &beta, j, 1e-5);
            prop_assert!(
                (actual - expected).abs() < 1e-6,
                "j={}, ties={:?}, fd={}, surrogate={}",
                j, ties, expected, actual,
            );
        }
    }
}
