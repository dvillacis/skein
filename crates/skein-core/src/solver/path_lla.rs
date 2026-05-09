//! Scalar λ-path solver with an LLA outer loop.
//!
//! Mirrors [`solve_block_path_lla`](crate::solver::block_path_lla) but for
//! **separable** non-convex penalties (bridge `|β|^q` with q < 1, adaptive
//! lasso pilot-fit reweighting, etc.) where the inner surrogate is a
//! **weighted L1** instead of a weighted group lasso.
//!
//! The user supplies `make_inner(β, λ, base_weights) → Box<dyn Penalty>`
//! that builds the inner convex surrogate from the current iterate. For
//! bridge, this is typically
//! `ElasticNet::with_weights(λ, 1.0, surrogate_weights_bridge(β, q, ε, w_base))`.
//! For adaptive lasso, the closure ignores `β` (and the pilot is
//! pre-computed before this function is called).
//!
//! Inner solve uses [`cd_solve_warm`]; the outer loop runs until either
//! `max_outer` iterations or `max_g |β_new − β_old|` falls below
//! `outer_tol`. λ-grid is built from `lambda_max` evaluated at `β = 0`
//! against the base weights — i.e. the surrogate at the cold start
//! (which for bridge with `eps > 0` has finite weights).

use crate::datafit::Datafit;
use crate::design::DesignMatrix;
use crate::penalty::Penalty;
use crate::solver::cd::{cd_solve_warm, CdConfig};
use crate::solver::path::{lambda_grid, lambda_max};
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone)]
pub struct PathLLAReport {
    pub lambdas: Vec<f64>,
    /// Outer LLA iterations performed at each λ.
    pub outer_iters: Vec<usize>,
    /// Whether each λ's outer loop converged within `outer_tol`.
    pub outer_converged: Vec<bool>,
    /// Sum of CD inner iters across all outer iters at each λ.
    pub inner_iters: Vec<usize>,
    /// Final objective at each λ — `datafit.value(r) + inner_pen.value(β)`,
    /// where `inner_pen` is the surrogate at the converged β.
    pub final_objs: Vec<f64>,
}

/// λ-path solve with an LLA outer loop wrapping `cd_solve_warm`.
///
/// `make_inner` is called once per outer iteration with the current
/// `(β, λ, base_weights)` and must return the convex inner surrogate.
/// `base_weights` is forwarded as a view so callers don't need to clone
/// the per-feature weight vector inside the closure on every call.
#[allow(clippy::too_many_arguments)]
pub fn solve_path_lla<F>(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    base_weights: Array1<f64>,
    make_inner: F,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    explicit_lambdas: Option<Vec<f64>>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array2<f64>, PathLLAReport)
where
    F: Fn(ArrayView1<'_, f64>, f64, ArrayView1<'_, f64>) -> Box<dyn Penalty>,
{
    let p = design.n_features();
    debug_assert_eq!(
        base_weights.len(),
        p,
        "base_weights length must equal n_features"
    );

    let lambdas = match explicit_lambdas {
        Some(v) => v,
        None => {
            // The LLA inner surrogate at β = 0 has weights `q · ε^(q−1) · w_base`
            // for bridge (a constant rescaling), or the pilot-fit weights for
            // adaptive — either way, the cold-start L1-effective view is
            // proportional to `base_weights`. We use `base_weights` directly
            // here so users specify the natural λ scale; callers wanting an
            // exact match to the surrogate's λ_max can pass `explicit_lambdas`.
            let lam_max = lambda_max(design, datafit, base_weights.view());
            lambda_grid(lam_max, n_lambdas, lambda_min_ratio)
        }
    };

    let n_lams = lambdas.len();
    let mut betas = Array2::<f64>::zeros((n_lams, p));
    let mut warm = Array1::<f64>::zeros(p);
    let mut outer_iters_out = Vec::with_capacity(n_lams);
    let mut outer_converged_out = Vec::with_capacity(n_lams);
    let mut inner_iters_out = Vec::with_capacity(n_lams);
    let mut final_objs_out = Vec::with_capacity(n_lams);

    for (k, &lam) in lambdas.iter().enumerate() {
        let mut total_inner = 0usize;
        let mut outer_iters = 0usize;
        let mut outer_converged = false;

        for outer in 0..max_outer {
            outer_iters = outer + 1;
            let pen = make_inner(warm.view(), lam, base_weights.view());
            let beta_old = warm.clone();
            let (new_beta, inner_report) =
                cd_solve_warm(warm, design, datafit, &*pen, cd_config);
            warm = new_beta;
            total_inner += inner_report.iter;

            let max_change = (0..p)
                .map(|j| (warm[j] - beta_old[j]).abs())
                .fold(0.0_f64, f64::max);
            if max_change < outer_tol {
                outer_converged = true;
                break;
            }
        }

        let final_pen = make_inner(warm.view(), lam, base_weights.view());
        let r = datafit.init_residual(design, warm.view());
        let final_obj = datafit.value(r.view()) + final_pen.value(warm.view());

        betas.row_mut(k).assign(&warm);
        outer_iters_out.push(outer_iters);
        outer_converged_out.push(outer_converged);
        inner_iters_out.push(total_inner);
        final_objs_out.push(final_obj);
    }

    (
        betas,
        PathLLAReport {
            lambdas,
            outer_iters: outer_iters_out,
            outer_converged: outer_converged_out,
            inner_iters: inner_iters_out,
            final_objs: final_objs_out,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::DenseMatrix;
    use crate::penalty::{ElasticNet, Mcp};
    use crate::solver::lla::surrogate_weights_bridge;
    use crate::solver::path::{solve_path, PathConfig, Screening};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    fn ls_problem(seed: u64) -> (DenseMatrix, Array1<f64>) {
        let n = 200;
        let p = 10;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 2.0;
        true_beta[1] = -1.5;
        true_beta[2] = 1.0;
        let mut y = x.dot(&true_beta);
        for i in 0..n {
            y[i] += 0.1 * sample();
        }
        (DenseMatrix::new(x), y)
    }

    #[test]
    fn path_lla_at_q_one_matches_lasso_path() {
        // Bridge at q = 1 IS plain weighted L1. With base_weights = 1 and
        // ε small but positive, `surrogate_weights_bridge(β, 1, ε, 1)`
        // returns `1 · (|β| + ε)^0 = 1` for every coordinate — exact lasso.
        let (design, y) = ls_problem(1);
        let p = design.n_features();
        let datafit = LeastSquares::new(y.clone());
        let base = Array1::<f64>::ones(p);

        let make_inner = |beta: ArrayView1<'_, f64>,
                          lam: f64,
                          w_base: ArrayView1<'_, f64>|
         -> Box<dyn Penalty> {
            let w = surrogate_weights_bridge(beta, 1.0, 1e-12, w_base);
            Box::new(ElasticNet::with_weights(lam, 1.0, w))
        };
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let (betas_lla, report_lla) = solve_path_lla(
            &design,
            &datafit,
            base.clone(),
            make_inner,
            10,
            1e-2,
            None,
            &cd_cfg,
            5,
            1e-9,
        );

        // Reference: plain lasso path via `Mcp` at very large γ.
        let datafit_ref = LeastSquares::new(y);
        let cfg = PathConfig {
            n_lambdas: 10,
            lambda_min_ratio: 1e-2,
            lambdas: Some(report_lla.lambdas.clone()),
            cd: cd_cfg.clone(),
            screening: Screening::Off,
        };
        let make_pen_ref = |lam: f64| -> Box<dyn Penalty> { Box::new(Mcp::new(lam, 1e9, p)) };
        let (betas_ref, _) = solve_path(&design, &datafit_ref, make_pen_ref, &cfg);

        for k in 0..betas_lla.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_lla[[k, j]], betas_ref[[k, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn path_lla_bridge_q_half_recovers_signal_at_smallest_lambda() {
        let (design, y) = ls_problem(2);
        let p = design.n_features();
        let datafit = LeastSquares::new(y);
        let base = Array1::<f64>::ones(p);

        let make_inner = |beta: ArrayView1<'_, f64>,
                          lam: f64,
                          w_base: ArrayView1<'_, f64>|
         -> Box<dyn Penalty> {
            let w = surrogate_weights_bridge(beta, 0.5, 1e-6, w_base);
            Box::new(ElasticNet::with_weights(lam, 1.0, w))
        };
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: Some(5),
        };
        let (betas, report) = solve_path_lla(
            &design,
            &datafit,
            base,
            make_inner,
            25,
            1e-3,
            None,
            &cd_cfg,
            10,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        // Active features 0, 1, 2 should have meaningful magnitudes,
        // noise features should be near zero.
        assert!(last_beta[0].abs() > 1.0);
        assert!(last_beta[1].abs() > 0.7);
        assert!(last_beta[2].abs() > 0.5);
        for j in 3..p {
            assert!(
                last_beta[j].abs() < 0.3,
                "noise feature {} has magnitude {}",
                j,
                last_beta[j]
            );
        }
    }

    #[test]
    fn path_lla_lambda_max_returns_zero() {
        let (design, y) = ls_problem(3);
        let p = design.n_features();
        let datafit = LeastSquares::new(y);
        let base = Array1::<f64>::ones(p);

        let make_inner = |beta: ArrayView1<'_, f64>,
                          lam: f64,
                          w_base: ArrayView1<'_, f64>|
         -> Box<dyn Penalty> {
            let w = surrogate_weights_bridge(beta, 0.5, 1e-6, w_base);
            Box::new(ElasticNet::with_weights(lam, 1.0, w))
        };
        let lam_max = lambda_max(&design, &datafit, base.view());
        let cd_cfg = CdConfig {
            max_iter: 200,
            tol: 1e-10,
            acceleration: None,
        };
        let (betas, _) = solve_path_lla(
            &design,
            &datafit,
            base,
            make_inner,
            1,
            1.0,
            Some(vec![lam_max]),
            &cd_cfg,
            5,
            1e-9,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-6);
        }
    }
}
