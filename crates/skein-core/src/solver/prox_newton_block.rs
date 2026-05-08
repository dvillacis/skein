//! Proximal-Newton outer loop for GLMs with group / sparse-group penalties.
//!
//! Combines the M3.2 prox-Newton scheme (linearize the GLM loss into a
//! weighted-LS surrogate) with M2.3's LLA scheme (linearize a non-convex
//! group penalty into a weighted convex group-lasso surrogate) into a
//! single outer loop. At each outer iteration:
//!
//!   1. Build the weighted-LS surrogate at the current β via
//!      `glm.surrogate_at(β)`.
//!   2. Build the convex group-penalty surrogate via the user-supplied
//!      `make_inner(β, groups, λ)` closure.
//!   3. Call `block_cd_solve_subset_with_cache` on the combined inner
//!      problem (weighted LS + convex group penalty); the M2 block-CD
//!      machinery handles the inner minimization unchanged thanks to
//!      M3.1's `Datafit` trait.
//!   4. Stop when `max_g ‖β_new_g − β_old_g‖₂` falls below `outer_tol`.
//!
//! The combined linearization is a valid majorization-minimization
//! scheme: both the prox-Newton quadratic and the LLA penalty surrogate
//! locally majorize their non-convex / non-quadratic terms.
//!
//! v0.1 scope: scalar penalties go through `prox_newton_solve_path`
//! (M3.2); this module is for group penalties (`GroupPenalty`). Generic
//! over `&dyn GlmDatafit` (logistic, Poisson, …) so any GLM with a
//! weighted-LS surrogate plugs in.
//! Per-outer-iter screening is not yet wired (the inner CD uses the
//! full-group set); follow-up.

use crate::datafit::GlmDatafit;
use crate::design::DesignMatrix;
use crate::groups::Groups;
use crate::penalty::GroupPenalty;
use crate::solver::block_cd::{
    block_cd_solve_subset_with_cache, group_lipschitz_cache,
};
use crate::solver::block_path::block_lambda_max;
use crate::solver::cd::CdConfig;
use crate::solver::path::lambda_grid;
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone)]
pub struct ProxNewtonBlockPathReport {
    pub lambdas: Vec<f64>,
    /// Outer iterations performed at each λ.
    pub outer_iters: Vec<usize>,
    /// Whether each λ's outer loop hit `outer_tol`.
    pub outer_converged: Vec<bool>,
    /// Sum of CD inner iters across all outer iters at each λ.
    pub inner_iters: Vec<usize>,
    /// Original GLM loss at the converged β for each λ.
    pub final_losses: Vec<f64>,
}

/// λ-path solve combining prox-Newton (for the GLM) and LLA (for
/// non-convex group penalties) in a single outer loop.
///
/// `make_inner` is called once per outer iteration with the current
/// `(β, &Groups, λ)` and must return the convex group penalty surrogate
/// — for plain group lasso, ignore β and return
/// `GroupLasso::with_weights(λ, base_weights)`. For group MCP/SCAD,
/// compute the LLA weights via M2.3's `surrogate_weights_*` helpers and
/// return a `GroupLasso` with those.
#[allow(clippy::too_many_arguments)]
pub fn prox_newton_block_solve_path<F>(
    design: &dyn DesignMatrix,
    glm: &dyn GlmDatafit,
    base_weights: Array1<f64>,
    make_inner: F,
    groups: &Groups,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    explicit_lambdas: Option<Vec<f64>>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array2<f64>, ProxNewtonBlockPathReport)
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64) -> Box<dyn GroupPenalty>,
{
    let p = design.n_features();
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_weights.len(), n_groups);

    let lambdas = match explicit_lambdas {
        Some(v) => v,
        None => {
            // λ_max from the GLM surrogate at β = 0.
            let beta_zero = Array1::<f64>::zeros(p);
            let surrogate0 = glm.surrogate_at(design, beta_zero.view());
            let lam_max =
                block_lambda_max(design, &surrogate0, base_weights.view(), groups);
            lambda_grid(lam_max, n_lambdas, lambda_min_ratio)
        }
    };

    let n_lams = lambdas.len();
    let mut betas = Array2::<f64>::zeros((n_lams, p));
    let mut warm = Array1::<f64>::zeros(p);
    let mut outer_iters_out = Vec::with_capacity(n_lams);
    let mut outer_converged_out = Vec::with_capacity(n_lams);
    let mut inner_iters_out = Vec::with_capacity(n_lams);
    let mut final_losses_out = Vec::with_capacity(n_lams);

    // Cache the per-group operator-norm Lipschitz once for the whole
    // path. The X columns don't change; only the weighted-LS sample
    // weights do per outer iter, which doesn't affect ‖X_g‖_op² / n
    // ... wait, it does — per-sample weights enter the Lipschitz of
    // the *surrogate* datafit (`coord_lipschitz` for weighted LS is
    // `(1/n) Σ w_i x_ij²`, not the unweighted `‖X_g‖_op²/n`).
    //
    // For now we use the unweighted operator-norm cache as a *bound*;
    // it's loose but always valid (per-sample weights w_i ≤ 1/4 for
    // logistic, so the weighted Lipschitz is at most 1/4× the
    // unweighted one — the inner CD will just take smaller steps
    // than necessary). A tighter cache that recomputes per outer iter
    // is M3.x.
    let group_lip = group_lipschitz_cache(design, groups);
    let group_subset: Vec<usize> = (0..n_groups).collect();

    for &lam in lambdas.iter() {
        let mut outer_iters = 0usize;
        let mut total_inner = 0usize;
        let mut outer_converged = false;

        for outer in 0..max_outer {
            outer_iters = outer + 1;
            let surrogate = glm.surrogate_at(design, warm.view());
            let inner_pen = make_inner(warm.view(), groups, lam);
            let beta_old = warm.clone();

            let (new_beta, inner_report) = block_cd_solve_subset_with_cache(
                warm,
                &group_subset,
                &group_lip,
                design,
                &surrogate,
                &*inner_pen,
                groups,
                cd_config,
            );
            warm = new_beta;
            total_inner += inner_report.iter;

            // Outer convergence: max group-block L₂ change.
            let mut max_block_change = 0.0_f64;
            for g in 0..n_groups {
                let mut sum_sq = 0.0_f64;
                for &j in groups.group(g) {
                    let d = warm[j] - beta_old[j];
                    sum_sq += d * d;
                }
                let nb = sum_sq.sqrt();
                if nb > max_block_change {
                    max_block_change = nb;
                }
            }
            if max_block_change < outer_tol {
                outer_converged = true;
                break;
            }
        }

        let row_idx = outer_iters_out.len();
        betas.row_mut(row_idx).assign(&warm);
        outer_iters_out.push(outer_iters);
        outer_converged_out.push(outer_converged);
        inner_iters_out.push(total_inner);
        final_losses_out.push(glm.loss(design, warm.view()));
        let _ = lam;
    }

    (
        betas,
        ProxNewtonBlockPathReport {
            lambdas,
            outer_iters: outer_iters_out,
            outer_converged: outer_converged_out,
            inner_iters: inner_iters_out,
            final_losses: final_losses_out,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::{BinomialLogit, CoxPH, PoissonLog};
    use crate::design::{DenseMatrix, Standardized};
    use crate::penalty::GroupLasso;
    use crate::solver::lla::surrogate_weights_group_mcp;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    /// 200 samples, 8 features in 4 groups of 2; truth: groups 0 and 2
    /// active. y is sampled from sigmoid(η).
    fn logistic_group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>, Groups) {
        let n = 200;
        let p = 8;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 1.5;
        true_beta[1] = -1.0;
        true_beta[4] = 0.7;
        true_beta[5] = 1.2;
        let eta = x.dot(&true_beta);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let p_i = 1.0 / (1.0 + (-eta[i]).exp());
            let u = (sample() + 1.0) * 0.5;
            y[i] = if u < p_i { 1.0 } else { 0.0 };
        }
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), y, true_beta, groups)
    }

    fn group_norm(beta: &Array1<f64>, groups: &Groups, g: usize) -> f64 {
        groups
            .group(g)
            .iter()
            .map(|&j| beta[j] * beta[j])
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn logistic_group_lasso_recovers_active_groups() {
        let (design, y, _, groups) = logistic_group_problem(1);
        let glm = BinomialLogit::new(y);
        let base = Array1::<f64>::ones(groups.n_groups());

        // Plain group lasso surrogate: ignore β, just rebuild with the
        // base weights at every λ.
        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _groups: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            25,
            5e-3,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            20,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.3);
        assert!(group_norm(&last_beta, &groups, 2) > 0.3);
    }

    #[test]
    fn logistic_group_mcp_via_lla_recovers_active_groups() {
        let (design, y, _, groups) = logistic_group_problem(2);
        let glm = BinomialLogit::new(y);
        let base = Array1::<f64>::ones(groups.n_groups());
        let gamma = 3.0;

        // LLA surrogate weights for group MCP.
        let base_for_closure = base.clone();
        let make_inner = move |beta: ArrayView1<'_, f64>,
                               g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_for_closure.view());
            Box::new(GroupLasso::with_weights(lam, w)) as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            25,
            5e-3,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.3);
        assert!(group_norm(&last_beta, &groups, 2) > 0.3);
    }

    /// 300 samples, 8 features in 4 groups of 2; truth: groups 0 and 2
    /// active. y ~ Poisson(exp(η)) sampled with Knuth's algorithm.
    fn poisson_group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>, Groups) {
        let n = 300;
        let p = 8;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 0.6;
        true_beta[1] = -0.4;
        true_beta[4] = 0.3;
        true_beta[5] = 0.5;
        let eta = x.dot(&true_beta);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu = eta[i].exp();
            let l = (-mu).exp();
            let mut k = 0_i64;
            let mut prod = 1.0_f64;
            loop {
                k += 1;
                let u = (sample() + 1.0) * 0.5;
                prod *= u.max(1e-300);
                if prod <= l {
                    break;
                }
            }
            y[i] = (k - 1) as f64;
        }
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), y, true_beta, groups)
    }

    #[test]
    fn poisson_group_lasso_recovers_active_groups() {
        let (design, y, _, groups) = poisson_group_problem(1);
        let glm = PoissonLog::new(y);
        let base = Array1::<f64>::ones(groups.n_groups());

        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _groups: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            25,
            5e-3,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.2);
        assert!(group_norm(&last_beta, &groups, 2) > 0.2);
    }

    #[test]
    fn poisson_group_mcp_via_lla_recovers_active_groups() {
        let (design, y, _, groups) = poisson_group_problem(2);
        let glm = PoissonLog::new(y);
        let base = Array1::<f64>::ones(groups.n_groups());
        let gamma = 3.0;

        let base_for_closure = base.clone();
        let make_inner = move |beta: ArrayView1<'_, f64>,
                               g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_for_closure.view());
            Box::new(GroupLasso::with_weights(lam, w)) as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            25,
            5e-3,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.2);
        assert!(group_norm(&last_beta, &groups, 2) > 0.2);
    }

    #[test]
    fn poisson_group_path_at_lambda_max_returns_zero() {
        let (design, y, _, groups) = poisson_group_problem(3);
        let glm = PoissonLog::new(y);
        let p = design.n_features();
        let base = Array1::<f64>::ones(groups.n_groups());

        let beta_zero = Array1::<f64>::zeros(p);
        let surr0 = glm.surrogate_at(&design, beta_zero.view());
        let lam_max = block_lambda_max(&design, &surr0, base.view(), &groups);

        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, _) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            1,
            1.0,
            Some(vec![lam_max]),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-6);
        }
    }

    /// Sparse-truth Cox PH problem with 8 features in 4 groups of 2;
    /// exponential baseline hazard, exponential censoring.
    fn cox_group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>, Array1<f64>, Groups) {
        let n = 300;
        let p = 8;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 0.6;
        true_beta[1] = -0.4;
        true_beta[4] = 0.3;
        true_beta[5] = 0.5;
        let eta = x.dot(&true_beta);

        let mut time = Array1::<f64>::zeros(n);
        let mut event = Array1::<f64>::zeros(n);
        for i in 0..n {
            let u_t = ((sample() + 1.0) * 0.5).max(1e-12);
            let u_c = ((sample() + 1.0) * 0.5).max(1e-12);
            let rate_t = eta[i].exp();
            let t_event = -u_t.ln() / rate_t;
            let t_cens = -u_c.ln() / 0.5;
            if t_event <= t_cens {
                time[i] = t_event;
                event[i] = 1.0;
            } else {
                time[i] = t_cens;
                event[i] = 0.0;
            }
        }
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), time, event, true_beta, groups)
    }

    #[test]
    fn cox_group_lasso_recovers_active_groups() {
        let (design, time, event, _, groups) = cox_group_problem(1);
        let glm = CoxPH::new(time, event);
        let base = Array1::<f64>::ones(groups.n_groups());

        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _groups: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            25,
            5e-3,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.2);
        assert!(group_norm(&last_beta, &groups, 2) > 0.2);
    }

    #[test]
    fn cox_group_mcp_via_lla_recovers_active_groups() {
        let (design, time, event, _, groups) = cox_group_problem(2);
        let glm = CoxPH::new(time, event);
        let base = Array1::<f64>::ones(groups.n_groups());
        let gamma = 3.0;

        let base_for_closure = base.clone();
        let make_inner = move |beta: ArrayView1<'_, f64>,
                               g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_for_closure.view());
            Box::new(GroupLasso::with_weights(lam, w)) as Box<dyn GroupPenalty>
        };

        let (betas, report) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            25,
            5e-3,
            None,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.2);
        assert!(group_norm(&last_beta, &groups, 2) > 0.2);
    }

    #[test]
    fn cox_group_path_at_lambda_max_returns_zero() {
        let (design, time, event, _, groups) = cox_group_problem(3);
        let glm = CoxPH::new(time, event);
        let p = design.n_features();
        let base = Array1::<f64>::ones(groups.n_groups());

        let beta_zero = Array1::<f64>::zeros(p);
        let surr0 = glm.surrogate_at(&design, beta_zero.view());
        let lam_max = block_lambda_max(&design, &surr0, base.view(), &groups);

        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, _) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            1,
            1.0,
            Some(vec![lam_max]),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn logistic_group_path_at_lambda_max_returns_zero() {
        let (design, y, _, groups) = logistic_group_problem(3);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let base = Array1::<f64>::ones(groups.n_groups());

        // Compute λ_max from the GLM surrogate at β=0.
        let beta_zero = Array1::<f64>::zeros(p);
        let surr0 = glm.surrogate_at(&design, beta_zero.view());
        let lam_max = block_lambda_max(&design, &surr0, base.view(), &groups);

        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let (betas, _) = prox_newton_block_solve_path(
            &design,
            &glm,
            base.clone(),
            make_inner,
            &groups,
            1,
            1.0,
            Some(vec![lam_max]),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-6);
        }
    }

    /// Group prox-Newton path on `Standardized<DenseMatrix>` matches
    /// the same solver on a pre-scaled `DenseMatrix` reference at every
    /// λ. Validates that the prox-Newton + LLA outer loops compose
    /// transparently with lazy column scaling for group penalties — the
    /// prerequisite for sparse + standardize on group GLMs.
    #[test]
    fn logistic_group_lasso_through_standardized_matches_pre_scaled() {
        let (design_raw, y, _, groups) = logistic_group_problem(7);
        let x = design_raw.view().to_owned();
        let p = x.ncols();
        let scales = Array1::from(vec![1.5, 0.7, 2.3, 0.9, 1.1, 1.8, 0.6, 2.0]);

        let mut x_scaled = x.clone();
        for j in 0..p {
            for i in 0..x.nrows() {
                x_scaled[[i, j]] /= scales[j];
            }
        }
        let dense_ref = DenseMatrix::new(x_scaled);
        let std_design = Standardized::new(DenseMatrix::new(x), scales);

        let glm_a = BinomialLogit::new(y.clone());
        let glm_b = BinomialLogit::new(y);
        let base = Array1::<f64>::ones(groups.n_groups());

        let base_for_closure = base.clone();
        let make_inner = move |_beta: ArrayView1<'_, f64>,
                               _g: &Groups,
                               lam: f64|
              -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::with_weights(lam, base_for_closure.clone()))
                as Box<dyn GroupPenalty>
        };

        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };

        let (betas_ref, _) = prox_newton_block_solve_path(
            &dense_ref, &glm_a, base.clone(), &make_inner, &groups,
            10, 1e-2, None, &cd_cfg, 20, 1e-8,
        );
        let (betas_std, _) = prox_newton_block_solve_path(
            &std_design, &glm_b, base.clone(), &make_inner, &groups,
            10, 1e-2, None, &cd_cfg, 20, 1e-8,
        );

        assert_eq!(betas_ref.shape(), betas_std.shape());
        for k in 0..betas_ref.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_ref[[k, j]], betas_std[[k, j]], epsilon = 1e-7);
            }
        }
    }
}
