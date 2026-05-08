//! LLA-wrapped λ-path solver for non-convex group penalties.
//!
//! Per λ, runs the M2.3 LLA outer loop with the inner being a weighted
//! convex group lasso solved via M2.4's working-set + KKT cycle on the
//! current iterate's surrogate weights. β warm-starts across the path
//! exactly like [`solve_block_path`]; the strong rule + KKT verifier use
//! the *surrogate* weights of each LLA outer iteration.
//!
//! The user supplies a `make_inner` closure that builds the surrogate
//! convex group penalty from the current iterate plus the current λ —
//! typically a `GroupLasso::with_weights(...)` (group MCP/SCAD) or
//! `SparseGroupLasso::with_coord_weights(...)` (sparse-group MCP/SCAD).
//! Pair with `surrogate_weights_group_mcp` or `surrogate_sparse_group_mcp`
//! to compute the surrogate weights inside the closure.

use crate::datafit::Datafit;
use crate::design::DesignMatrix;
use crate::groups::Groups;
use crate::penalty::GroupPenalty;
use crate::solver::block_cd::{
    block_cd_solve_subset_parallel_with_cache, block_cd_solve_subset_with_cache,
    block_find_kkt_violators, block_strong_rule_screen, group_lipschitz_cache,
};
use crate::solver::block_path::{block_lambda_max, BlockPathConfig};
use crate::solver::cd::CdReport;
use crate::solver::path::{lambda_grid, Screening};
use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone)]
pub struct BlockPathLLAReport {
    pub lambdas: Vec<f64>,
    /// LLA outer iterations performed at each λ.
    pub outer_iters: Vec<usize>,
    /// Whether each λ's outer loop hit `outer_tol` (otherwise it ran to
    /// `max_outer`).
    pub outer_converged: Vec<bool>,
    /// Sum of CD inner iterations across all LLA outer iters at each λ.
    pub inner_iters: Vec<usize>,
    pub final_objs: Vec<f64>,
    /// Final group working-set size at each λ (post final KKT cycle).
    pub working_set_sizes: Vec<usize>,
    /// Total KKT-loop passes (summed across LLA outer iters) at each λ.
    pub kkt_passes: Vec<usize>,
}

/// Solve a non-convex group-penalty problem along a λ-path with warm
/// starts, using LLA at every λ.
///
/// `make_inner` is called once per LLA outer iteration with the current
/// `(β, groups, λ)` and must return the surrogate convex group penalty —
/// typically a `GroupLasso::with_weights(...)` or
/// `SparseGroupLasso::with_coord_weights(...)` whose weights are computed
/// from `β` via a helper like [`crate::solver::surrogate_weights_group_mcp`]
/// or [`crate::solver::surrogate_sparse_group_mcp`]. The strong rule reads
/// per-group L2 weights from `inner.weights()`.
#[allow(clippy::too_many_arguments)]
pub fn solve_block_path_lla<F>(
    design: &(dyn DesignMatrix + Sync),
    datafit: &(dyn Datafit + Sync),
    base_weights: Array1<f64>,
    make_inner: F,
    groups: &Groups,
    config: &BlockPathConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array2<f64>, BlockPathLLAReport)
where
    F: Fn(ArrayView1<f64>, &Groups, f64) -> Box<dyn GroupPenalty>,
{
    let p = design.n_features();
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_weights.len(), n_groups);

    let lambdas = match &config.lambdas {
        Some(v) => v.clone(),
        None => {
            let lam_max = block_lambda_max(design, datafit, base_weights.view(), groups);
            lambda_grid(lam_max, config.n_lambdas, config.lambda_min_ratio)
        }
    };

    let n_lams = lambdas.len();
    let mut betas = Array2::<f64>::zeros((n_lams, p));
    let mut outer_iters_out = Vec::with_capacity(n_lams);
    let mut outer_converged_out = Vec::with_capacity(n_lams);
    let mut inner_iters_out = Vec::with_capacity(n_lams);
    let mut final_objs_out = Vec::with_capacity(n_lams);
    let mut working_set_sizes_out = Vec::with_capacity(n_lams);
    let mut kkt_passes_out = Vec::with_capacity(n_lams);

    // Per-group operator-norm Lipschitz, computed once for the whole path.
    let group_lip = group_lipschitz_cache(design, groups);

    let mut warm = Array1::<f64>::zeros(p);
    let mut prev_residual: Option<Array1<f64>> = None;
    let mut prev_lambda: Option<f64> = None;

    for (k, &lam) in lambdas.iter().enumerate() {
        let mut last_inner_iters_total = 0usize;
        let mut last_kkt_passes_total = 0usize;
        let mut last_ws_size = n_groups;
        let mut last_inner_obj = 0.0;
        let mut last_residual: Array1<f64> = prev_residual
            .clone()
            .unwrap_or_else(|| datafit.init_residual(design, warm.view()));
        let mut outer_converged = false;
        let mut outer_iters_done = 0usize;

        for outer in 0..max_outer {
            outer_iters_done = outer + 1;
            let inner_pen = make_inner(warm.view(), groups, lam);
            // The strong rule and KKT verifier need a per-group weight
            // vector — for SGL inner this is the L2 weights. Owned copy
            // since `inner_pen.weights()` borrows from `inner_pen` which
            // we want to keep alive across the loop.
            let weights: Array1<f64> = inner_pen.weights().to_owned();

            // Strong-rule screen against the previous λ's residual + the
            // CURRENT surrogate weights. For the first λ, no prev residual
            // ⇒ full WS.
            let mut ws: Vec<usize> = match config.screening {
                Screening::Off => (0..n_groups).collect(),
                Screening::Strong | Screening::GapSafe => {
                    if k == 0 || lam >= prev_lambda.unwrap() {
                        (0..n_groups).collect()
                    } else {
                        block_strong_rule_screen(
                            design,
                            datafit,
                            prev_residual.as_ref().unwrap().view(),
                            weights.view(),
                            warm.view(),
                            groups,
                            lam,
                            prev_lambda.unwrap(),
                        )
                    }
                }
            };

            let kkt_tol = lam.max(1e-12) * 1e-6;
            let mut passes = 0usize;
            let beta_pre_outer = warm.clone();

            let (final_residual, last_report): (Array1<f64>, CdReport) = loop {
                passes += 1;
                let (new_beta, report) = if config.parallel {
                    block_cd_solve_subset_parallel_with_cache(
                        warm,
                        &ws,
                        &group_lip,
                        design,
                        datafit,
                        &*inner_pen,
                        groups,
                        &config.cd,
                    )
                } else {
                    block_cd_solve_subset_with_cache(
                        warm,
                        &ws,
                        &group_lip,
                        design,
                        datafit,
                        &*inner_pen,
                        groups,
                        &config.cd,
                    )
                };
                warm = new_beta;
                let r = datafit.init_residual(design, warm.view());

                if matches!(config.screening, Screening::Off) {
                    break (r, report);
                }
                let violators = block_find_kkt_violators(
                    design,
                    datafit,
                    r.view(),
                    weights.view(),
                    &ws,
                    groups,
                    lam,
                    kkt_tol,
                );
                if violators.is_empty() {
                    break (r, report);
                }
                ws.extend(violators);
                ws.sort_unstable();
                ws.dedup();
            };

            last_inner_iters_total += last_report.iter;
            last_kkt_passes_total += passes;
            last_ws_size = ws.len();
            last_inner_obj = last_report.final_obj;
            last_residual = final_residual;

            // Outer convergence check (max block change between LLA outer iters).
            let mut max_block_change = 0.0_f64;
            for g in 0..n_groups {
                let mut sum_sq = 0.0_f64;
                for &j in groups.group(g) {
                    let d = warm[j] - beta_pre_outer[j];
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

        betas.row_mut(k).assign(&warm);
        outer_iters_out.push(outer_iters_done);
        outer_converged_out.push(outer_converged);
        inner_iters_out.push(last_inner_iters_total);
        final_objs_out.push(last_inner_obj);
        working_set_sizes_out.push(last_ws_size);
        kkt_passes_out.push(last_kkt_passes_total);

        prev_residual = Some(last_residual);
        prev_lambda = Some(lam);
    }

    (
        betas,
        BlockPathLLAReport {
            lambdas,
            outer_iters: outer_iters_out,
            outer_converged: outer_converged_out,
            inner_iters: inner_iters_out,
            final_objs: final_objs_out,
            working_set_sizes: working_set_sizes_out,
            kkt_passes: kkt_passes_out,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::DenseMatrix;
    use crate::penalty::{GroupLasso, SparseGroupLasso};
    use crate::solver::cd::CdConfig;
    use crate::solver::lla::{surrogate_sparse_group_mcp, surrogate_weights_group_mcp};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    fn sparse_group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Groups) {
        let n = 60;
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
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), y, groups)
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
    fn block_path_lla_at_lambda_max_returns_zero() {
        let (design, y, groups) = sparse_group_problem(120);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let base = Array1::<f64>::ones(groups.n_groups());
        let lam_max = block_lambda_max(&design, &datafit, base.view(), &groups);

        let cfg = BlockPathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![lam_max]),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            parallel: false,
        };
        let gamma = 100.0;
        let make_inner = |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base.view());
            Box::new(GroupLasso::with_weights(lam, w))
        };
        let (betas, _) = solve_block_path_lla(
            &design,
            &datafit,
            base.clone(),
            make_inner,
            &groups,
            &cfg,
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn block_path_lla_lambdas_decreasing_with_correct_shape() {
        let (design, y, groups) = sparse_group_problem(121);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let base = Array1::<f64>::ones(groups.n_groups());

        let cfg = BlockPathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Strong,
            parallel: false,
        };
        let gamma = 100.0;
        let make_inner = |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base.view());
            Box::new(GroupLasso::with_weights(lam, w))
        };
        let (betas, report) = solve_block_path_lla(
            &design,
            &datafit,
            base.clone(),
            make_inner,
            &groups,
            &cfg,
            10,
            1e-8,
        );
        assert_eq!(betas.shape(), &[8, p]);
        assert_eq!(report.lambdas.len(), 8);
        assert_eq!(report.outer_iters.len(), 8);
        for k in 1..report.lambdas.len() {
            assert!(report.lambdas[k] < report.lambdas[k - 1]);
        }
    }

    #[test]
    fn block_path_lla_recovers_sparse_group_truth_via_group_mcp() {
        // Truth: groups 0 and 2 active. LLA-wrapped path must recover
        // them at the small-λ end of the auto-path.
        let (design, y, groups) = sparse_group_problem(122);
        let datafit = LeastSquares::new(y);
        let base = Array1::<f64>::ones(groups.n_groups());

        let cfg = BlockPathConfig {
            n_lambdas: 25,
            lambda_min_ratio: 5e-3,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            screening: Screening::Strong,
            parallel: false,
        };
        let gamma = 3.0;
        let make_inner = |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base.view());
            Box::new(GroupLasso::with_weights(lam, w))
        };
        let (betas, report) = solve_block_path_lla(
            &design,
            &datafit,
            base.clone(),
            make_inner,
            &groups,
            &cfg,
            20,
            1e-8,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(
            group_norm(&last_beta, &groups, 0) > 0.5,
            "active group 0 should have norm > 0.5 at smallest λ, got {}",
            group_norm(&last_beta, &groups, 0)
        );
        assert!(
            group_norm(&last_beta, &groups, 2) > 0.5,
            "active group 2 should have norm > 0.5 at smallest λ, got {}",
            group_norm(&last_beta, &groups, 2)
        );
    }

    #[test]
    fn block_path_lla_with_sparse_group_mcp_recovers_sparse_in_group_truth() {
        // 4 features in 2 groups. Truth: only feature 0 is active.
        // Sparse-group MCP via LLA should:
        //   - Activate group 0 (because feature 0 is real signal).
        //   - Within group 0, *zero* feature 1 thanks to the L1 part of SGL
        //     and the unbiasedness of MCP (the L1 surrogate weights drop
        //     toward 0 for the active coord, sparing it from shrinkage).
        let n = 80;
        let p = 4;
        let mut state: u64 = 250;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let true_beta = ndarray::array![2.0, 0.0, 0.0, 0.0];
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let groups = Groups::contiguous_blocks(p, 2);

        let base_group = Array1::<f64>::ones(groups.n_groups());
        let base_coord = Array1::<f64>::ones(p);
        let cfg = BlockPathConfig {
            n_lambdas: 25,
            lambda_min_ratio: 5e-3,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            screening: Screening::Strong,
            parallel: false,
        };
        let gamma = 3.0;
        let alpha = 0.5;
        let make_inner = |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                base_group.view(),
                base_coord.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        };

        let (betas, report) = solve_block_path_lla(
            &design,
            &datafit,
            base_group.clone(),
            make_inner,
            &groups,
            &cfg,
            20,
            1e-8,
        );
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(
            last_beta[0].abs() > 0.5,
            "feature 0 (signal) should be active, got |β_0|={}",
            last_beta[0].abs()
        );
        assert!(
            last_beta[1].abs() < 0.3,
            "feature 1 should be zeroed by within-group L1, got |β_1|={}",
            last_beta[1].abs()
        );
    }
}
