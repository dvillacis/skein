//! λ-path solver for separable group penalties.
//!
//! Mirror of [`crate::solver::path`] but for group-structured problems.
//! Computes `block_lambda_max` (the smallest λ at which β = 0 is optimal
//! per the group KKT condition), builds a geometric grid via the shared
//! `lambda_grid`, and warm-starts [`block_cd_solve_subset`] across the
//! grid with a strong-rule + KKT-verification cycle on groups.
//!
//! `Screening::GapSafe` falls back to `Screening::Strong` for now —
//! block-level gap-safe screening is a separate follow-up.
//!
//! Currently assumes LS-style scaling (`∂_j L = X_jᵀ r / n`) just like the
//! scalar path solver; will become datafit-agnostic when M3 lands.

use crate::datafit::Datafit;
use crate::design::DesignMatrix;
use crate::groups::Groups;
use crate::penalty::GroupPenalty;
use crate::solver::block_cd::{
    block_cd_solve_subset_parallel_with_cache, block_cd_solve_subset_with_cache,
    block_find_kkt_violators, block_gap_safe_screen, block_strong_rule_screen,
    group_lipschitz_cache,
};
use crate::solver::cd::{CdConfig, CdReport};
use crate::solver::path::{lambda_grid, Screening};
use ndarray::{Array1, Array2, ArrayView1};

/// Smallest λ at which β = 0 is optimal under a separable convex group
/// penalty (group lasso, large-γ group MCP/SCAD).
///
/// Formula: `max_{g: w_g > 0} ‖X_g^T y‖₂ / (n · w_g)`. Computed at β = 0
/// where the residual is `-y`. Reduces exactly to the scalar `lambda_max`
/// when groups are singletons.
pub fn block_lambda_max(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    weights: ArrayView1<f64>,
    groups: &Groups,
) -> f64 {
    let p = design.n_features();
    let zero_beta = Array1::<f64>::zeros(p);
    let r0 = datafit.init_residual(design, zero_beta.view());

    let mut max_g = 0.0_f64;
    for g in 0..groups.n_groups() {
        let w = weights[g];
        if w <= 0.0 {
            continue;
        }
        let cols = groups.group(g);
        let group_grad_norm: f64 = cols
            .iter()
            .map(|&j| {
                let coord = datafit.coord_grad(design, j, r0.view());
                coord * coord
            })
            .sum::<f64>()
            .sqrt();
        let candidate = group_grad_norm / w;
        if candidate > max_g {
            max_g = candidate;
        }
    }
    max_g
}

#[derive(Debug, Clone)]
pub struct BlockPathConfig {
    pub n_lambdas: usize,
    pub lambda_min_ratio: f64,
    pub lambdas: Option<Vec<f64>>,
    pub cd: CdConfig,
    /// `Off` and `Strong` are honored; `GapSafe` silently falls back to
    /// `Strong` until block gap-safe screening is implemented.
    pub screening: Screening,
    /// When `true`, dispatch the per-λ inner CD via Rayon. Each sweep is
    /// Jacobi-style (groups compute against the snapshot residual). The
    /// per-group Lipschitz `‖X_g‖_F²/n` is correct for serial Gauss-Seidel;
    /// for Jacobi it's correct when off-diagonal `X_gᵀ X_{g'}` coupling is
    /// small. Highly correlated groups across blocks may oscillate — fall
    /// back to `false` in that case.
    pub parallel: bool,
}

impl Default for BlockPathConfig {
    fn default() -> Self {
        Self {
            n_lambdas: 100,
            lambda_min_ratio: 1e-3,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::default(),
            parallel: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockPathReport {
    pub lambdas: Vec<f64>,
    pub iters: Vec<usize>,
    pub converged: Vec<bool>,
    pub final_objs: Vec<f64>,
    /// Final group working-set size at each λ (post KKT-verification loop).
    pub working_set_sizes: Vec<usize>,
    pub kkt_passes: Vec<usize>,
}

/// Solve a separable group-penalty problem along a λ-path with warm
/// starts. Returns coefficients of shape `(n_lambdas, n_features)`; row
/// `k` is the solution at `report.lambdas[k]` (decreasing in λ).
pub fn solve_block_path<F>(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    make_penalty: F,
    groups: &Groups,
    config: &BlockPathConfig,
) -> (Array2<f64>, BlockPathReport)
where
    F: Fn(f64) -> Box<dyn GroupPenalty>,
{
    let p = design.n_features();
    let n_groups = groups.n_groups();

    let lambdas = match &config.lambdas {
        Some(v) => v.clone(),
        None => {
            // Group weights are λ-independent; sample at any λ to read them.
            let sample = make_penalty(1.0);
            let lam_max = block_lambda_max(design, datafit, sample.weights(), groups);
            lambda_grid(lam_max, config.n_lambdas, config.lambda_min_ratio)
        }
    };

    let n_lams = lambdas.len();
    let mut betas = Array2::<f64>::zeros((n_lams, p));
    let mut iters = Vec::with_capacity(n_lams);
    let mut converged = Vec::with_capacity(n_lams);
    let mut final_objs = Vec::with_capacity(n_lams);
    let mut working_set_sizes = Vec::with_capacity(n_lams);
    let mut kkt_passes_out = Vec::with_capacity(n_lams);

    // Per-group operator-norm Lipschitz, computed once for the whole path
    // and shared between the gap-safe screen and the inner CD.
    let group_lip = group_lipschitz_cache(design, groups);

    let mut warm = Array1::<f64>::zeros(p);
    let mut prev_residual: Option<Array1<f64>> = None;
    let mut prev_lambda: Option<f64> = None;

    for (k, &lam) in lambdas.iter().enumerate() {
        let pen = make_penalty(lam);
        let weights: Array1<f64> = pen.weights().to_owned();

        // Initial group working set per screening strategy.
        let mut ws: Vec<usize> = match config.screening {
            Screening::Off => (0..n_groups).collect(),
            Screening::Strong => {
                // Loop invariant: every iteration before returning sets
                // `prev_lambda = Some(lam)` and `prev_residual = Some(...)`,
                // so `k > 0 ⇒ both are Some`. The `k == 0` short-circuit
                // ensures we never evaluate the unwraps in iteration 0.
                if k == 0 || lam >= prev_lambda.expect("prev_lambda is Some when k > 0") {
                    (0..n_groups).collect()
                } else {
                    block_strong_rule_screen(
                        design,
                        datafit,
                        prev_residual
                            .as_ref()
                            .expect("prev_residual is Some when k > 0")
                            .view(),
                        weights.view(),
                        warm.view(),
                        groups,
                        lam,
                        prev_lambda.expect("prev_lambda is Some when k > 0"),
                    )
                }
            }
            Screening::GapSafe => {
                // Gap-safe works at every λ including the first (uses the
                // cold-start residual = −y when k = 0).
                let res_view = if k == 0 {
                    let r = datafit.init_residual(design, warm.view());
                    prev_residual = Some(r);
                    prev_residual
                        .as_ref()
                        .expect("just set prev_residual = Some(r)")
                        .view()
                } else {
                    prev_residual
                        .as_ref()
                        .expect("loop invariant: prev_residual is Some when k > 0")
                        .view()
                };
                block_gap_safe_screen(
                    design,
                    datafit,
                    res_view,
                    warm.view(),
                    weights.view(),
                    groups,
                    lam,
                    &group_lip,
                )
            }
        };

        let kkt_tol = lam.max(1e-12) * 1e-6;
        let mut passes = 0usize;

        let (final_residual, last_report): (Array1<f64>, CdReport) = loop {
            passes += 1;
            let (new_beta, report) = if config.parallel {
                block_cd_solve_subset_parallel_with_cache(
                    warm, &ws, &group_lip, design, datafit, &*pen, groups, &config.cd,
                )
            } else {
                block_cd_solve_subset_with_cache(
                    warm, &ws, &group_lip, design, datafit, &*pen, groups, &config.cd,
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

        betas.row_mut(k).assign(&warm);
        iters.push(last_report.iter);
        converged.push(last_report.converged);
        final_objs.push(last_report.final_obj);
        working_set_sizes.push(ws.len());
        kkt_passes_out.push(passes);

        prev_residual = Some(final_residual);
        prev_lambda = Some(lam);
    }

    (
        betas,
        BlockPathReport {
            lambdas,
            iters,
            converged,
            final_objs,
            working_set_sizes,
            kkt_passes: kkt_passes_out,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::DenseMatrix;
    use crate::penalty::{GroupLasso, Mcp};
    use crate::solver::cd::cd_solve;
    use crate::solver::path::lambda_max;
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

    // ---- block_lambda_max -----------------------------------------------

    #[test]
    fn block_lambda_max_matches_max_group_correlation() {
        let (design, y, groups) = sparse_group_problem(101);
        let datafit = LeastSquares::new(y.clone());
        let weights = Array1::<f64>::ones(groups.n_groups());
        let lam = block_lambda_max(&design, &datafit, weights.view(), &groups);

        let n = design.n_samples() as f64;
        let mut expected = 0.0_f64;
        for g in 0..groups.n_groups() {
            let cols = groups.group(g);
            let norm_sq: f64 = cols
                .iter()
                .map(|&j| {
                    let coord = design.col_dot(j, y.view());
                    coord * coord
                })
                .sum();
            let candidate = norm_sq.sqrt() / n;
            if candidate > expected {
                expected = candidate;
            }
        }
        assert_abs_diff_eq!(lam, expected, epsilon = 1e-12);
    }

    #[test]
    fn block_lambda_max_with_singleton_groups_matches_scalar_lambda_max() {
        let (design, y, _) = sparse_group_problem(102);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let weights = Array1::<f64>::ones(p);
        let groups = Groups::singletons(p);
        let block = block_lambda_max(&design, &datafit, weights.view(), &groups);
        let scalar = lambda_max(&design, &datafit, weights.view());
        assert_abs_diff_eq!(block, scalar, epsilon = 1e-12);
    }

    // ---- solve_block_path basic shape & boundary ------------------------

    #[test]
    fn block_path_lambdas_decreasing_with_correct_shape() {
        let (design, y, groups) = sparse_group_problem(103);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let cfg = BlockPathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Strong,
            parallel: false,
        };
        let (betas, report) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &cfg,
        );
        assert_eq!(betas.shape(), &[8, p]);
        assert_eq!(report.lambdas.len(), 8);
        for k in 1..report.lambdas.len() {
            assert!(report.lambdas[k] < report.lambdas[k - 1]);
        }
    }

    #[test]
    fn block_path_at_lambda_max_returns_zero() {
        let (design, y, groups) = sparse_group_problem(104);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let weights = Array1::<f64>::ones(groups.n_groups());
        let lam_max = block_lambda_max(&design, &datafit, weights.view(), &groups);

        let cfg = BlockPathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![lam_max]),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            parallel: false,
        };
        let (betas, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &cfg,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn block_path_with_explicit_lambdas_honored() {
        let (design, y, groups) = sparse_group_problem(105);
        let datafit = LeastSquares::new(y);
        let custom = vec![1.0, 0.5, 0.25, 0.1];
        let cfg = BlockPathConfig {
            n_lambdas: 0,
            lambda_min_ratio: 0.0,
            lambdas: Some(custom.clone()),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            parallel: false,
        };
        let (betas, report) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &cfg,
        );
        assert_eq!(report.lambdas, custom);
        assert_eq!(betas.shape(), &[4, design.n_features()]);
    }

    #[test]
    fn block_path_recovers_sparse_group_truth_at_small_lambda() {
        let (design, y, groups) = sparse_group_problem(106);
        let datafit = LeastSquares::new(y);

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
        let (betas, report) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &cfg,
        );
        // At smallest λ, the active groups (0 and 2) should clearly
        // dominate the inactive ones.
        let last = report.lambdas.len() - 1;
        let last_beta = betas.row(last).to_owned();
        assert!(group_norm(&last_beta, &groups, 0) > 0.5);
        assert!(group_norm(&last_beta, &groups, 2) > 0.5);
    }

    // ---- screening dispatch ---------------------------------------------

    #[test]
    fn block_path_screening_off_uses_full_working_set() {
        let (design, y, groups) = sparse_group_problem(107);
        let datafit = LeastSquares::new(y);
        let cfg = BlockPathConfig {
            n_lambdas: 5,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Off,
            parallel: false,
        };
        let (_, report) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &cfg,
        );
        for &ws in &report.working_set_sizes {
            assert_eq!(ws, groups.n_groups());
        }
        for &kk in &report.kkt_passes {
            assert_eq!(kk, 1);
        }
    }

    #[test]
    fn block_path_screening_on_matches_screening_off_within_tol() {
        let (design, y, groups) = sparse_group_problem(108);
        let datafit = LeastSquares::new(y);
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let mk_cfg = |s: Screening| BlockPathConfig {
            n_lambdas: 10,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: cd_cfg.clone(),
            screening: s,
            parallel: false,
        };
        let (b_off, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &mk_cfg(Screening::Off),
        );
        let (b_on, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &mk_cfg(Screening::Strong),
        );
        assert_eq!(b_off.shape(), b_on.shape());
        for k in 0..b_off.nrows() {
            for j in 0..b_off.ncols() {
                assert_abs_diff_eq!(b_off[[k, j]], b_on[[k, j]], epsilon = 1e-6);
            }
        }
    }

    // ---- equivalence with scalar path on singleton groups ---------------

    #[test]
    fn block_path_with_singleton_groups_matches_scalar_solve_path_on_lasso() {
        // When every group is a singleton, the group path on GroupLasso
        // should produce the same β as the scalar path on Mcp at γ→∞.
        // Smoke-checks that block_lambda_max + the path pipeline reduce
        // to the M1 case correctly.
        let (design, y, _) = sparse_group_problem(109);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let groups = Groups::singletons(p);
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let cfg = BlockPathConfig {
            n_lambdas: 6,
            lambda_min_ratio: 5e-2,
            lambdas: None,
            cd: cd_cfg.clone(),
            screening: Screening::Strong,
            parallel: false,
        };
        let (b_block, report) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, p)),
            &groups,
            &cfg,
        );
        // Solve the same λ list scalar-style with γ→∞ MCP (≈ lasso).
        // Use cd_solve per λ (cold start) for an apples-to-apples check.
        for k in 0..report.lambdas.len() {
            let lam = report.lambdas[k];
            let (cold, _) = cd_solve(&design, &datafit, &Mcp::new(lam, 1e10, p), &cd_cfg);
            for j in 0..p {
                assert_abs_diff_eq!(b_block[[k, j]], cold[j], epsilon = 1e-5);
            }
        }
    }

    // ---- parallel path -------------------------------------------------

    #[test]
    fn block_path_parallel_matches_serial_within_tol() {
        // Convex group lasso, well-conditioned random columns: Jacobi
        // sweeps must converge to the same β at every λ as the Gauss-Seidel
        // serial path (given enough inner iterations).
        let (design, y, groups) = sparse_group_problem(110);
        let datafit = LeastSquares::new(y);
        let cd_cfg = CdConfig {
            max_iter: 10_000,
            tol: 1e-12,
            acceleration: None,
        };
        let mk_cfg = |parallel: bool| BlockPathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: cd_cfg.clone(),
            screening: Screening::Strong,
            parallel,
        };
        let (b_serial, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &mk_cfg(false),
        );
        let (b_parallel, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &mk_cfg(true),
        );
        for k in 0..b_serial.nrows() {
            for j in 0..b_serial.ncols() {
                assert_abs_diff_eq!(b_serial[[k, j]], b_parallel[[k, j]], epsilon = 1e-5);
            }
        }
    }

    // ---- block gap-safe screening ---------------------------------------

    #[test]
    fn block_path_gap_safe_matches_strong_rule_within_tol_on_group_lasso() {
        // Both screening modes must converge to the same β at every λ on
        // a convex group-lasso problem.
        let (design, y, groups) = sparse_group_problem(150);
        let datafit = LeastSquares::new(y);
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let mk_cfg = |s: Screening| BlockPathConfig {
            n_lambdas: 10,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: cd_cfg.clone(),
            screening: s,
            parallel: false,
        };
        let (b_strong, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &mk_cfg(Screening::Strong),
        );
        let (b_gap, _) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &mk_cfg(Screening::GapSafe),
        );
        for k in 0..b_strong.nrows() {
            for j in 0..b_strong.ncols() {
                assert_abs_diff_eq!(b_strong[[k, j]], b_gap[[k, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn block_path_gap_safe_drops_inactive_groups_on_sparse_group_truth() {
        // Sparse-group truth (active: groups 0, 2). Mid-path with gap-safe
        // should screen most inactive groups.
        let (design, y, groups) = sparse_group_problem(151);
        let datafit = LeastSquares::new(y);
        let cfg = BlockPathConfig {
            n_lambdas: 15,
            lambda_min_ratio: 5e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            screening: Screening::GapSafe,
            parallel: false,
        };
        let (_, report) = solve_block_path(
            &design,
            &datafit,
            |lam| Box::new(GroupLasso::new(lam, groups.n_groups())),
            &groups,
            &cfg,
        );
        let mid = report.working_set_sizes.len() / 2;
        let mid_ws = report.working_set_sizes[mid];
        let n_groups = groups.n_groups();
        assert!(
            mid_ws < n_groups,
            "gap-safe ws at mid-path should be < n_groups = {} (got {})",
            n_groups,
            mid_ws,
        );
    }
}
