//! Local Linear Approximation outer loop.
//!
//! Folds a non-convex group penalty (group MCP, group SCAD, …) into a
//! sequence of weighted convex group-lasso problems by linearizing the
//! penalty around the current iterate. Each outer iteration:
//!   1. Build per-group surrogate weights from `β_old`
//!   2. Solve weighted group lasso (via `block_cd_solve_subset`) → `β_new`
//!   3. Stop if max block-change `‖β_new_g − β_old_g‖₂` falls below `outer_tol`
//!
//! Typical convergence is 2–5 outer iterations in practice. Inner solver
//! warm starts from the previous outer iterate, so each successive inner
//! solve is cheaper than the last.

use crate::datafit::Datafit;
use crate::design::DesignMatrix;
use crate::groups::Groups;
use crate::penalty::GroupLasso;
use crate::solver::block_cd::block_cd_solve_subset;
use crate::solver::cd::CdConfig;
use ndarray::{Array1, ArrayView1};

#[derive(Debug, Clone)]
pub struct LLAReport {
    pub outer_iters: usize,
    pub converged: bool,
    /// CD inner-iteration counts per outer iteration.
    pub inner_iters: Vec<usize>,
    /// Whether each inner CD call hit its own convergence tolerance.
    pub inner_converged: Vec<bool>,
}

/// LLA outer loop. Caller supplies `update_weights(β, groups) → w` that
/// computes per-group surrogate weights from the current iterate; the
/// outer loop wraps an inner weighted group-lasso solve. `lambda` is the
/// outer-problem regularizer used to scale the inner penalty.
#[allow(clippy::too_many_arguments)]
pub fn lla_solve<F>(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    groups: &Groups,
    init_beta: Array1<f64>,
    lambda: f64,
    update_weights: F,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array1<f64>, LLAReport)
where
    F: Fn(ArrayView1<f64>, &Groups) -> Array1<f64>,
{
    let p = design.n_features();
    let n_groups = groups.n_groups();
    debug_assert_eq!(init_beta.len(), p, "init_beta length must equal n_features");

    let group_subset: Vec<usize> = (0..n_groups).collect();
    let mut beta = init_beta;
    let mut inner_iters = Vec::with_capacity(max_outer);
    let mut inner_converged = Vec::with_capacity(max_outer);

    let mut report = LLAReport {
        outer_iters: 0,
        converged: false,
        inner_iters: Vec::new(),
        inner_converged: Vec::new(),
    };

    for outer in 0..max_outer {
        let weights = update_weights(beta.view(), groups);
        debug_assert_eq!(weights.len(), n_groups, "surrogate weights length must equal n_groups");
        let inner_pen = GroupLasso::with_weights(lambda, weights);

        let beta_old = beta.clone();
        let (new_beta, inner_report) = block_cd_solve_subset(
            beta,
            &group_subset,
            design,
            datafit,
            &inner_pen,
            groups,
            cd_config,
        );
        beta = new_beta;
        inner_iters.push(inner_report.iter);
        inner_converged.push(inner_report.converged);

        // Outer convergence: max L₂ block change across all groups.
        let mut max_block_change = 0.0_f64;
        for g in 0..n_groups {
            let mut sum_sq = 0.0_f64;
            for &j in groups.group(g) {
                let d = beta[j] - beta_old[j];
                sum_sq += d * d;
            }
            let block_change = sum_sq.sqrt();
            if block_change > max_block_change {
                max_block_change = block_change;
            }
        }

        report.outer_iters = outer + 1;
        if max_block_change < outer_tol {
            report.converged = true;
            break;
        }
    }
    report.inner_iters = inner_iters;
    report.inner_converged = inner_converged;

    (beta, report)
}

/// LLA surrogate weights for sparse-group MCP, mixing parameter `alpha`.
///
/// The original penalty per group `g` is
/// `MCP(‖β_g‖₂; λ(1−α)·w_g, γ) + Σ_{k∈g} MCP(|β_{g,k}|; λα·v_{g,k}, γ)`.
///
/// LLA at the current iterate produces a weighted SGL inner penalty:
///   - per-group L2 weight: `w_g' = max(0, w_g − ‖β_g‖₂ / ((1−α)·λ·γ))`
///   - per-coord L1 weight: `v_{g,k}' = max(0, v_{g,k} − |β_{g,k}| / (α·λ·γ))`
///
/// Returns `(group_weights, coord_weights_per_group)` ready to feed into
/// [`crate::penalty::SparseGroupLasso::with_coord_weights`]. Edge cases:
///   - `α = 0` (pure group MCP): per-coord weights are returned as zeros
///     (the L1 part vanishes; weights are irrelevant).
///   - `α = 1` (pure scalar MCP per coord): per-group weights are zeros
///     (the L2 part vanishes).
pub fn surrogate_sparse_group_mcp(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    gamma: f64,
    alpha: f64,
    base_group: ArrayView1<f64>,
    base_coord: ArrayView1<f64>,
) -> (Array1<f64>, Vec<Array1<f64>>) {
    assert!(
        (0.0..=1.0).contains(&alpha),
        "alpha must be in [0, 1] (got {})",
        alpha
    );
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_group.len(), n_groups);
    debug_assert_eq!(base_coord.len(), beta.len());

    let mut group_w = Array1::<f64>::zeros(n_groups);
    let mut coord_w: Vec<Array1<f64>> = Vec::with_capacity(n_groups);

    let group_denom = if alpha < 1.0 {
        Some((1.0 - alpha) * lambda * gamma)
    } else {
        None
    };
    let coord_denom = if alpha > 0.0 {
        Some(alpha * lambda * gamma)
    } else {
        None
    };

    for g in 0..n_groups {
        let cols = groups.group(g);
        let block_norm: f64 = cols
            .iter()
            .map(|&j| beta[j] * beta[j])
            .sum::<f64>()
            .sqrt();
        group_w[g] = match group_denom {
            Some(d) => (base_group[g] - block_norm / d).max(0.0),
            None => 0.0,
        };
        let mut cw_g = Array1::<f64>::zeros(cols.len());
        for (k, &j) in cols.iter().enumerate() {
            cw_g[k] = match coord_denom {
                Some(d) => (base_coord[j] - beta[j].abs() / d).max(0.0),
                None => 0.0,
            };
        }
        coord_w.push(cw_g);
    }
    (group_w, coord_w)
}

/// SCAD's LLA shrinkage factor: returns `f` such that `w_lla = base · f`,
/// equivalently `SCAD'(t; λ_eff, a) = λ_eff · f`. Piecewise:
///   - `t ≤ λ_eff`            : 1   (base weight unchanged)
///   - `λ_eff < t ≤ a·λ_eff`  : `(a − t/λ_eff) / (a − 1)`   (linearly decays)
///   - `t > a·λ_eff`          : 0   (saturated)
fn scad_lla_factor(t: f64, lambda_eff: f64, a: f64) -> f64 {
    if t <= lambda_eff {
        1.0
    } else if t <= a * lambda_eff {
        (a - t / lambda_eff) / (a - 1.0)
    } else {
        0.0
    }
}

/// LLA surrogate weights for **group SCAD** with shape `a > 2`. Mirrors
/// [`surrogate_weights_group_mcp`] but uses SCAD's piecewise-linear
/// derivative. At `β = 0` returns the base weights; in the linear-decay
/// region returns `(a·w_base − ‖β_g‖/λ) / (a − 1)`; for saturated groups
/// (`‖β_g‖ > a·λ·w_base`) returns 0.
pub fn surrogate_weights_group_scad(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    a: f64,
    base_weights: ArrayView1<f64>,
) -> Array1<f64> {
    assert!(a > 2.0, "SCAD shape parameter `a` must be > 2 (got {})", a);
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_weights.len(), n_groups);
    Array1::from_iter((0..n_groups).map(|g| {
        let norm: f64 = groups
            .group(g)
            .iter()
            .map(|&j| beta[j] * beta[j])
            .sum::<f64>()
            .sqrt();
        let lam_eff = lambda * base_weights[g];
        if lam_eff <= 0.0 {
            return 0.0;
        }
        base_weights[g] * scad_lla_factor(norm, lam_eff, a)
    }))
}

/// LLA surrogate weights for **sparse-group SCAD**, mixing parameter `α`.
///
/// Returns `(group_weights, coord_weights_per_group)` ready to feed into
/// [`crate::penalty::SparseGroupLasso::with_coord_weights`]. Same edge-case
/// handling as [`surrogate_sparse_group_mcp`] (`α = 0` zeros L1 weights;
/// `α = 1` zeros L2 weights).
pub fn surrogate_sparse_group_scad(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    a: f64,
    alpha: f64,
    base_group: ArrayView1<f64>,
    base_coord: ArrayView1<f64>,
) -> (Array1<f64>, Vec<Array1<f64>>) {
    assert!(a > 2.0, "SCAD shape parameter `a` must be > 2 (got {})", a);
    assert!(
        (0.0..=1.0).contains(&alpha),
        "alpha must be in [0, 1] (got {})",
        alpha
    );
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_group.len(), n_groups);
    debug_assert_eq!(base_coord.len(), beta.len());

    let mut group_w = Array1::<f64>::zeros(n_groups);
    let mut coord_w: Vec<Array1<f64>> = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let cols = groups.group(g);
        let block_norm: f64 = cols
            .iter()
            .map(|&j| beta[j] * beta[j])
            .sum::<f64>()
            .sqrt();
        // L2 surrogate
        if alpha < 1.0 {
            let lam_eff = lambda * (1.0 - alpha) * base_group[g];
            if lam_eff > 0.0 {
                group_w[g] = base_group[g] * scad_lla_factor(block_norm, lam_eff, a);
            }
        }
        // Per-coord L1 surrogates
        let mut cw_g = Array1::<f64>::zeros(cols.len());
        if alpha > 0.0 {
            for (k, &j) in cols.iter().enumerate() {
                let lam_eff = lambda * alpha * base_coord[j];
                if lam_eff > 0.0 {
                    cw_g[k] = base_coord[j] * scad_lla_factor(beta[j].abs(), lam_eff, a);
                }
            }
        }
        coord_w.push(cw_g);
    }
    (group_w, coord_w)
}

/// Per-group surrogate weights for group MCP:
///   `w_g_lla = max(0, w_g_base − ‖β_g‖₂ / (λ · γ))`.
///
/// Equals `w_g_base` when `β_g = 0`, decreases linearly with `‖β_g‖`,
/// and clamps to 0 once the group enters the saturated regime
/// `‖β_g‖ ≥ λγ · w_g_base`. Pass into `lla_solve` as the `update_weights`
/// closure.
pub fn surrogate_weights_group_mcp(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    gamma: f64,
    base_weights: ArrayView1<f64>,
) -> Array1<f64> {
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_weights.len(), n_groups);
    let denom = lambda * gamma;
    Array1::from_iter((0..n_groups).map(|g| {
        let norm_sq: f64 = groups
            .group(g)
            .iter()
            .map(|&j| beta[j] * beta[j])
            .sum();
        let norm = norm_sq.sqrt();
        (base_weights[g] - norm / denom).max(0.0)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

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

    // ---- surrogate-weight helper ----------------------------------------

    #[test]
    fn surrogate_weights_group_mcp_at_zero_beta_returns_base_weights() {
        let beta = Array1::<f64>::zeros(4);
        let groups = Groups::contiguous_blocks(4, 2);
        let base = array![1.5, 0.7];
        let w = surrogate_weights_group_mcp(beta.view(), &groups, 0.1, 3.0, base.view());
        for g in 0..2 {
            assert_abs_diff_eq!(w[g], base[g], epsilon = 1e-12);
        }
    }

    #[test]
    fn surrogate_weights_group_mcp_zeros_saturated_group_keeps_small_one() {
        let lambda = 0.1;
        let gamma = 3.0;
        let base = array![1.0, 1.0];
        // Group 0: norm = 1.0 ≥ λγ·w = 0.3 ⇒ saturated ⇒ w_lla = 0.
        // Group 1: norm ≈ 0.0707, w_lla ≈ 1.0 − 0.0707/0.3 ≈ 0.764.
        let beta = array![0.6, 0.8, 0.05, 0.05];
        let groups = Groups::contiguous_blocks(4, 2);
        let w = surrogate_weights_group_mcp(beta.view(), &groups, lambda, gamma, base.view());
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-12);
        assert!(
            w[1] > 0.5 && w[1] < 0.9,
            "expected 0.5 < w[1] < 0.9, got {}",
            w[1]
        );
    }

    // ---- LLA outer loop -------------------------------------------------

    #[test]
    fn lla_zeros_all_groups_under_strong_lambda() {
        let (design, y, groups) = sparse_group_problem(1);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let lambda = 100.0;
        let gamma = 100.0;
        let base = Array1::<f64>::ones(groups.n_groups());

        let update = |beta: ArrayView1<f64>, g: &Groups| {
            surrogate_weights_group_mcp(beta, g, lambda, gamma, base.view())
        };

        let (beta, _) = lla_solve(
            &design,
            &datafit,
            &groups,
            Array1::<f64>::zeros(p),
            lambda,
            update,
            &CdConfig {
                max_iter: 200,
                tol: 1e-8,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta[j], 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn lla_recovers_sparse_group_truth_via_group_mcp() {
        let (design, y, groups) = sparse_group_problem(2);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let lambda = 0.005;
        let gamma = 3.0;
        let base = Array1::<f64>::ones(groups.n_groups());

        let update = |beta: ArrayView1<f64>, g: &Groups| {
            surrogate_weights_group_mcp(beta, g, lambda, gamma, base.view())
        };

        let (beta, report) = lla_solve(
            &design,
            &datafit,
            &groups,
            Array1::<f64>::zeros(p),
            lambda,
            update,
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            20,
            1e-8,
        );
        assert!(
            report.converged,
            "LLA should converge in ≤ 20 outer iterations (got {})",
            report.outer_iters
        );
        // Truth: groups 0 (features 0, 1) and 2 (features 4, 5) are active.
        assert!(group_norm(&beta, &groups, 0) > 0.5);
        assert!(group_norm(&beta, &groups, 2) > 0.5);
    }

    // ---- sparse-group MCP surrogate weights ------------------------------

    #[test]
    fn surrogate_sparse_group_mcp_at_zero_beta_returns_base_weights() {
        // β = 0 ⇒ both L1 and L2 surrogate weights equal their base.
        let p = 4;
        let groups = Groups::contiguous_blocks(p, 2);
        let beta = Array1::<f64>::zeros(p);
        let base_group = array![1.5, 0.7];
        let base_coord = array![2.0, 1.0, 3.0, 0.5];
        let alpha = 0.4;
        let (gw, cw) = surrogate_sparse_group_mcp(
            beta.view(),
            &groups,
            0.1,
            3.0,
            alpha,
            base_group.view(),
            base_coord.view(),
        );
        for g in 0..2 {
            assert_abs_diff_eq!(gw[g], base_group[g], epsilon = 1e-12);
        }
        assert_abs_diff_eq!(cw[0][0], base_coord[0], epsilon = 1e-12);
        assert_abs_diff_eq!(cw[0][1], base_coord[1], epsilon = 1e-12);
        assert_abs_diff_eq!(cw[1][0], base_coord[2], epsilon = 1e-12);
        assert_abs_diff_eq!(cw[1][1], base_coord[3], epsilon = 1e-12);
    }

    #[test]
    fn surrogate_sparse_group_mcp_zeros_saturated_components() {
        // λ=0.1, γ=3, α=0.5. Group 0 has ‖β‖=√2 ≈ 1.414. L2 saturation
        // threshold = (1−α)·λ·γ·base_group = 0.5·0.1·3·1 = 0.15. Norm
        // 1.414 ≫ 0.15 ⇒ group L2 weight = 0.
        // Coord 0 has |β|=1, base_coord=1. L1 saturation threshold =
        // α·λ·γ·base = 0.5·0.1·3·1 = 0.15. |β|=1 ≫ 0.15 ⇒ coord L1 = 0.
        // Group 1 has β=[0.05, 0.05], coords have base_coord=1; thresholds
        // ≈ 0.15 — coord L1 weights stay positive (1 − 0.05/0.15 ≈ 0.667).
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = array![1.0, 1.0, 0.05, 0.05];
        let base_group = array![1.0, 1.0];
        let base_coord = array![1.0, 1.0, 1.0, 1.0];
        let (gw, cw) = surrogate_sparse_group_mcp(
            beta.view(),
            &groups,
            0.1,
            3.0,
            0.5,
            base_group.view(),
            base_coord.view(),
        );
        // Group 0: saturated.
        assert_abs_diff_eq!(gw[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw[0][0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw[0][1], 0.0, epsilon = 1e-12);
        // Group 1: still positive.
        assert!(gw[1] > 0.5);
        assert!(cw[1][0] > 0.5 && cw[1][0] < 0.9);
    }

    // ---- group SCAD surrogate weights -----------------------------------

    #[test]
    fn surrogate_weights_group_scad_at_zero_beta_returns_base_weights() {
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = Array1::<f64>::zeros(4);
        let base = array![1.5, 0.7];
        let w = surrogate_weights_group_scad(beta.view(), &groups, 0.1, 3.7, base.view());
        for g in 0..2 {
            assert_abs_diff_eq!(w[g], base[g], epsilon = 1e-12);
        }
    }

    #[test]
    fn surrogate_weights_group_scad_zeros_saturated_group() {
        // λ = 0.1, a = 3.7, base = 1 ⇒ saturation threshold a·λ = 0.37.
        // ‖β_0‖ = √2 ≈ 1.41 ≫ 0.37 ⇒ saturated → 0.
        // ‖β_1‖ = √0.005 ≈ 0.071 < λ = 0.1 ⇒ weight = base = 1.
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = array![1.0, 1.0, 0.05, 0.05];
        let base = array![1.0, 1.0];
        let w = surrogate_weights_group_scad(beta.view(), &groups, 0.1, 3.7, base.view());
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn surrogate_weights_group_scad_in_linear_decay_region() {
        // Construct ‖β‖ such that λ < ‖β‖ < a·λ: w_lla = (a·base − ‖β‖/λ)/(a−1).
        // λ = 0.1, a = 4, base = 1. λ_eff = 0.1, a·λ_eff = 0.4.
        // Pick ‖β‖ = 0.2 ⇒ w_lla = (4·1 − 2)/3 = 2/3.
        let groups = Groups::contiguous_blocks(2, 2);
        // Construct β with norm exactly 0.2: e.g., β = [0.16, 0.12] gives
        // norm = √(0.0256+0.0144) = √0.04 = 0.2.
        let beta = array![0.16, 0.12];
        let base = array![1.0];
        let w = surrogate_weights_group_scad(beta.view(), &groups, 0.1, 4.0, base.view());
        assert_abs_diff_eq!(w[0], 2.0 / 3.0, epsilon = 1e-12);
    }

    // ---- sparse-group SCAD surrogate weights ----------------------------

    #[test]
    fn surrogate_sparse_group_scad_at_zero_beta_returns_base_weights() {
        let p = 4;
        let groups = Groups::contiguous_blocks(p, 2);
        let beta = Array1::<f64>::zeros(p);
        let base_group = array![1.5, 0.7];
        let base_coord = array![2.0, 1.0, 3.0, 0.5];
        let (gw, cw) = surrogate_sparse_group_scad(
            beta.view(),
            &groups,
            0.1,
            3.7,
            0.4,
            base_group.view(),
            base_coord.view(),
        );
        for g in 0..2 {
            assert_abs_diff_eq!(gw[g], base_group[g], epsilon = 1e-12);
        }
        assert_abs_diff_eq!(cw[0][0], base_coord[0], epsilon = 1e-12);
        assert_abs_diff_eq!(cw[0][1], base_coord[1], epsilon = 1e-12);
        assert_abs_diff_eq!(cw[1][0], base_coord[2], epsilon = 1e-12);
        assert_abs_diff_eq!(cw[1][1], base_coord[3], epsilon = 1e-12);
    }

    #[test]
    fn surrogate_sparse_group_scad_zeros_saturated_components() {
        // λ=0.1, a=3.7, α=0.5, base=1.
        // Group 0: ‖β‖=√2 ≫ a·(1−α)·λ = 0.185 → L2 saturated → 0.
        //          |β_0|=1 ≫ a·α·λ = 0.185 → L1 saturated → 0.
        // Group 1: ‖β‖=√0.02 ≈ 0.141, between (1−α)λ=0.05 and a·(1−α)·λ=0.185
        //          → linear decay. Coords similarly mid-range.
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = array![1.0, 1.0, 0.1, 0.1];
        let base_group = array![1.0, 1.0];
        let base_coord = array![1.0, 1.0, 1.0, 1.0];
        let (gw, cw) = surrogate_sparse_group_scad(
            beta.view(),
            &groups,
            0.1,
            3.7,
            0.5,
            base_group.view(),
            base_coord.view(),
        );
        // Group 0 saturated.
        assert_abs_diff_eq!(gw[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw[0][0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw[0][1], 0.0, epsilon = 1e-12);
        // Group 1 strictly in (0, 1): linear-decay region, partially shrunk.
        assert!(gw[1] > 0.0 && gw[1] < 1.0);
        assert!(cw[1][0] > 0.0 && cw[1][0] < 1.0);
    }
}
