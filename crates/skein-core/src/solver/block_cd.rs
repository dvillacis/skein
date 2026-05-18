//! Group block coordinate descent for separable group penalties.
//!
//! Each outer sweep visits every group, takes a block prox-gradient step
//! `β_g ← prox_group(β_g − (X_g^T r / n)/L_g; 1/L_g)`, and incrementally
//! updates the residual `r = Xβ − y`. Convergence is the CD fixed-point
//! criterion in coefficient space: the largest L₂ block-update across a
//! sweep falling below `tol`.
//!
//! The Lipschitz bound used per group is `L_g = ‖X_g‖_F² / n` (the
//! Frobenius bound). It's always safe but loose for multi-column groups —
//! a tighter operator-norm bound is a follow-up. Singleton groups recover
//! the scalar `cd_solve` step exactly.
//!
//! v0.1 scope: full group set, no working set, no LLA, no parallelism. The
//! M2.2/M2.3/M2.5 follow-ups slot in around this core.

use crate::datafit::Datafit;
use crate::design::DesignMatrix;
use crate::groups::Groups;
use crate::penalty::GroupPenalty;
use crate::solver::cd::{CdConfig, CdReport};
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use std::sync::Once;

/// One-time stderr warning when the parallel block-CD entry point is
/// invoked with overlapping groups. Fires at most once per process; any
/// downstream call after that silently uses the serial fallback. Behind
/// a `Once` so embedded users (the Python facade in particular) don't
/// see N copies on a long path × CV grid.
static OVERLAP_FALLBACK_WARNED: Once = Once::new();

/// Group block-CD with cold start at β = 0. Thin wrapper over
/// [`block_cd_solve_subset`] with the full group set.
pub fn block_cd_solve(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn GroupPenalty,
    groups: &Groups,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let p = design.n_features();
    let group_subset: Vec<usize> = (0..groups.n_groups()).collect();
    block_cd_solve_subset(
        Array1::<f64>::zeros(p),
        &group_subset,
        design,
        datafit,
        penalty,
        groups,
        config,
    )
}

/// Per-group operator-norm Lipschitz cache: `L[g] = ‖X_g‖_op² / n` for
/// every group. Built once per fit by the path solvers and reused for
/// both the gap-safe screen and the inner CD; eliminates the M2.4 → M2.6
/// redundancy where each consumer ran its own power iteration.
pub fn group_lipschitz_cache(design: &dyn DesignMatrix, groups: &Groups) -> Vec<f64> {
    (0..groups.n_groups())
        .map(|g| group_lipschitz(design, groups.group(g)))
        .collect()
}

/// Operator-norm-squared Lipschitz constant for one group:
/// `L_g = ‖X_g‖_op² / n`. Power iteration on `X_gᵀ X_g` (30-iter budget;
/// converges fast for the small block sizes typical of group penalties).
///
/// Singleton groups short-circuit to `col_sq_norm(j) / n` (operator and
/// Frobenius norms coincide for one column). Zero-norm blocks return 0.
pub fn group_lipschitz(design: &dyn DesignMatrix, cols: &[usize]) -> f64 {
    let n_f = design.n_samples() as f64;
    if cols.is_empty() {
        return 0.0;
    }
    if cols.len() == 1 {
        return design.col_sq_norm(cols[0]) / n_f;
    }
    const N_ITER: usize = 30;
    let block_size = cols.len();
    let x_g = design.columns(cols);
    let mut v = Array1::<f64>::from_elem(block_size, 1.0 / (block_size as f64).sqrt());
    for _ in 0..N_ITER {
        let xv = x_g.dot(&v);
        let mut new_v = x_g.t().dot(&xv);
        let norm = (new_v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm < 1e-30 {
            return 0.0;
        }
        let inv = 1.0 / norm;
        for x in new_v.iter_mut() {
            *x *= inv;
        }
        v = new_v;
    }
    // Rayleigh quotient at the converged unit eigenvector: ‖X_g v‖² = v^T X^T X v.
    let xv = x_g.dot(&v);
    let lambda = xv.iter().map(|x| x * x).sum::<f64>();
    lambda / n_f
}

/// Jacobi-style parallel group block-CD restricted to `group_subset`.
///
/// Each sweep snapshots β and r at the start, dispatches per-group prox-
/// gradient steps across Rayon threads (each computing against the
/// snapshot), then serially folds the resulting deltas into β and r. Same
/// per-group Frobenius Lipschitz as the serial variant — correct for
/// Jacobi when off-diagonal `X_gᵀ X_{g'}` coupling is small (uncorrelated
/// groups). For pathologically correlated groups the iterates may
/// oscillate; switch to the serial [`block_cd_solve_subset`] in that case.
///
/// **Overlapping groups are silently downgraded to serial.** The Jacobi
/// snapshot+fold scheme writes `beta[j] = new_block[k]` (assignment, not
/// increment), so two threads that both hold column `j` overwrite each
/// other's update for that coordinate and corrupt the residual fold. If
/// `groups.has_overlap()`, this function dispatches to
/// [`block_cd_solve_subset_with_cache`] instead and emits a one-time
/// stderr warning. The serial path is mathematically the right thing to
/// do in that case (Gauss-Seidel composes safely under overlap).
///
/// Empty `group_subset` returns immediately. Groups with `L_g = 0` are
/// skipped.
pub fn block_cd_solve_subset_parallel(
    beta_init: Array1<f64>,
    group_subset: &[usize],
    design: &(dyn DesignMatrix + Sync),
    datafit: &(dyn Datafit + Sync),
    penalty: &(dyn GroupPenalty + Sync),
    groups: &Groups,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let group_lip = group_lipschitz_cache(design, groups);
    block_cd_solve_subset_parallel_with_cache(
        beta_init,
        group_subset,
        &group_lip,
        design,
        datafit,
        penalty,
        groups,
        config,
    )
}

/// Cache-aware variant of [`block_cd_solve_subset_parallel`]. `group_lip`
/// must have length `groups.n_groups()` and contain the operator-norm
/// Lipschitz `‖X_g‖_op² / n` for every group (typically built once via
/// [`group_lipschitz_cache`] at the start of a path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_cd_solve_subset_parallel_with_cache(
    beta_init: Array1<f64>,
    group_subset: &[usize],
    group_lip: &[f64],
    design: &(dyn DesignMatrix + Sync),
    datafit: &(dyn Datafit + Sync),
    penalty: &(dyn GroupPenalty + Sync),
    groups: &Groups,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let p = design.n_features();
    debug_assert_eq!(beta_init.len(), p, "beta_init length must equal n_features");
    debug_assert_eq!(
        group_lip.len(),
        groups.n_groups(),
        "group_lip length must equal n_groups"
    );

    // C5 safety: Jacobi updates corrupt shared coordinates under
    // overlap (snapshot-fold writes `beta[j] = new_block[k]`, so two
    // threads holding column `j` overwrite each other). Dispatch to the
    // serial Gauss-Seidel path; that's the correct algorithm for
    // overlapping groups. Warn once so the misuse doesn't ship silently
    // — repeating per-λ × per-fold would be noise spam from joblib.
    if groups.has_overlap() {
        OVERLAP_FALLBACK_WARNED.call_once(|| {
            eprintln!(
                "skein-core: parallel block-CD requested with overlapping groups; \
                 falling back to serial Gauss-Seidel (Jacobi snapshot-fold corrupts \
                 shared coordinates). This warning fires once per process."
            );
        });
        return block_cd_solve_subset_with_cache(
            beta_init,
            group_subset,
            group_lip,
            design,
            datafit,
            penalty,
            groups,
            config,
        );
    }

    let mut beta = beta_init;
    let mut r = datafit.init_residual(design, beta.view());

    if group_subset.is_empty() {
        let obj = datafit.value(r.view()) + penalty.value(beta.view(), groups);
        return (
            beta,
            CdReport {
                iter: 0,
                converged: true,
                final_obj: obj,
            },
        );
    }

    // Project the cached per-all-groups Lipschitz onto the subset for the
    // hot path's iteration order.
    let group_lip_subset: Vec<(usize, f64)> =
        group_subset.iter().map(|&g| (g, group_lip[g])).collect();
    let group_lip = group_lip_subset; // shadow for the rest of the body

    let mut report = CdReport {
        iter: 0,
        converged: false,
        final_obj: 0.0,
    };

    for it in 0..config.max_iter {
        // Snapshot β and r at the start of the sweep — every group's prox
        // computation reads this snapshot, never the in-flight state.
        let beta_snapshot = beta.clone();
        let r_snapshot = r.clone();
        let beta_snap_ref = &beta_snapshot;
        let r_snap_ref = &r_snapshot;

        let updates: Vec<(Vec<usize>, Array1<f64>)> = group_lip
            .par_iter()
            .filter(|&&(_, lg)| lg != 0.0)
            .map(|&(g, lg)| {
                let cols = groups.group(g).to_vec();
                let mut new_block = Array1::<f64>::zeros(cols.len());
                for (k, &j) in cols.iter().enumerate() {
                    let grad = datafit.coord_grad(design, j, r_snap_ref.view());
                    new_block[k] = beta_snap_ref[j] - grad / lg;
                }
                penalty.prox_group(g, new_block.view_mut(), 1.0 / lg);
                (cols, new_block)
            })
            .collect();

        // Serially apply each group's δ to β and r. Each δ is computed
        // against the snapshot, so the order of application doesn't matter.
        let mut max_block_change = 0.0_f64;
        for (cols, new_block) in &updates {
            let mut delta_norm_sq = 0.0_f64;
            for (k, &j) in cols.iter().enumerate() {
                let delta = new_block[k] - beta_snapshot[j];
                if delta != 0.0 {
                    // r += δ · X[:, j] — zero-alloc via DesignMatrix::col_axpy.
                    design.col_axpy(j, delta, r.view_mut());
                    beta[j] = new_block[k];
                    delta_norm_sq += delta * delta;
                }
            }
            let delta_norm = delta_norm_sq.sqrt();
            if delta_norm > max_block_change {
                max_block_change = delta_norm;
            }
        }

        let obj = datafit.value(r.view()) + penalty.value(beta.view(), groups);
        report.iter = it + 1;
        report.final_obj = obj;
        if max_block_change < config.tol {
            report.converged = true;
            break;
        }
    }

    (beta, report)
}

/// Group block-CD restricted to `group_subset`. Groups not in the subset
/// keep their `beta_init` values; their contribution to `Xβ` is captured
/// because the residual is initialized from the full β.
///
/// Empty `group_subset` returns immediately (no work to do, considered
/// converged). Groups with `L_g = 0` (all-zero columns) are skipped.
pub fn block_cd_solve_subset(
    beta_init: Array1<f64>,
    group_subset: &[usize],
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn GroupPenalty,
    groups: &Groups,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let group_lip = group_lipschitz_cache(design, groups);
    block_cd_solve_subset_with_cache(
        beta_init,
        group_subset,
        &group_lip,
        design,
        datafit,
        penalty,
        groups,
        config,
    )
}

/// Cache-aware variant of [`block_cd_solve_subset`]. See
/// [`block_cd_solve_subset_parallel_with_cache`] for the cache contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_cd_solve_subset_with_cache(
    beta_init: Array1<f64>,
    group_subset: &[usize],
    group_lip: &[f64],
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn GroupPenalty,
    groups: &Groups,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let n = design.n_samples();
    let p = design.n_features();
    debug_assert_eq!(beta_init.len(), p, "beta_init length must equal n_features");
    debug_assert_eq!(
        group_lip.len(),
        groups.n_groups(),
        "group_lip length must equal n_groups"
    );

    let mut beta = beta_init;
    let mut r = datafit.init_residual(design, beta.view());

    if group_subset.is_empty() {
        let obj = datafit.value(r.view()) + penalty.value(beta.view(), groups);
        return (
            beta,
            CdReport {
                iter: 0,
                converged: true,
                final_obj: obj,
            },
        );
    }

    // Project the cached per-all-groups Lipschitz onto the subset.
    let group_lip_subset: Vec<(usize, f64)> =
        group_subset.iter().map(|&g| (g, group_lip[g])).collect();
    let group_lip = group_lip_subset; // shadow for the rest of the body

    let mut report = CdReport {
        iter: 0,
        converged: false,
        final_obj: 0.0,
    };

    for it in 0..config.max_iter {
        let mut max_block_change = 0.0_f64;

        for &(g, lg) in &group_lip {
            if lg == 0.0 {
                continue;
            }
            let cols: Vec<usize> = groups.group(g).to_vec();
            let block_size = cols.len();

            let x_g = design.columns(&cols);

            let mut new_block = Array1::<f64>::zeros(block_size);
            for (k, &j) in cols.iter().enumerate() {
                let grad = datafit.coord_grad(design, j, r.view());
                new_block[k] = beta[j] - grad / lg;
            }

            penalty.prox_group(g, new_block.view_mut(), 1.0 / lg);

            let mut delta_norm_sq = 0.0_f64;
            for (k, &j) in cols.iter().enumerate() {
                let delta = new_block[k] - beta[j];
                if delta != 0.0 {
                    for i in 0..n {
                        r[i] += delta * x_g[[i, k]];
                    }
                    beta[j] = new_block[k];
                    delta_norm_sq += delta * delta;
                }
            }
            let delta_norm = delta_norm_sq.sqrt();
            if delta_norm > max_block_change {
                max_block_change = delta_norm;
            }
        }

        let obj = datafit.value(r.view()) + penalty.value(beta.view(), groups);
        report.iter = it + 1;
        report.final_obj = obj;
        if max_block_change < config.tol {
            report.converged = true;
            break;
        }
    }

    (beta, report)
}

// Wired up by the group path solver in M2.4; only used by unit tests today.
#[allow(dead_code)]
/// Tibshirani sequential strong rule for group penalties.
///
/// Drops group `g` when:
///   - it's penalized (`w_g > 0`), AND
///   - it's currently inactive (every coordinate of `β_g` is exactly zero), AND
///   - `‖X_gᵀ r_{k-1}‖₂ / n < (2 λ_k − λ_{k-1}) · w_g`.
///
/// Currently-active groups are always kept, mirroring the scalar rule.
/// This matters for nonconvex group penalties in the saturated regime
/// where the gradient at the optimum can be ≈ 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_strong_rule_screen(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    residual: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    beta: ArrayView1<'_, f64>,
    groups: &Groups,
    lambda_k: f64,
    lambda_prev: f64,
) -> Vec<usize> {
    let threshold = 2.0 * lambda_k - lambda_prev;
    let n_groups = groups.n_groups();
    let mut ws = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let w = weights[g];
        let cols = groups.group(g);
        let active = cols.iter().any(|&j| beta[j] != 0.0);
        if w <= 0.0 || active {
            ws.push(g);
            continue;
        }
        let group_grad_norm: f64 = cols
            .iter()
            .map(|&j| {
                let g_coord = datafit.coord_grad(design, j, residual);
                g_coord * g_coord
            })
            .sum::<f64>()
            .sqrt();
        if group_grad_norm >= threshold * w {
            ws.push(g);
        }
    }
    ws
}

/// Gap-safe sphere screen for LS + convex group lasso (Fercoq–Gramfort–Salmon
/// 2015 generalized to groups).
///
/// Constructs a feasible dual `θ = scale · (−r/n)` where `scale =
/// min(1, λ·w_g* / max_g ‖g_g‖₂)` (`g_g = X_gᵀ r / n` per-group gradient
/// vector), computes the duality gap, and screens group `g` when
/// `scale · ‖g_g‖₂ + √(2 gap / n) · ‖X_g‖_op < λ · w_g`.
///
/// Provably tighter than the strong rule on convex problems.
/// Currently-active groups (`‖β_g‖ > 0`) and unpenalized ones are kept
/// regardless, mirroring the strong-rule convention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_gap_safe_screen(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    residual: ArrayView1<'_, f64>,
    beta: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    groups: &Groups,
    lambda: f64,
    group_lip: &[f64],
) -> Vec<usize> {
    debug_assert_eq!(
        group_lip.len(),
        groups.n_groups(),
        "group_lip length must equal n_groups"
    );
    let n = design.n_samples() as f64;
    let n_groups = groups.n_groups();

    // Full gradient ∂L/∂β. LS overrides this with one matvec; default
    // impl loops over coord_grad.
    let g_full = datafit.full_grad(design, residual);

    // Per-group gradient L₂ norm + dual feasibility scale.
    let mut group_grad_norm = vec![0.0_f64; n_groups];
    let mut max_ratio = 0.0_f64;
    for gi in 0..n_groups {
        let cols = groups.group(gi);
        let norm_sq: f64 = cols.iter().map(|&j| g_full[j].powi(2)).sum();
        let norm = norm_sq.sqrt();
        group_grad_norm[gi] = norm;
        let w = weights[gi];
        if w > 0.0 {
            let ratio = norm / (lambda * w);
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }
    let scale = if max_ratio > 1.0 {
        1.0 / max_ratio
    } else {
        1.0
    };

    // Primal: datafit.value(r) (handles weighted vs unweighted) + λ Σ w_g ‖β_g‖₂.
    let mut pen_value = 0.0_f64;
    for gi in 0..n_groups {
        let w = weights[gi].max(0.0);
        if w == 0.0 {
            continue;
        }
        let cols = groups.group(gi);
        let norm: f64 = cols.iter().map(|&j| beta[j] * beta[j]).sum::<f64>().sqrt();
        pen_value += w * norm;
    }
    let primal_obj = datafit.value(residual) + lambda * pen_value;

    // Dual obj via the trait method: returns the closed-form
    // `D(scale·θ_naive)` for whichever LS variant the datafit is
    // (unweighted ‖r‖² or weighted Σwᵢrᵢ² — Phase 1 weighted-LS dual
    // unlock). Returns `None` for datafits without a closed-form lasso
    // dual; in that case we fall back to "no screening" (empty WS
    // expansion is still safe — the caller's KKT verifier catches
    // anything missed). For the GLM surrogate path the surrogate is a
    // weighted-LS so `Some(_)` always.
    let dual_obj_opt = datafit.lasso_dual_obj(design, beta, residual, g_full.view(), scale);
    // Weighted-LS strong-convexity correction to the safe sphere radius:
    // σ_dual = n / max(w), so r_safe² = 2·gap·max(w)/n. Matches the
    // scalar `compute_outer_state` in `solver::path` — see the long
    // comment there for the derivation and the Poisson-vs-logistic
    // trade-off.
    let max_w = match datafit.sample_weights() {
        Some(w) => w.iter().fold(0.0_f64, |a, &v| a.max(v)),
        None => 1.0,
    };
    let safe_r = match dual_obj_opt {
        Some(d) => {
            let g = (primal_obj - d).max(0.0);
            (2.0 * g * max_w / n).sqrt()
        }
        None => {
            // No dual ⇒ no safe radius ⇒ no screening. Return all groups
            // so the caller doesn't accidentally prune any.
            let mut ws = Vec::with_capacity(n_groups);
            for gi in 0..n_groups {
                ws.push(gi);
            }
            return ws;
        }
    };

    let mut ws = Vec::with_capacity(n_groups);
    for gi in 0..n_groups {
        let w = weights[gi];
        let cols = groups.group(gi);
        let active = cols.iter().any(|&j| beta[j] != 0.0);
        if w <= 0.0 || active {
            ws.push(gi);
            continue;
        }
        // ‖X_g‖_op recovered from the precomputed cache (M2.6's
        // group_lipschitz returns ‖X_g‖_op² / n).
        let xg_op = (group_lip[gi] * n).sqrt();
        let lhs = scale * group_grad_norm[gi];
        if lhs + safe_r * xg_op >= lambda * w {
            ws.push(gi);
        }
    }
    ws
}

// Wired up by the group path solver in M2.4; only used by unit tests today.
#[allow(dead_code)]
/// KKT verifier on the complement of the current group working set.
///
/// Returns indices of groups `g ∉ in_ws` whose group gradient violates
/// `‖X_gᵀ r / n‖₂ ≤ λ · w_g`. `in_ws` must be sorted ascending. Used by
/// the path solver's outer loop to pull missed groups back into the WS.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_find_kkt_violators(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    residual: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    in_ws: &[usize],
    groups: &Groups,
    lambda: f64,
    tol: f64,
) -> Vec<usize> {
    let n_groups = groups.n_groups();
    let mut violators = Vec::new();
    let mut idx = 0usize;
    for g in 0..n_groups {
        if idx < in_ws.len() && in_ws[idx] == g {
            idx += 1;
            continue;
        }
        let w = weights[g];
        let cols = groups.group(g);
        let group_grad_norm: f64 = cols
            .iter()
            .map(|&j| {
                let g_coord = datafit.coord_grad(design, j, residual);
                g_coord * g_coord
            })
            .sum::<f64>()
            .sqrt();
        let bound = if w <= 0.0 { tol } else { lambda * w + tol };
        if group_grad_norm > bound {
            violators.push(g);
        }
    }
    violators
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::DenseMatrix;
    use crate::penalty::{GroupLasso, GroupMcp, Mcp};
    use crate::solver::{cd_solve, CdConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    fn sparse_group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Groups) {
        // 8 features in 4 groups of 2; truth has groups 0 and 2 active.
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
        true_beta[1] = -1.0; // group 0
        true_beta[4] = 0.7;
        true_beta[5] = 1.2; // group 2
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), y, groups)
    }

    fn group_block_norm(beta: &Array1<f64>, groups: &Groups, g: usize) -> f64 {
        groups
            .group(g)
            .iter()
            .map(|&j| beta[j] * beta[j])
            .sum::<f64>()
            .sqrt()
    }

    // ---- group lasso correctness ----------------------------------------

    #[test]
    fn block_cd_zeros_all_groups_under_strong_penalty() {
        let (design, y, groups) = sparse_group_problem(1);
        let datafit = LeastSquares::new(y);
        let penalty = GroupLasso::new(10.0, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 100,
            tol: 1e-8,
            acceleration: None,
        };
        let (beta, report) = block_cd_solve(&design, &datafit, &penalty, &groups, &cfg);
        assert!(report.iter > 0);
        for g in 0..groups.n_groups() {
            assert_abs_diff_eq!(group_block_norm(&beta, &groups, g), 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn block_cd_recovers_active_groups_under_small_penalty() {
        let (design, y, groups) = sparse_group_problem(2);
        let datafit = LeastSquares::new(y);
        // Small λ ⇒ active groups (0 and 2) survive, inactive (1, 3) shrink.
        let penalty = GroupLasso::new(0.005, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta, _) = block_cd_solve(&design, &datafit, &penalty, &groups, &cfg);
        // Active groups should have meaningful norm.
        assert!(group_block_norm(&beta, &groups, 0) > 0.5);
        assert!(group_block_norm(&beta, &groups, 2) > 0.5);
        // Inactive groups should be much smaller (group lasso shrinks but
        // doesn't necessarily zero noise groups exactly at this λ).
        assert!(group_block_norm(&beta, &groups, 1) < 0.5);
        assert!(group_block_norm(&beta, &groups, 3) < 0.5);
    }

    #[test]
    fn block_cd_singleton_groups_match_scalar_cd_solve_on_lasso() {
        // When every group is a singleton, group lasso reduces to lasso and
        // block-CD must produce the same β as scalar cd_solve.
        let x = array![
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.2, 0.8, 1.0],
            [0.1, 0.4, 0.9]
        ];
        let y = array![1.0, 0.5, 0.3, 0.2];
        let p = 3;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };

        let groups = Groups::singletons(p);
        let group_pen = GroupLasso::new(0.05, p);
        let (beta_block, _) = block_cd_solve(&design, &datafit, &group_pen, &groups, &cfg);

        // Mcp at γ → ∞ ≈ lasso, with the same per-feature weights (= 1).
        let scalar_pen = Mcp::new(0.05, 1e10, p);
        let (beta_scalar, _) = cd_solve(&design, &datafit, &scalar_pen, &cfg);

        for j in 0..p {
            assert_abs_diff_eq!(beta_block[j], beta_scalar[j], epsilon = 1e-6);
        }
    }

    // ---- group MCP single-call correctness ------------------------------

    #[test]
    fn block_cd_with_group_mcp_zeros_all_groups_under_strong_penalty() {
        let (design, y, groups) = sparse_group_problem(3);
        let datafit = LeastSquares::new(y);
        // γ = 100 keeps the prox in its convex regime here.
        let penalty = GroupMcp::new(10.0, 100.0, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 200,
            tol: 1e-8,
            acceleration: None,
        };
        let (beta, report) = block_cd_solve(&design, &datafit, &penalty, &groups, &cfg);
        assert!(report.iter > 0);
        for g in 0..groups.n_groups() {
            assert_abs_diff_eq!(group_block_norm(&beta, &groups, g), 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn block_cd_with_group_mcp_keeps_signal_under_small_penalty() {
        let (design, y, groups) = sparse_group_problem(4);
        let datafit = LeastSquares::new(y);
        let penalty = GroupMcp::new(0.005, 100.0, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta, _) = block_cd_solve(&design, &datafit, &penalty, &groups, &cfg);
        assert!(group_block_norm(&beta, &groups, 0) > 0.5);
        assert!(group_block_norm(&beta, &groups, 2) > 0.5);
    }

    // ---- edge cases -----------------------------------------------------

    #[test]
    fn block_cd_with_zero_column_group_stays_zero() {
        // A group whose columns are all zero ⇒ Lipschitz = 0 ⇒ skipped.
        let n = 10;
        let p = 4;
        let mut x = Array2::<f64>::from_elem((n, p), 0.5);
        // Group 1 = features 2,3 — set both columns to zero.
        x.column_mut(2).fill(0.0);
        x.column_mut(3).fill(0.0);
        let y = Array1::<f64>::from_shape_fn(n, |i| (i as f64).sin());
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let groups = Groups::contiguous_blocks(p, 2);
        let penalty = GroupLasso::new(0.01, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 500,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta, _) = block_cd_solve(&design, &datafit, &penalty, &groups, &cfg);
        assert_abs_diff_eq!(beta[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(beta[3], 0.0, epsilon = 1e-12);
    }

    // ---- block_cd_solve_subset ------------------------------------------

    #[test]
    fn block_cd_subset_holds_excluded_groups_fixed() {
        let (design, y, groups) = sparse_group_problem(10);
        let datafit = LeastSquares::new(y);
        let penalty = GroupLasso::new(0.01, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 1000,
            tol: 1e-10,
            acceleration: None,
        };
        // β starts non-zero in group 1 (features 2, 3) and group 3 (6, 7);
        // only update group 0 and group 2.
        let mut beta_init = Array1::<f64>::zeros(8);
        beta_init[2] = 0.9;
        beta_init[3] = -0.4;
        beta_init[6] = 0.3;
        beta_init[7] = 0.5;
        let subset = vec![0, 2];
        let (beta_out, _) = block_cd_solve_subset(
            beta_init.clone(),
            &subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );
        // Excluded coordinates must be unchanged.
        assert_abs_diff_eq!(beta_out[2], beta_init[2], epsilon = 1e-12);
        assert_abs_diff_eq!(beta_out[3], beta_init[3], epsilon = 1e-12);
        assert_abs_diff_eq!(beta_out[6], beta_init[6], epsilon = 1e-12);
        assert_abs_diff_eq!(beta_out[7], beta_init[7], epsilon = 1e-12);
    }

    #[test]
    fn block_cd_subset_full_matches_block_cd_solve() {
        let (design, y, groups) = sparse_group_problem(11);
        let datafit = LeastSquares::new(y);
        let penalty = GroupLasso::new(0.01, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let (beta_full, _) = block_cd_solve(&design, &datafit, &penalty, &groups, &cfg);
        let p = design.n_features();
        let subset: Vec<usize> = (0..groups.n_groups()).collect();
        let (beta_subset, _) = block_cd_solve_subset(
            Array1::<f64>::zeros(p),
            &subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta_full[j], beta_subset[j], epsilon = 1e-10);
        }
    }

    #[test]
    fn block_cd_subset_empty_returns_immediately() {
        let (design, y, groups) = sparse_group_problem(12);
        let datafit = LeastSquares::new(y);
        let penalty = GroupLasso::new(0.1, groups.n_groups());
        let p = design.n_features();
        let mut beta_init = Array1::<f64>::zeros(p);
        beta_init[0] = 0.7;
        beta_init[5] = -0.3;
        let (beta_out, report) = block_cd_solve_subset(
            beta_init.clone(),
            &[],
            &design,
            &datafit,
            &penalty,
            &groups,
            &CdConfig {
                max_iter: 500,
                tol: 1e-10,
                acceleration: None,
            },
        );
        assert_eq!(beta_out, beta_init);
        assert!(report.converged);
        assert_eq!(report.iter, 0);
    }

    // ---- group strong rule + KKT verifier -------------------------------

    #[test]
    fn block_strong_rule_keeps_active_groups() {
        // Group 1's gradient is below threshold but it has a non-zero β
        // coordinate ⇒ the rule must retain it.
        let x = array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.2]];
        let design = DenseMatrix::new(x);
        let r = array![0.6, 0.4];
        let beta = array![0.0, 0.0, 0.3, 0.0]; // group 1 active
        let weights = array![1.0, 1.0];
        let groups = Groups::contiguous_blocks(4, 2);
        let datafit = LeastSquares::new(Array1::<f64>::zeros(2));
        let ws = block_strong_rule_screen(
            &design,
            &datafit,
            r.view(),
            weights.view(),
            beta.view(),
            &groups,
            0.5,
            0.8,
        );
        assert_eq!(ws, vec![0, 1]);
    }

    #[test]
    fn block_strong_rule_drops_below_threshold_inactive_group() {
        // Hand-computed: group 0 gradient norm/n = 0.36 (≥ 0.2 threshold,
        // kept), group 1 gradient norm/n ≈ 0.155 (< 0.2, dropped).
        let x = array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.2]];
        let design = DenseMatrix::new(x);
        let r = array![0.6, 0.4];
        let beta = Array1::<f64>::zeros(4);
        let weights = array![1.0, 1.0];
        let groups = Groups::contiguous_blocks(4, 2);
        // 2λ_k - λ_prev = 2·0.5 − 0.8 = 0.2.
        let datafit = LeastSquares::new(Array1::<f64>::zeros(2));
        let ws = block_strong_rule_screen(
            &design,
            &datafit,
            r.view(),
            weights.view(),
            beta.view(),
            &groups,
            0.5,
            0.8,
        );
        assert_eq!(ws, vec![0]);
    }

    #[test]
    fn block_kkt_violators_finds_groups_above_threshold() {
        // Both groups have gradient norms above λ·w_g (λ = 0.1) ⇒ both
        // flagged when the WS is empty.
        let x = array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.2]];
        let design = DenseMatrix::new(x);
        let r = array![0.6, 0.4];
        let weights = array![1.0, 1.0];
        let groups = Groups::contiguous_blocks(4, 2);
        let datafit = LeastSquares::new(Array1::<f64>::zeros(2));
        let violators = block_find_kkt_violators(
            &design,
            &datafit,
            r.view(),
            weights.view(),
            &[],
            &groups,
            0.1,
            1e-6,
        );
        assert_eq!(violators, vec![0, 1]);
    }

    #[test]
    fn block_kkt_violators_returns_empty_when_kkt_satisfied() {
        // Both groups already in the WS ⇒ complement empty ⇒ no violators.
        let x = array![[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.2]];
        let design = DenseMatrix::new(x);
        let r = array![0.6, 0.4];
        let weights = array![1.0, 1.0];
        let groups = Groups::contiguous_blocks(4, 2);
        let datafit = LeastSquares::new(Array1::<f64>::zeros(2));
        let violators = block_find_kkt_violators(
            &design,
            &datafit,
            r.view(),
            weights.view(),
            &[0, 1],
            &groups,
            0.1,
            1e-6,
        );
        assert!(violators.is_empty());
    }

    // ---- parallel block CD ---------------------------------------------

    #[test]
    fn block_cd_subset_parallel_matches_serial_within_tol() {
        // Convex problem with random uncorrelated columns ⇒ Jacobi sweeps
        // converge to the same β as Gauss-Seidel within tolerance.
        let (design, y, groups) = sparse_group_problem(20);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let penalty = GroupLasso::new(0.005, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let group_subset: Vec<usize> = (0..groups.n_groups()).collect();
        let (beta_serial, _) = block_cd_solve_subset(
            Array1::<f64>::zeros(p),
            &group_subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );
        let (beta_parallel, _) = block_cd_solve_subset_parallel(
            Array1::<f64>::zeros(p),
            &group_subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta_serial[j], beta_parallel[j], epsilon = 1e-6);
        }
    }

    #[test]
    fn block_cd_subset_parallel_with_overlapping_groups_falls_back_to_serial() {
        // Two groups sharing column 1: G0 = {0, 1}, G1 = {1, 2}. Jacobi
        // would corrupt β[1] (both threads write it from different
        // snapshots); the parallel entry point must detect overlap and
        // dispatch to serial. We verify by running serial directly and
        // checking bit-identical output.
        //
        // sparse_group_problem returns p=8 features; we only need 3, so
        // build a small problem inline to keep groups well-defined.
        let n = 40;
        let p = 3;
        let mut state = 7_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let true_beta = array![1.0, -0.5, 0.8];
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.02 * sample());
        let y = x.dot(&true_beta) + &noise;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);

        // Overlap: {0,1} and {1,2}.
        let groups = Groups::from_csr(vec![0, 2, 4], vec![0, 1, 1, 2]).unwrap();
        assert!(
            groups.has_overlap(),
            "test setup invariant — fixture must overlap"
        );

        let penalty = GroupLasso::new(0.05, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 2000,
            tol: 1e-12,
            acceleration: None,
        };
        let group_subset: Vec<usize> = (0..groups.n_groups()).collect();

        let (beta_serial, rep_serial) = block_cd_solve_subset(
            Array1::<f64>::zeros(p),
            &group_subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );
        let (beta_parallel, rep_parallel) = block_cd_solve_subset_parallel(
            Array1::<f64>::zeros(p),
            &group_subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );

        // Parallel-with-overlap fell back to serial ⇒ bit-identical β
        // and matching iteration count / convergence flag.
        for j in 0..p {
            assert_eq!(
                beta_serial[j], beta_parallel[j],
                "fallback must produce bit-identical β at coord {j}"
            );
        }
        assert_eq!(rep_serial.iter, rep_parallel.iter);
        assert_eq!(rep_serial.converged, rep_parallel.converged);
    }

    #[test]
    fn block_cd_subset_parallel_holds_excluded_groups_fixed() {
        let (design, y, groups) = sparse_group_problem(21);
        let datafit = LeastSquares::new(y);
        let penalty = GroupLasso::new(0.01, groups.n_groups());
        let cfg = CdConfig {
            max_iter: 1000,
            tol: 1e-10,
            acceleration: None,
        };
        let mut beta_init = Array1::<f64>::zeros(8);
        beta_init[2] = 0.9;
        beta_init[3] = -0.4;
        beta_init[6] = 0.3;
        beta_init[7] = 0.5;
        let subset = vec![0, 2];
        let (beta_out, _) = block_cd_solve_subset_parallel(
            beta_init.clone(),
            &subset,
            &design,
            &datafit,
            &penalty,
            &groups,
            &cfg,
        );
        // Groups 1 (cols 2, 3) and 3 (cols 6, 7) are excluded → unchanged.
        assert_abs_diff_eq!(beta_out[2], beta_init[2], epsilon = 1e-12);
        assert_abs_diff_eq!(beta_out[3], beta_init[3], epsilon = 1e-12);
        assert_abs_diff_eq!(beta_out[6], beta_init[6], epsilon = 1e-12);
        assert_abs_diff_eq!(beta_out[7], beta_init[7], epsilon = 1e-12);
    }

    // ---- group_lipschitz (operator-norm via power iteration) ------------

    #[test]
    fn group_lipschitz_singleton_matches_col_sq_norm_over_n() {
        // Singleton group ⇒ operator and Frobenius norms coincide.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let n = x.nrows() as f64;
        let design = DenseMatrix::new(x);
        for j in 0..2 {
            let lip = group_lipschitz(&design, &[j]);
            let expected = design.col_sq_norm(j) / n;
            assert_abs_diff_eq!(lip, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn group_lipschitz_diagonal_block_matches_max_squared_entry_over_n() {
        // X_g = diag(1, 2): X_gᵀ X_g = diag(1, 4). Largest eigenvalue = 4.
        // n = 2 ⇒ Lipschitz = 4/2 = 2.
        let x = array![[1.0, 0.0], [0.0, 2.0]];
        let design = DenseMatrix::new(x);
        let lip = group_lipschitz(&design, &[0, 1]);
        assert_abs_diff_eq!(lip, 2.0, epsilon = 1e-8);
    }

    #[test]
    fn group_lipschitz_op_norm_at_most_frobenius_bound() {
        // For any matrix, ‖M‖_op² ≤ ‖M‖_F². So power-iteration result ≤
        // sum-of-col-sq-norms result. Random 5×4 block.
        let mut state = 42_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((5, 4), |_| sample());
        let n = x.nrows() as f64;
        let design = DenseMatrix::new(x);
        let cols: Vec<usize> = (0..4).collect();
        let lip_op = group_lipschitz(&design, &cols);
        let lip_frob: f64 = cols.iter().map(|&j| design.col_sq_norm(j)).sum::<f64>() / n;
        assert!(
            lip_op <= lip_frob + 1e-12,
            "operator≤Frobenius: op={}, frob={}",
            lip_op,
            lip_frob
        );
        // For a random 5×4 block, op should be strictly less than Frob
        // (the bound is loose for non-orthogonal columns).
        assert!(
            lip_op < lip_frob,
            "expected op < frob for random block, got op={} ≥ frob={}",
            lip_op,
            lip_frob
        );
    }

    #[test]
    fn group_lipschitz_zero_columns_returns_zero() {
        let x = array![[0.0, 0.0], [0.0, 0.0]];
        let design = DenseMatrix::new(x);
        assert_abs_diff_eq!(group_lipschitz(&design, &[0, 1]), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(group_lipschitz(&design, &[]), 0.0, epsilon = 1e-12);
    }
}
