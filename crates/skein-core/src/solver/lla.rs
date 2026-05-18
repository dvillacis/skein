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
        debug_assert_eq!(
            weights.len(),
            n_groups,
            "surrogate weights length must equal n_groups"
        );
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
        let block_norm: f64 = cols.iter().map(|&j| beta[j] * beta[j]).sum::<f64>().sqrt();
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
        let block_norm: f64 = cols.iter().map(|&j| beta[j] * beta[j]).sum::<f64>().sqrt();
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

/// LLA surrogate weights for the **bridge** (a.k.a. ℓ_q) penalty
/// `λ · Σ_j w_j |β_j|^q`, with `q ∈ (0, 1]`. The derivative of `|β|^q`
/// at `|β| > 0` is `q · sign(β) · |β|^(q-1)`, so the LLA inner per-
/// coordinate weight is `q · |β_old|^(q-1) · w_j_base`. At β = 0 this is
/// infinite — we add an `eps` floor to the magnitude before exponentiation
/// so the inner weight stays finite. `eps = 1e-6` works well in practice;
/// smaller `eps` produces sharper sparsification but more outer LLA
/// iterations.
///
/// Pair with [`crate::penalty::ElasticNet::with_weights`] at `α = 1` (i.e.
/// weighted lasso) as the inner penalty inside `solve_path_lla`'s closure.
pub fn surrogate_weights_bridge(
    beta: ArrayView1<f64>,
    q: f64,
    eps: f64,
    base_weights: ArrayView1<f64>,
) -> Array1<f64> {
    assert!(
        q > 0.0 && q <= 1.0,
        "bridge q must be in (0, 1] (got {})",
        q
    );
    assert!(eps > 0.0, "bridge eps must be > 0 (got {})", eps);
    debug_assert_eq!(beta.len(), base_weights.len());
    Array1::from_iter((0..beta.len()).map(|j| {
        let m = beta[j].abs() + eps;
        q * m.powf(q - 1.0) * base_weights[j]
    }))
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
        let norm_sq: f64 = groups.group(g).iter().map(|&j| beta[j] * beta[j]).sum();
        let norm = norm_sq.sqrt();
        (base_weights[g] - norm / denom).max(0.0)
    }))
}

/// Scalar MCP penalty value `MCP(t; λ, γ)`. Helper for composite-MCP /
/// gel value functions. Caller is responsible for handling per-coord
/// weights — this returns the unweighted value at a single nonneg `t`.
fn mcp_scalar_value(t: f64, lambda: f64, gamma: f64) -> f64 {
    // ρ_MCP(t; λ, γ) = λt − t²/(2γ)  for t ∈ [0, γλ];  γλ²/2  otherwise.
    let cutoff = gamma * lambda;
    if t < cutoff {
        lambda * t - t * t / (2.0 * gamma)
    } else {
        0.5 * gamma * lambda * lambda
    }
}

/// MCP's L1-equivalent factor at magnitude `t`: `ρ'_MCP(t; λ, γ) / λ`.
/// Equals `(1 − t/(γλ))_+`. Reused by the cMCP / gel LLA surrogates.
fn mcp_l1_factor(t: f64, lambda: f64, gamma: f64) -> f64 {
    let denom = gamma * lambda;
    if denom <= 0.0 {
        return 0.0;
    }
    (1.0 - t / denom).max(0.0)
}

/// LLA surrogate weights for the **composite MCP (cMCP)** penalty
/// (Breheny & Huang 2009, "bi-level selection"). The outer MCP is applied
/// to the sum of per-coordinate inner MCPs in each group, producing
/// hierarchical group / within-group sparsity:
///
/// ```text
/// P(β) = Σ_g w^g_g · MCP_{λ, γ₁}(Σ_k w^c_{g,k} · MCP_{λ, γ₂}(|β_{g,k}|))
/// ```
///
/// At the current iterate `β` the LLA inner weight on `|β_{g,k}|` is the
/// chain-rule derivative `∂P/∂|β_{g,k}|`, divided by `λ` so that the
/// inner penalty inside [`solve_path_lla`] (which uses `lam · w` as the
/// L1 threshold) reproduces the correct first-order condition:
///
/// ```text
/// W^lla_{g,k}(β, λ) = w^g_g · w^c_{g,k} · λ
///                   · (1 − s_g/(γ₁λ))_+ · (1 − |β_{g,k}|/(γ₂λ))_+
/// ```
///
/// where `s_g = Σ_k w^c_{g,k} · MCP_{λ,γ₂}(|β_{g,k}|)`. At `β = 0` this
/// reduces to `λ · w^g_g · w^c_{g,k}` — i.e. cMCP's boundary gradient
/// scales as `λ²`, the well-known wrinkle of the composite parameterization.
/// Callers should use [`cmcp_lambda_max`] (not the generic `lambda_max`)
/// to compute the cold-start λ for the auto grid.
///
/// Pair with [`crate::penalty::ElasticNet::with_weights`] at `α = 1`
/// inside `solve_path_lla`'s closure.
pub fn surrogate_weights_cmcp(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    gamma1: f64,
    gamma2: f64,
    base_group: ArrayView1<f64>,
    base_coord: ArrayView1<f64>,
) -> Array1<f64> {
    assert!(gamma1 > 1.0, "cMCP outer γ₁ must be > 1 (got {})", gamma1);
    assert!(gamma2 > 1.0, "cMCP inner γ₂ must be > 1 (got {})", gamma2);
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_group.len(), n_groups);
    debug_assert_eq!(base_coord.len(), beta.len());

    let mut w = Array1::<f64>::zeros(beta.len());
    for g in 0..n_groups {
        let cols = groups.group(g);
        // Inner aggregate s_g = Σ_k w^c_{g,k} · MCP(|β_{g,k}|; λ, γ₂).
        let s_g: f64 = cols
            .iter()
            .map(|&j| base_coord[j] * mcp_scalar_value(beta[j].abs(), lambda, gamma2))
            .sum();
        let outer_factor = mcp_l1_factor(s_g, lambda, gamma1);
        if outer_factor <= 0.0 {
            continue; // saturated group → all coords get zero weight
        }
        let group_scale = base_group[g] * outer_factor * lambda;
        for &j in cols {
            let inner_factor = mcp_l1_factor(beta[j].abs(), lambda, gamma2);
            w[j] = group_scale * base_coord[j] * inner_factor;
        }
    }
    w
}

/// Closed-form λ_max for cMCP: the smallest λ at which `β = 0` is optimal
/// under the composite MCP. Because the cold-start gradient scales as `λ²`,
/// the formula is `sqrt(max_j |∂L/∂β_j| / (w^g_g(j) · w^c_j))` rather than
/// the linear `lambda_max` used for L1-equivalent penalties.
pub fn cmcp_lambda_max(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    groups: &Groups,
    base_group: ArrayView1<f64>,
    base_coord: ArrayView1<f64>,
) -> f64 {
    let p = design.n_features();
    let zero_beta = Array1::<f64>::zeros(p);
    let r0 = datafit.init_residual(design, zero_beta.view());
    let mut max_q = 0.0_f64;
    for g in 0..groups.n_groups() {
        let wg = base_group[g];
        if wg <= 0.0 {
            continue;
        }
        for &j in groups.group(g) {
            let wc = base_coord[j];
            if wc <= 0.0 {
                continue;
            }
            let coord = datafit.coord_grad(design, j, r0.view()).abs();
            let q = coord / (wg * wc);
            if q > max_q {
                max_q = q;
            }
        }
    }
    max_q.sqrt()
}

/// Total cMCP penalty value at the current iterate. Used for objective
/// reporting; not on the solver hot path.
pub fn cmcp_value(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    gamma1: f64,
    gamma2: f64,
    base_group: ArrayView1<f64>,
    base_coord: ArrayView1<f64>,
) -> f64 {
    let mut total = 0.0;
    for g in 0..groups.n_groups() {
        let cols = groups.group(g);
        let s_g: f64 = cols
            .iter()
            .map(|&j| base_coord[j] * mcp_scalar_value(beta[j].abs(), lambda, gamma2))
            .sum();
        total += base_group[g] * mcp_scalar_value(s_g, lambda, gamma1);
    }
    total
}

/// LLA surrogate weights for the **group exponential lasso (gel)**
/// (Breheny 2015). The penalty per group is an exponential decay on the
/// group's L1 norm:
///
/// ```text
/// P(β) = Σ_g w^g_g · (λ²/τ) · [1 − exp(−τ · ‖β_g‖₁ / λ)]
/// ```
///
/// Its derivative w.r.t. `|β_{g,k}|` at the current iterate is
/// `w^g_g · λ · exp(−τ · ‖β_g‖₁ / λ)` — uniform across coords within a
/// group. The LLA per-coord L1 weight (so that `lam · w` reproduces this
/// in [`solve_path_lla`]'s inner penalty) is therefore the group's
/// exponential factor multiplied by the per-group base weight:
///
/// ```text
/// W^lla_{g,k}(β, λ) = w^g_g · exp(−τ · ‖β_g‖₁ / λ)
/// ```
///
/// At `β = 0` this reduces to `w^g_g` — the same boundary scaling as
/// plain weighted lasso, so the generic [`lambda_max`] works for gel
/// when called with `base_group` broadcast to per-coord weights (each
/// coord in group `g` takes weight `base_group[g]`).
pub fn surrogate_weights_gel(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    tau: f64,
    base_group: ArrayView1<f64>,
) -> Array1<f64> {
    assert!(tau > 0.0, "gel τ must be > 0 (got {})", tau);
    assert!(lambda > 0.0, "gel λ must be > 0 (got {})", lambda);
    let n_groups = groups.n_groups();
    debug_assert_eq!(base_group.len(), n_groups);

    let mut w = Array1::<f64>::zeros(beta.len());
    for g in 0..n_groups {
        let cols = groups.group(g);
        let l1_norm: f64 = cols.iter().map(|&j| beta[j].abs()).sum();
        let factor = (-tau * l1_norm / lambda).exp();
        let group_scale = base_group[g] * factor;
        for &j in cols {
            w[j] = group_scale;
        }
    }
    w
}

/// Total gel penalty value at the current iterate. Used for objective
/// reporting; not on the solver hot path.
pub fn gel_value(
    beta: ArrayView1<f64>,
    groups: &Groups,
    lambda: f64,
    tau: f64,
    base_group: ArrayView1<f64>,
) -> f64 {
    let coeff = lambda * lambda / tau;
    let mut total = 0.0;
    for g in 0..groups.n_groups() {
        let cols = groups.group(g);
        let l1_norm: f64 = cols.iter().map(|&j| beta[j].abs()).sum();
        total += base_group[g] * coeff * (1.0 - (-tau * l1_norm / lambda).exp());
    }
    total
}

/// Broadcast per-group weights to per-coord weights — `w_coord[j] = w_group[g(j)]`.
/// Useful for routing group-structured penalties (gel) through the scalar
/// LLA path solver, which expects a flat per-coord weight vector for
/// `lambda_max` computation and warm-start convergence checks.
pub fn broadcast_group_weights_to_coord(
    group_weights: ArrayView1<f64>,
    groups: &Groups,
    p: usize,
) -> Array1<f64> {
    debug_assert_eq!(group_weights.len(), groups.n_groups());
    let mut w = Array1::<f64>::zeros(p);
    for g in 0..groups.n_groups() {
        for &j in groups.group(g) {
            w[j] = group_weights[g];
        }
    }
    w
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

    // ---- cMCP surrogate -------------------------------------------------

    #[test]
    fn surrogate_weights_cmcp_at_zero_beta_returns_lambda_times_base() {
        // At β = 0, both inner and outer MCP derivatives equal λ, so
        // W^lla_{g,k}(0, λ) = w^g_g · w^c_{g,k} · λ · 1 · 1.
        let beta = Array1::<f64>::zeros(4);
        let groups = Groups::contiguous_blocks(4, 2);
        let base_group = array![1.5, 0.7];
        let base_coord = array![1.0, 2.0, 0.5, 0.8];
        let lambda = 0.3;
        let w = surrogate_weights_cmcp(
            beta.view(),
            &groups,
            lambda,
            3.0,
            3.0,
            base_group.view(),
            base_coord.view(),
        );
        for g in 0..2 {
            for &j in groups.group(g) {
                let expected = lambda * base_group[g] * base_coord[j];
                assert_abs_diff_eq!(w[j], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn surrogate_weights_cmcp_zeros_saturated_outer_block() {
        // Force the outer MCP into saturation: pick β so that the inner
        // aggregate s_g exceeds γ₁ · λ.
        let beta = array![0.5, 0.5, 0.0, 0.0];
        let groups = Groups::contiguous_blocks(4, 2);
        let base = Array1::<f64>::ones(2);
        let base_coord = Array1::<f64>::ones(4);
        let lambda = 0.05;
        let w = surrogate_weights_cmcp(
            beta.view(),
            &groups,
            lambda,
            3.0,
            3.0,
            base.view(),
            base_coord.view(),
        );
        // Group 0 saturated → w[0..2] should be 0; group 1 unchanged.
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn cmcp_lambda_max_returns_finite_positive_value() {
        let (design, y, groups) = sparse_group_problem(42);
        let datafit = LeastSquares::new(y);
        let base_group = Array1::<f64>::ones(groups.n_groups());
        let base_coord = Array1::<f64>::ones(design.n_features());
        let lam_max = cmcp_lambda_max(
            &design,
            &datafit,
            &groups,
            base_group.view(),
            base_coord.view(),
        );
        assert!(lam_max > 0.0 && lam_max.is_finite(), "got {}", lam_max);
    }

    #[test]
    fn cmcp_value_is_zero_at_zero_beta_and_positive_at_nonzero() {
        let groups = Groups::contiguous_blocks(4, 2);
        let base_group = Array1::<f64>::ones(2);
        let base_coord = Array1::<f64>::ones(4);
        let v0 = cmcp_value(
            Array1::<f64>::zeros(4).view(),
            &groups,
            0.1,
            3.0,
            3.0,
            base_group.view(),
            base_coord.view(),
        );
        let v1 = cmcp_value(
            array![0.3, -0.2, 0.0, 0.0].view(),
            &groups,
            0.1,
            3.0,
            3.0,
            base_group.view(),
            base_coord.view(),
        );
        assert_abs_diff_eq!(v0, 0.0, epsilon = 1e-12);
        assert!(v1 > 0.0);
    }

    // ---- gel surrogate --------------------------------------------------

    #[test]
    fn surrogate_weights_gel_at_zero_beta_returns_base_group_per_coord() {
        // exp(0) = 1 ⇒ each coord gets its group's base weight.
        let beta = Array1::<f64>::zeros(4);
        let groups = Groups::contiguous_blocks(4, 2);
        let base = array![1.5, 0.7];
        let w = surrogate_weights_gel(beta.view(), &groups, 0.1, 1.0, base.view());
        for g in 0..2 {
            for &j in groups.group(g) {
                assert_abs_diff_eq!(w[j], base[g], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn surrogate_weights_gel_decays_exponentially_with_l1_norm() {
        let lambda = 0.1;
        let tau = 2.0;
        let base = array![1.0, 1.0];
        let groups = Groups::contiguous_blocks(4, 2);
        // Group 0 has L1 norm 1.4, group 1 has L1 norm 0.1.
        let beta = array![0.6, 0.8, 0.05, 0.05];
        let w = surrogate_weights_gel(beta.view(), &groups, lambda, tau, base.view());
        // Expected: exp(−τ · ‖β_g‖₁ / λ) per coord in group g.
        let expected_0 = (-tau * 1.4 / lambda).exp();
        let expected_1 = (-tau * 0.1 / lambda).exp();
        for &j in groups.group(0) {
            assert_abs_diff_eq!(w[j], expected_0, epsilon = 1e-12);
        }
        for &j in groups.group(1) {
            assert_abs_diff_eq!(w[j], expected_1, epsilon = 1e-12);
        }
    }

    #[test]
    fn gel_value_is_zero_at_zero_and_monotone_in_l1_norm() {
        let groups = Groups::contiguous_blocks(4, 2);
        let base = Array1::<f64>::ones(2);
        let v0 = gel_value(
            Array1::<f64>::zeros(4).view(),
            &groups,
            0.1,
            1.0,
            base.view(),
        );
        let v_small = gel_value(
            array![0.1, 0.0, 0.0, 0.0].view(),
            &groups,
            0.1,
            1.0,
            base.view(),
        );
        let v_large = gel_value(
            array![0.5, 0.0, 0.0, 0.0].view(),
            &groups,
            0.1,
            1.0,
            base.view(),
        );
        assert_abs_diff_eq!(v0, 0.0, epsilon = 1e-12);
        assert!(v_small > 0.0);
        assert!(v_large > v_small);
    }

    #[test]
    fn broadcast_group_weights_to_coord_works() {
        let groups = Groups::contiguous_blocks(5, 2); // groups: [0,1], [2,3], [4]
        let wg = array![1.0, 2.0, 3.0];
        let wc = broadcast_group_weights_to_coord(wg.view(), &groups, 5);
        let expected = array![1.0, 1.0, 2.0, 2.0, 3.0];
        for j in 0..5 {
            assert_abs_diff_eq!(wc[j], expected[j], epsilon = 1e-12);
        }
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
