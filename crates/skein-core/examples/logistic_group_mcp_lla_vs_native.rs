//! M13.4c — LLA-wrapped GroupLasso vs native GroupMcp inside the
//! prox-Newton block path, on a logistic group-MCP problem.
//!
//! Mirrors `group_mcp_lla_vs_native.rs` (which proved the win for LS in
//! M13.4b), but the outer driver is `prox_newton_block_solve_path` so
//! the GLM's weighted-LS surrogate is rebuilt every outer iter. The two
//! solvers compared:
//!
//! 1. **LLA** (current production):
//!    `make_inner` returns `GroupLasso::with_weights(λ, w_LLA(β))` —
//!    a β-dependent convex surrogate of the non-convex group-MCP
//!    penalty.
//! 2. **Native** (proposed M13.4c):
//!    `make_inner` returns `GroupMcp::with_weights(λ, γ, w_base)` —
//!    β-independent, applies `GroupMcp::prox_group` (Breheny & Huang
//!    2015 §3 closed-form) directly inside block-CD.
//!
//! Both go through the *same* prox-Newton outer loop; only the inner
//! penalty differs. `max_outer` / `outer_tol` still govern the GLM
//! linearization convergence under both regimes.
//!
//! Reports wall-clock, outer / inner iter counts, support agreement at
//! every λ, max coefficient ℓ₂ deviation, and the original logistic-
//! NLL-plus-group-MCP-penalty objective gap.

use ndarray::{Array1, Array2, ArrayView1};
use skein_core::{
    datafit::BinomialLogit,
    design::DenseMatrix,
    groups::Groups,
    penalty::{GroupLasso, GroupMcp},
    solver::{prox_newton_block_solve_path, surrogate_weights_group_mcp, CdConfig},
    GroupPenalty,
};
use std::time::Instant;

const N: usize = 4_000;
const P: usize = 400;
const GROUP_SIZE: usize = 5;
const K_ACTIVE_GROUPS: usize = 5;
const N_LAMBDAS: usize = 50;
const LAMBDA_MIN_RATIO: f64 = 5e-3;
const TOL: f64 = 1e-8;
const GAMMA: f64 = 3.0;
const MAX_OUTER: usize = 20;
const OUTER_TOL: f64 = 1e-7;

struct Xorshift {
    state: u64,
}
impl Xorshift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn make_problem() -> (DenseMatrix, Array1<f64>, Groups) {
    let mut rng = Xorshift::new(1);
    let mut x = Array2::<f64>::zeros((N, P));
    for v in x.iter_mut() {
        *v = rng.normal();
    }
    let n_groups = P / GROUP_SIZE;
    let groups = Groups::contiguous_blocks(P, GROUP_SIZE);

    let mut beta = Array1::<f64>::zeros(P);
    let active_groups: Vec<usize> = {
        let mut idx: Vec<usize> = (0..n_groups).collect();
        for i in (1..n_groups).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        idx[..K_ACTIVE_GROUPS].to_vec()
    };
    for &g in &active_groups {
        let start = g * GROUP_SIZE;
        for j in 0..GROUP_SIZE {
            beta[start + j] = 0.8 * rng.normal();
        }
    }

    let eta = x.dot(&beta);
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let p = sigmoid(eta[i]);
        y[i] = if rng.next_f64() < p { 1.0 } else { 0.0 };
    }
    (DenseMatrix::new(x), y, groups)
}

/// Original group-MCP penalized logistic objective:
/// `BinomialLogit::loss(β) + Σ_g MCP(‖β_g‖_2; λ, γ, w_g)`.
fn objective(
    design: &DenseMatrix,
    glm: &BinomialLogit,
    groups: &Groups,
    base_w: ArrayView1<f64>,
    lambda: f64,
    gamma: f64,
    beta: ArrayView1<f64>,
) -> f64 {
    let data = glm.loss(design, beta);
    let pen = GroupMcp::with_weights(lambda, gamma, base_w.to_owned());
    data + pen.value(beta, groups)
}

fn main() {
    println!("# M13.4c — logistic LLA-wrapped vs native group-MCP BCD");
    println!();
    println!(
        "Config: n={N}, p={P}, group_size={GROUP_SIZE}, n_groups={}, \
         k_active_groups={K_ACTIVE_GROUPS}, n_lambdas={N_LAMBDAS}, \
         lambda_min_ratio={LAMBDA_MIN_RATIO}, tol={TOL}, γ={GAMMA}, \
         max_outer={MAX_OUTER}, outer_tol={OUTER_TOL}",
        P / GROUP_SIZE
    );
    println!();

    let (design, y, groups) = make_problem();
    let n_groups = groups.n_groups();
    let glm = BinomialLogit::new(y);
    let base = Array1::<f64>::ones(n_groups);

    let cd = CdConfig {
        max_iter: 5000,
        tol: TOL,
        acceleration: Some(5),
    };

    // ---- LLA (current) -----------------------------------------------
    let base_for_lla = base.clone();
    let make_inner_lla =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, GAMMA, base_for_lla.view());
            Box::new(GroupLasso::with_weights(lam, w))
        };

    println!("LLA warm-up…");
    let _ = prox_newton_block_solve_path(
        &design,
        &glm,
        base.clone(),
        &make_inner_lla,
        &groups,
        N_LAMBDAS,
        LAMBDA_MIN_RATIO,
        None,
        &cd,
        MAX_OUTER,
        OUTER_TOL,
    );
    println!("LLA measured fit…");
    let t0 = Instant::now();
    let (betas_lla, report_lla) = prox_newton_block_solve_path(
        &design,
        &glm,
        base.clone(),
        &make_inner_lla,
        &groups,
        N_LAMBDAS,
        LAMBDA_MIN_RATIO,
        None,
        &cd,
        MAX_OUTER,
        OUTER_TOL,
    );
    let lla_wall = t0.elapsed();
    let lla_outer: usize = report_lla.outer_iters.iter().sum();
    let lla_inner: usize = report_lla.inner_iters.iter().sum();

    // ---- Native (proposed) ------------------------------------------
    // Pin native to the same λ-grid as LLA so per-λ comparisons are
    // apples-to-apples (λ_max is identical anyway since both start at
    // β=0 with the same `base_weights`).
    let lambdas_pinned: Vec<f64> = report_lla.lambdas.clone();

    let base_for_native = base.clone();
    let make_inner_native =
        move |_beta: ArrayView1<'_, f64>, _g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            Box::new(GroupMcp::with_weights(lam, GAMMA, base_for_native.clone()))
        };

    println!("Native warm-up…");
    let _ = prox_newton_block_solve_path(
        &design,
        &glm,
        base.clone(),
        &make_inner_native,
        &groups,
        N_LAMBDAS,
        LAMBDA_MIN_RATIO,
        Some(lambdas_pinned.clone()),
        &cd,
        MAX_OUTER,
        OUTER_TOL,
    );
    println!("Native measured fit…");
    let t0 = Instant::now();
    let (betas_native, report_native) = prox_newton_block_solve_path(
        &design,
        &glm,
        base.clone(),
        &make_inner_native,
        &groups,
        N_LAMBDAS,
        LAMBDA_MIN_RATIO,
        Some(lambdas_pinned),
        &cd,
        MAX_OUTER,
        OUTER_TOL,
    );
    let native_wall = t0.elapsed();
    let native_outer: usize = report_native.outer_iters.iter().sum();
    let native_inner: usize = report_native.inner_iters.iter().sum();

    // ---- Headline ----
    println!();
    println!("## Headline");
    println!();
    println!("| solver | wall | outer | inner |");
    println!("|---|---:|---:|---:|");
    println!("| LLA-wrapped GroupLasso | {lla_wall:?} | {lla_outer} | {lla_inner} |");
    println!("| Native GroupMcp BCD    | {native_wall:?} | {native_outer} | {native_inner} |");
    let ratio = lla_wall.as_secs_f64() / native_wall.as_secs_f64();
    println!();
    println!(
        "Native is **{ratio:.2}×** {} than LLA on this problem.",
        if ratio > 1.0 { "faster" } else { "slower" }
    );

    // ---- Per-λ agreement ----
    let mut min_jaccard = 1.0_f64;
    let mut max_l2_rel = 0.0_f64;
    let mut max_obj_gap_rel = 0.0_f64;
    let mut sum_obj_gap_abs = 0.0_f64;
    for k in 0..report_lla.lambdas.len() {
        let beta_l = betas_lla.row(k);
        let beta_n = betas_native.row(k);

        let active_l: std::collections::HashSet<usize> =
            (0..P).filter(|&j| beta_l[j] != 0.0).collect();
        let active_n: std::collections::HashSet<usize> =
            (0..P).filter(|&j| beta_n[j] != 0.0).collect();
        let inter = active_l.intersection(&active_n).count();
        let union = active_l.union(&active_n).count();
        let jaccard = if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        };
        if jaccard < min_jaccard {
            min_jaccard = jaccard;
        }

        let mut sq_diff = 0.0_f64;
        let mut sq_l = 0.0_f64;
        for j in 0..P {
            let d = beta_l[j] - beta_n[j];
            sq_diff += d * d;
            sq_l += beta_l[j] * beta_l[j];
        }
        let l2_rel = if sq_l > 0.0 {
            (sq_diff / sq_l).sqrt()
        } else {
            sq_diff.sqrt()
        };
        if l2_rel > max_l2_rel {
            max_l2_rel = l2_rel;
        }

        let lam = report_lla.lambdas[k];
        let obj_l = objective(&design, &glm, &groups, base.view(), lam, GAMMA, beta_l);
        let obj_n = objective(&design, &glm, &groups, base.view(), lam, GAMMA, beta_n);
        let abs_gap = (obj_n - obj_l).abs();
        sum_obj_gap_abs += abs_gap;
        let rel_gap = if obj_l.abs() > 1e-12 {
            abs_gap / obj_l.abs()
        } else {
            abs_gap
        };
        if rel_gap > max_obj_gap_rel {
            max_obj_gap_rel = rel_gap;
        }
    }

    println!();
    println!("## Cross-solver agreement (per λ)");
    println!();
    println!("| metric | value |");
    println!("|---|---:|");
    println!("| min Jaccard (support overlap) | {min_jaccard:.4} |");
    println!("| max relative ℓ₂ coef deviation | {max_l2_rel:.4} |");
    println!("| max relative objective gap | {max_obj_gap_rel:.4e} |");
    println!(
        "| mean abs objective gap | {:.4e} |",
        sum_obj_gap_abs / report_lla.lambdas.len() as f64
    );

    // Final-λ snapshot.
    let kf = report_lla.lambdas.len() - 1;
    let beta_l_final = betas_lla.row(kf);
    let beta_n_final = betas_native.row(kf);
    let active_l_final = (0..P).filter(|&j| beta_l_final[j] != 0.0).count();
    let active_n_final = (0..P).filter(|&j| beta_n_final[j] != 0.0).count();
    let lam_min = report_lla.lambdas[kf];
    let obj_l_final = objective(
        &design,
        &glm,
        &groups,
        base.view(),
        lam_min,
        GAMMA,
        beta_l_final,
    );
    let obj_n_final = objective(
        &design,
        &glm,
        &groups,
        base.view(),
        lam_min,
        GAMMA,
        beta_n_final,
    );
    println!();
    println!("## Final-λ (smallest λ, densest tail)");
    println!();
    println!("| solver | active features | objective |");
    println!("|---|---:|---:|");
    println!("| LLA    | {active_l_final} / {P} | {obj_l_final:.6e} |");
    println!("| Native | {active_n_final} / {P} | {obj_n_final:.6e} |");
}
