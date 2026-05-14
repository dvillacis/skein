//! M13.4b — LLA-wrapped GroupLasso vs native GroupMcp on the canonical
//! `ls_group_mcp` medium/dense bench cell.
//!
//! Setup matches `group_mcp_ls_medium.rs` exactly (same xorshift seed,
//! same generator). The two solvers compared:
//!
//! 1. **LLA** (current production):
//!    `solve_block_path_lla` with `make_inner` building
//!    `GroupLasso::with_weights(λ, surrogate_weights_group_mcp(...))`.
//!    Outer LLA loop refines the surrogate weights between block-CD
//!    inner solves.
//! 2. **Native** (proposed M13.4b):
//!    `solve_block_path` with `GroupMcp::with_weights(λ, γ, w)`.
//!    Block-CD applied directly to the group MCP prox. No outer LLA
//!    refinement; one block-CD solve per λ. Per Breheny & Huang 2015 §3.
//!
//! Reports wall-clock for each, support agreement at every λ, max
//! coefficient deviation, and final-λ objective. The objective is the
//! ground truth ½‖y − Xβ‖²/n + Σ MCP(‖β_g‖) — same on both sides.
//!
//! **Screening note**: strong-rule / gap-safe screening for `Native` is
//! disabled (`Screening::Off`) because the screening derivations assume
//! a convex penalty and may prune groups that direct BCD on the
//! non-convex MCP would later re-activate. LLA can keep its strong
//! rule because each inner solve is convex (weighted GroupLasso). This
//! makes the comparison conservative for native — if it wins despite
//! the handicap, M13.4b is solidly worth pursuing.

use ndarray::{Array1, Array2, ArrayView1};
use skein_core::{
    datafit::{Datafit, LeastSquares},
    design::DenseMatrix,
    penalty::{GroupLasso, GroupMcp},
    solver::{
        block_lambda_max, solve_block_path, solve_block_path_lla, surrogate_weights_group_mcp,
        BlockPathConfig, CdConfig, Screening,
    },
    GroupPenalty, Groups,
};
use std::time::Instant;

const N: usize = 10_000;
const P: usize = 1_000;
const GROUP_SIZE: usize = 5;
const K_ACTIVE_GROUPS: usize = 5;
const SNR: f64 = 5.0;
const N_LAMBDAS: usize = 100;
const LAMBDA_MIN_RATIO: f64 = 1e-3;
const TOL: f64 = 1e-7;
const GAMMA: f64 = 3.0;
const MAX_OUTER: usize = 10;
const OUTER_TOL: f64 = 1e-6;

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
            beta[start + j] = rng.normal();
        }
    }

    let signal = x.dot(&beta);
    let signal_std = {
        let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        (signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / signal.len() as f64).sqrt()
    };
    let noise_scale = signal_std / SNR;
    let mut y = signal.clone();
    for v in y.iter_mut() {
        *v += noise_scale * rng.normal();
    }
    (DenseMatrix::new(x), y, groups)
}

/// Group-MCP objective: ½‖r‖²/n + Σ_g MCP(‖β_g‖_2; λ, γ) with
/// per-group weight `w_g`. Same penalty for both solvers — only the
/// solver differs.
fn group_mcp_objective(
    design: &DenseMatrix,
    datafit: &LeastSquares,
    groups: &Groups,
    base_w: ArrayView1<f64>,
    lambda: f64,
    gamma: f64,
    beta: ArrayView1<f64>,
) -> f64 {
    let r = datafit.init_residual(design, beta);
    let data_term = datafit.value(r.view());
    // GroupMcp::value is the penalty closed form.
    let pen = GroupMcp::with_weights(lambda, gamma, base_w.to_owned());
    let pen_term = pen.value(beta, groups);
    data_term + pen_term
}

fn main() {
    println!("# M13.4b — LLA-wrapped vs native group-MCP BCD");
    println!();
    println!(
        "Config: n={N}, p={P}, group_size={GROUP_SIZE}, n_groups={}, \
         k_active_groups={K_ACTIVE_GROUPS}, snr={SNR}, n_lambdas={N_LAMBDAS}, \
         lambda_min_ratio={LAMBDA_MIN_RATIO}, tol={TOL}, γ={GAMMA}, \
         max_outer={MAX_OUTER}, outer_tol={OUTER_TOL}",
        P / GROUP_SIZE
    );
    println!();

    let (design, y, groups) = make_problem();
    let n_groups = groups.n_groups();
    let datafit = LeastSquares::new(y);
    let base = Array1::<f64>::ones(n_groups);
    let lam_max = block_lambda_max(&design, &datafit, base.view(), &groups);
    println!("block_lambda_max = {:.6e}", lam_max);
    println!();

    // Common BlockPathConfig (LLA uses Strong, native uses Off — see
    // module doc).
    let cd = CdConfig {
        max_iter: 100,
        tol: TOL,
        acceleration: Some(5),
    };

    // ---- LLA (current) -----------------------------------------------
    let cfg_lla = BlockPathConfig {
        n_lambdas: N_LAMBDAS,
        lambda_min_ratio: LAMBDA_MIN_RATIO,
        lambdas: None,
        cd: cd.clone(),
        screening: Screening::Strong,
        parallel: false,
    };
    let base_for_lla = base.clone();
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_mcp(beta, g, lam, GAMMA, base_for_lla.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };

    println!("LLA warm-up…");
    let _ = solve_block_path_lla(
        &design,
        &datafit,
        base.clone(),
        make_inner.clone(),
        &groups,
        &cfg_lla,
        MAX_OUTER,
        OUTER_TOL,
    );
    println!("LLA measured fit…");
    let t0 = Instant::now();
    let (betas_lla, report_lla) = solve_block_path_lla(
        &design,
        &datafit,
        base.clone(),
        make_inner,
        &groups,
        &cfg_lla,
        MAX_OUTER,
        OUTER_TOL,
    );
    let lla_wall = t0.elapsed();
    let lla_inner: usize = report_lla.inner_iters.iter().sum();
    let lla_outer: usize = report_lla.outer_iters.iter().sum();
    let lla_kkt: usize = report_lla.kkt_passes.iter().sum();
    println!(
        "  LLA fit in {:?}; total outer={lla_outer}, inner={lla_inner}, kkt={lla_kkt}",
        lla_wall
    );

    // ---- Native (proposed M13.4b) -----------------------------------
    let cfg_native = BlockPathConfig {
        n_lambdas: N_LAMBDAS,
        lambda_min_ratio: LAMBDA_MIN_RATIO,
        // Use the SAME λ-grid as LLA so per-λ comparisons are apples-
        // to-apples. (`solve_block_path` would otherwise compute its own
        // grid from `block_lambda_max` on the GroupMcp penalty's weights,
        // which equals the LLA grid here, but pinning is safer.)
        lambdas: Some(report_lla.lambdas.clone()),
        cd: cd.clone(),
        // Strong rule: penalty-agnostic at β_g=0 (GroupMcp and GroupLasso
        // share the same KKT subdifferential `λ·[-w_g, w_g]` at zero),
        // so the threshold formula `‖∇_g L‖₂ < (2λ−λ_prev)·w_g` carries
        // over unchanged. Re-enabled (was Off in the conservative first
        // pass) — see module doc.
        screening: Screening::Strong,
        parallel: false,
    };
    let base_for_native = base.clone();
    let make_pen_native = move |lam: f64| -> Box<dyn GroupPenalty> {
        Box::new(GroupMcp::with_weights(lam, GAMMA, base_for_native.clone()))
    };

    println!("Native warm-up…");
    let _ = solve_block_path(
        &design,
        &datafit,
        make_pen_native.clone(),
        &groups,
        &cfg_native,
    );
    println!("Native measured fit…");
    let t0 = Instant::now();
    let (betas_native, report_native) =
        solve_block_path(&design, &datafit, make_pen_native, &groups, &cfg_native);
    let native_wall = t0.elapsed();
    let native_inner: usize = report_native.iters.iter().sum();
    let native_kkt: usize = report_native.kkt_passes.iter().sum();
    println!(
        "  native fit in {:?}; total inner={native_inner}, kkt={native_kkt}",
        native_wall
    );

    // ---- Headline ----
    println!();
    println!("## Headline");
    println!();
    println!("| solver | wall | inner | outer | kkt |");
    println!("|---|---:|---:|---:|---:|");
    println!(
        "| LLA-wrapped GroupLasso | {:?} | {lla_inner} | {lla_outer} | {lla_kkt} |",
        lla_wall
    );
    println!(
        "| Native GroupMcp BCD    | {:?} | {native_inner} | — | {native_kkt} |",
        native_wall
    );
    println!();
    let ratio = lla_wall.as_secs_f64() / native_wall.as_secs_f64();
    println!(
        "Native is **{ratio:.2}×** {} than LLA on this problem.",
        if ratio > 1.0 { "faster" } else { "slower" }
    );
    println!();

    // ---- Per-λ agreement ----
    let mut max_jaccard_drop = 0.0_f64;
    let mut min_jaccard = 1.0_f64;
    let mut max_l2_rel = 0.0_f64;
    let mut max_obj_gap_rel = 0.0_f64;
    let mut sum_obj_gap_abs = 0.0_f64;
    for k in 0..N_LAMBDAS {
        let beta_l = betas_lla.row(k);
        let beta_n = betas_native.row(k);

        // Jaccard on the support sets.
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
        if 1.0 - jaccard > max_jaccard_drop {
            max_jaccard_drop = 1.0 - jaccard;
        }

        // ℓ₂ relative coefficient deviation.
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

        // Objective gap (reaches the SAME group-MCP objective at the
        // SAME λ; both should land at a stationary point — possibly
        // different ones for non-convex).
        let lam = report_lla.lambdas[k];
        let obj_l =
            group_mcp_objective(&design, &datafit, &groups, base.view(), lam, GAMMA, beta_l);
        let obj_n =
            group_mcp_objective(&design, &datafit, &groups, base.view(), lam, GAMMA, beta_n);
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

    println!("## Cross-solver agreement (per λ)");
    println!();
    println!("| metric | value |");
    println!("|---|---:|");
    println!("| min Jaccard (support overlap) | {min_jaccard:.4} |");
    println!("| max (1 − Jaccard) | {max_jaccard_drop:.4} |");
    println!("| max relative ℓ₂ coef deviation | {max_l2_rel:.4} |");
    println!("| max relative objective gap | {max_obj_gap_rel:.4e} |");
    println!(
        "| mean abs objective gap | {:.4e} |",
        sum_obj_gap_abs / N_LAMBDAS as f64
    );
    println!();

    // Final-λ deep dive
    let beta_l_final = betas_lla.row(N_LAMBDAS - 1);
    let beta_n_final = betas_native.row(N_LAMBDAS - 1);
    let active_l_final = (0..P).filter(|&j| beta_l_final[j] != 0.0).count();
    let active_n_final = (0..P).filter(|&j| beta_n_final[j] != 0.0).count();
    println!("## Final-λ (smallest λ, densest tail)");
    println!();
    println!("| solver | active features | objective |");
    println!("|---|---:|---:|");
    let lam_min = report_lla.lambdas[N_LAMBDAS - 1];
    let obj_l_final = group_mcp_objective(
        &design,
        &datafit,
        &groups,
        base.view(),
        lam_min,
        GAMMA,
        beta_l_final,
    );
    let obj_n_final = group_mcp_objective(
        &design,
        &datafit,
        &groups,
        base.view(),
        lam_min,
        GAMMA,
        beta_n_final,
    );
    println!("| LLA    | {active_l_final} / {P} | {obj_l_final:.6e} |");
    println!("| Native | {active_n_final} / {P} | {obj_n_final:.6e} |");
    println!();

    // Decision hint
    println!("## Read-out");
    println!();
    if ratio >= 1.5 && max_obj_gap_rel < 1e-3 {
        println!(
            "Native BCD is {:.2}× faster AND reaches an objective within {:.2e} \
             of the LLA value. **M13.4b is worth implementing.**",
            ratio, max_obj_gap_rel
        );
    } else if ratio >= 1.5 {
        println!(
            "Native BCD is {:.2}× faster but objective gap is {:.2e}. \
             Worth a closer look — may be a worse local minimum.",
            ratio, max_obj_gap_rel
        );
    } else if ratio < 1.0 {
        println!(
            "Native BCD is *slower* ({:.2}× wall vs LLA = {:.2}×). \
             The LLA outer-loop overhead the ROADMAP hypothesised is \
             smaller than the cost of running block-CD without screening, \
             at least on this scenario. M13.4b would need to keep some \
             form of screening compatible with the non-convex prox.",
            ratio,
            1.0 / ratio
        );
    } else {
        println!(
            "Native BCD is {:.2}× faster — modest. Worth re-running \
             with `Screening::Strong` on the native side (with fixture \
             tests to verify correctness) to see if the win is bigger.",
            ratio
        );
    }
}
