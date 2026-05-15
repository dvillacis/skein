//! M13.4c regression — native group-MCP BCD reaches a stationary point
//! of the same objective as the previous LLA-wrapped GroupLasso path,
//! for logistic / Poisson / Cox.
//!
//! Both solvers go through `prox_newton_block_solve_path`. They differ
//! only in the `make_inner` closure:
//!
//!   - LLA: returns `GroupLasso::with_weights(λ, surrogate_weights_group_mcp(β, …))`
//!     — β-dependent convex surrogate of the non-convex penalty.
//!   - Native: returns `GroupMcp::with_weights(λ, γ, w_base)` — β-
//!     independent, applies the closed-form group-MCP prox directly
//!     (Breheny & Huang 2015 §3).
//!
//! Group-MCP is non-convex, so the two solvers can land at different
//! stationary points of the same penalized objective. The thresholds
//! below are empirical on `n=200, p=20, 4 groups of 5, k_active=2` —
//! tight enough to catch a real regression (a closure that builds the
//! wrong penalty, or a screening rule mis-applied) but loose enough to
//! tolerate the non-convexity-induced disagreement at a handful of λ.

use ndarray::{Array1, Array2, ArrayView1};
use skein_core::datafit::{BinomialLogit, CoxPH, GlmDatafit, PoissonLog, TieHandling};
use skein_core::design::DenseMatrix;
use skein_core::groups::Groups;
use skein_core::penalty::{GroupLasso, GroupMcp, GroupPenalty};
use skein_core::solver::{prox_newton_block_solve_path, surrogate_weights_group_mcp, CdConfig};

const N: usize = 200;
const P: usize = 20;
const GROUP_SIZE: usize = 5;
const N_LAMBDAS: usize = 15;
const LAMBDA_MIN_RATIO: f64 = 5e-2;
const TOL: f64 = 1e-9;
const GAMMA: f64 = 3.0;
const MAX_OUTER: usize = 30;
const OUTER_TOL: f64 = 1e-8;

fn lcg_seq(seed: u64, n: usize) -> Vec<f64> {
    // Lehmer LCG → uniform in [-1, 1]; deterministic, dependency-free.
    let mut state: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 11) as u32;
            2.0 * (bits as f64) / (u32::MAX as f64) - 1.0
        })
        .collect()
}

fn box_muller(uniforms: &[f64]) -> Vec<f64> {
    // Pairs of (u1, u2) in (0,1) → standard normals.
    let mut out = Vec::with_capacity(uniforms.len());
    let mut i = 0;
    while i + 1 < uniforms.len() {
        let u1 = ((uniforms[i] + 1.0) * 0.5).max(1e-300);
        let u2 = (uniforms[i + 1] + 1.0) * 0.5;
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (2.0 * std::f64::consts::PI * u2).cos());
        out.push(r * (2.0 * std::f64::consts::PI * u2).sin());
        i += 2;
    }
    if out.len() < uniforms.len() {
        out.push(0.0);
    }
    out
}

fn design_and_groups(seed: u64) -> (Array2<f64>, Array1<f64>, Groups) {
    let raw = lcg_seq(seed, N * P);
    let xs = box_muller(&raw);
    let x = Array2::from_shape_vec((N, P), xs).unwrap();
    let mut beta = Array1::<f64>::zeros(P);
    // Active groups: 0 (cols 0..5) and 2 (cols 10..15).
    beta[0] = 0.7;
    beta[1] = -0.5;
    beta[2] = 0.6;
    beta[10] = -0.4;
    beta[11] = 0.5;
    beta[12] = -0.3;
    let groups = Groups::contiguous_blocks(P, GROUP_SIZE);
    (x, beta, groups)
}

fn objective_group_mcp(
    glm: &dyn GlmDatafit,
    design: &DenseMatrix,
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

struct AgreementStats {
    min_jaccard: f64,
    max_l2_rel: f64,
    max_obj_gap_rel: f64,
    final_lambda_jaccard: f64,
}

fn run_pair(glm: &dyn GlmDatafit, design: &DenseMatrix, groups: &Groups) -> AgreementStats {
    let n_groups = groups.n_groups();
    let base = Array1::<f64>::ones(n_groups);
    let cd = CdConfig {
        max_iter: 5000,
        tol: TOL,
        acceleration: None,
    };

    // LLA closure.
    let base_lla = base.clone();
    let make_inner_lla =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, GAMMA, base_lla.view());
            Box::new(GroupLasso::with_weights(lam, w))
        };
    let (betas_lla, report_lla) = prox_newton_block_solve_path(
        design,
        glm,
        base.clone(),
        &make_inner_lla,
        groups,
        N_LAMBDAS,
        LAMBDA_MIN_RATIO,
        None,
        &cd,
        MAX_OUTER,
        OUTER_TOL,
    );

    // Native closure on the *same* λ-grid for apples-to-apples comparison.
    let base_native = base.clone();
    let make_inner_native =
        move |_beta: ArrayView1<'_, f64>, _g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            Box::new(GroupMcp::with_weights(lam, GAMMA, base_native.clone()))
        };
    let (betas_native, report_native) = prox_newton_block_solve_path(
        design,
        glm,
        base.clone(),
        &make_inner_native,
        groups,
        N_LAMBDAS,
        LAMBDA_MIN_RATIO,
        Some(report_lla.lambdas.clone()),
        &cd,
        MAX_OUTER,
        OUTER_TOL,
    );

    assert_eq!(report_lla.lambdas.len(), report_native.lambdas.len());

    let mut min_jaccard = 1.0_f64;
    let mut max_l2_rel = 0.0_f64;
    let mut max_obj_gap_rel = 0.0_f64;
    let mut final_lambda_jaccard = 1.0_f64;
    for k in 0..report_lla.lambdas.len() {
        let beta_l = betas_lla.row(k);
        let beta_n = betas_native.row(k);
        let active_l: std::collections::HashSet<usize> =
            (0..P).filter(|&j| beta_l[j] != 0.0).collect();
        let active_n: std::collections::HashSet<usize> =
            (0..P).filter(|&j| beta_n[j] != 0.0).collect();
        let inter = active_l.intersection(&active_n).count();
        let union = active_l.union(&active_n).count();
        let jac = if union == 0 {
            1.0
        } else {
            inter as f64 / union as f64
        };
        if jac < min_jaccard {
            min_jaccard = jac;
        }
        if k == report_lla.lambdas.len() - 1 {
            final_lambda_jaccard = jac;
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
        let obj_l = objective_group_mcp(glm, design, groups, base.view(), lam, GAMMA, beta_l);
        let obj_n = objective_group_mcp(glm, design, groups, base.view(), lam, GAMMA, beta_n);
        let abs_gap = (obj_n - obj_l).abs();
        let rel_gap = if obj_l.abs() > 1e-12 {
            abs_gap / obj_l.abs()
        } else {
            abs_gap
        };
        if rel_gap > max_obj_gap_rel {
            max_obj_gap_rel = rel_gap;
        }
    }

    AgreementStats {
        min_jaccard,
        max_l2_rel,
        max_obj_gap_rel,
        final_lambda_jaccard,
    }
}

fn assert_native_compatible(name: &str, s: AgreementStats) {
    eprintln!(
        "{name}: min_jaccard={:.4} final_jaccard={:.4} max_l2_rel={:.4e} max_obj_gap_rel={:.4e}",
        s.min_jaccard, s.final_lambda_jaccard, s.max_l2_rel, s.max_obj_gap_rel
    );
    // Group-MCP is non-convex; LLA and native can legitimately reach
    // *different* stationary points at the same λ. The substantive
    // claim is that both reach equally-good local minima of the
    // *same* penalized objective. So we accept either:
    //   - tight support agreement (Jaccard ≥ 0.70), OR
    //   - tight objective agreement (max relative gap ≤ 1e-3)
    // A regression that attaches the wrong penalty or breaks the
    // outer-loop fails *both* checks. Empirically the Poisson cell
    // here finds different supports with bit-identical objectives
    // (~1e-16 relative gap) — that's fine; the optimizer found an
    // equally-good basin.
    let jaccard_ok = s.min_jaccard >= 0.70;
    let objective_ok = s.max_obj_gap_rel <= 1e-3;
    assert!(
        jaccard_ok || objective_ok,
        "{name}: min Jaccard {:.4} < 0.70 AND max relative objective gap {:.4e} > 1e-3 — native is at a meaningfully worse local minimum than LLA",
        s.min_jaccard, s.max_obj_gap_rel
    );
    // The dense-tail λ has the highest signal-to-noise; both solvers
    // should converge on the same active set there.
    assert!(
        s.final_lambda_jaccard >= 0.85,
        "{name}: final-λ Jaccard {} < 0.85 — native and LLA disagree at the dense tail",
        s.final_lambda_jaccard
    );
}

/// Generate (time, event) for Cox: positive times, ~70% events.
fn cox_times_and_events(
    seed: u64,
    x: &Array2<f64>,
    beta: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let eta = x.dot(beta);
    let u = lcg_seq(seed.wrapping_add(99), N);
    let mut time = Array1::<f64>::zeros(N);
    let mut event = Array1::<f64>::zeros(N);
    for i in 0..N {
        let u_i = ((u[i] + 1.0) * 0.5).clamp(1e-9, 1.0 - 1e-9);
        // Exponential baseline hazard; T = -ln(U) / exp(η).
        time[i] = -u_i.ln() / eta[i].exp();
        // Censor lightly (~30% censoring).
        let u_e = ((u[(i + 37) % N] + 1.0) * 0.5).max(0.0);
        event[i] = if u_e < 0.7 { 1.0 } else { 0.0 };
    }
    (time, event)
}

#[test]
fn logistic_group_mcp_native_matches_lla() {
    let (x, beta_true, groups) = design_and_groups(7);
    let eta = x.dot(&beta_true);
    let u = lcg_seq(101, N);
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let p_i = 1.0 / (1.0 + (-eta[i]).exp());
        let u_i = (u[i] + 1.0) * 0.5;
        y[i] = if u_i < p_i { 1.0 } else { 0.0 };
    }
    let design = DenseMatrix::new(x);
    let glm = BinomialLogit::new(y);
    let stats = run_pair(&glm, &design, &groups);
    assert_native_compatible("logistic", stats);
}

#[test]
fn poisson_group_mcp_native_matches_lla() {
    let (x, beta_true, groups) = design_and_groups(11);
    let eta = x.dot(&beta_true);
    let u = lcg_seq(202, N * 50);
    let mut y = Array1::<f64>::zeros(N);
    let mut k = 0;
    for i in 0..N {
        let mu = eta[i].exp().min(50.0);
        let l = (-mu).exp();
        let mut count = 0_i64;
        let mut prod = 1.0_f64;
        // Knuth's Poisson sampler.
        loop {
            count += 1;
            let u_i = ((u[k] + 1.0) * 0.5).max(1e-300);
            k = (k + 1) % u.len();
            prod *= u_i;
            if prod <= l {
                break;
            }
            if count > 200 {
                break;
            }
        }
        y[i] = (count - 1) as f64;
    }
    let design = DenseMatrix::new(x);
    let glm = PoissonLog::new(y);
    let stats = run_pair(&glm, &design, &groups);
    assert_native_compatible("poisson", stats);
}

#[test]
fn cox_group_mcp_native_matches_lla() {
    let (x, beta_true, groups) = design_and_groups(17);
    let (time, event) = cox_times_and_events(17, &x, &beta_true);
    let design = DenseMatrix::new(x);
    let glm = CoxPH::with_ties(time, event, TieHandling::Breslow);
    let stats = run_pair(&glm, &design, &groups);
    assert_native_compatible("cox", stats);
}
