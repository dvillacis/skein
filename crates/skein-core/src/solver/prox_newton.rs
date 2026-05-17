//! Proximal-Newton outer loop for GLMs.
//!
//! Wraps the M1 separable-penalty solvers around a non-quadratic loss by
//! re-linearizing the loss at the current iterate each outer iteration.
//! The inner problem is the resulting weighted-LS quadratic surrogate
//! (built by the GLM via `surrogate_at(β)`), which the M1 solvers absorb
//! unchanged because M3.1's `Datafit` trait now dispatches the gradient
//! and Lipschitz through `coord_grad` / `coord_lipschitz`.
//!
//! Generic over `&dyn GlmDatafit` (logistic, Poisson, …); the GLM
//! exposes `surrogate_at(β)` returning a weighted-LS [`LeastSquares`]
//! that the M1 inner solver consumes.
//!
//! Inner solve routes through `cd_solve_subset_weighted_ls` — the
//! surrogate is always a weighted [`crate::datafit::LeastSquares`], so we
//! pay the per-feature Lipschitz scan and the weighted-residual dot
//! product once per outer iter instead of once per coordinate update.
//!
//! Each outer iteration also restricts CD to a strong-rule-seeded working
//! set and protects it with a KKT verifier (same idiom as `solve_path`'s
//! outer KKT loop): features whose prox-gradient distance against the
//! current surrogate exceeds `tol` get added back. The KKT pass is one
//! `full_grad` matvec per outer iter, paid for many times over once the
//! sparse-regime active set is ~50× smaller than `p`.

use crate::datafit::{Datafit, GlmDatafit};
use crate::design::DesignMatrix;
use crate::penalty::Penalty;
use crate::solver::cd::{cd_solve_subset_weighted_ls_with_lips, CdConfig};
use crate::solver::path::{
    anderson_extrapolate_pair, compute_outer_state, lambda_grid, lambda_max,
    priority_rule_screen_with_grad,
};
use ndarray::{Array1, Array2};

/// Cap on the per-PN-iter dual-extrapolation history. Matches the path
/// solver's `DUAL_HISTORY_MAX` (celer's K=6); also bounds the per-iter
/// memory footprint at `2 · K · p · 8` bytes (= 96 KB at p=1000).
const DUAL_HISTORY_MAX: usize = 6;

/// Once the warm β has more than this fraction of nonzero entries, skip
/// the screened inner loop and fall back to the legacy KKT-only path.
/// Mirrors `solver::path::SCREENING_SATURATION_THRESHOLD` (M13.1) — at
/// saturation, dual extrapolation + safe-sphere screening overhead
/// (1 extra rmatvec per pass, O(p) prox calls) exceeds the screening
/// gain because nearly all features are active and can't be screened.
/// Measured on logistic_lasso small-deep (active 191/200): without
/// this bypass the screened loop is ~15% slower than the legacy path.
const PN_SCREENING_SATURATION_THRESHOLD: f64 = 0.5;

/// Minimum working-set size when the strong rule has nothing to lean on
/// (cold start with β = 0 at λ_max). Same role as `PathConfig::p0`; the
/// strong rule already grows the WS as the support fills in, so this is
/// just a floor for the initial pass.
const PROX_NEWTON_P0: usize = 10;

/// Cap on KKT-expansion passes per outer prox-Newton iteration. Each
/// pass adds at least one violator; the unbounded worst case is `p`,
/// so the cap bounds the per-outer-iter cost. In practice 1–3 passes
/// is plenty even for the densest Poisson cells.
const KKT_EXPANSION_PASSES: usize = 5;

#[derive(Debug, Clone)]
pub struct ProxNewtonReport {
    pub outer_iters: usize,
    pub outer_converged: bool,
    /// CD inner-iteration counts per outer iteration.
    pub inner_iters: Vec<usize>,
    /// Final loss at the converged β (using the original GLM
    /// cross-entropy, not the surrogate).
    pub final_loss: f64,
}

#[derive(Debug, Clone)]
pub struct ProxNewtonPathReport {
    pub lambdas: Vec<f64>,
    /// Outer prox-Newton iterations performed at each λ.
    pub outer_iters: Vec<usize>,
    /// Whether each λ's outer loop hit `outer_tol`.
    pub outer_converged: Vec<bool>,
    /// Sum of CD inner iters across all outer iters at each λ.
    pub inner_iters: Vec<usize>,
    /// Original GLM loss at the converged β for each λ.
    pub final_losses: Vec<f64>,
}

/// Single-λ proximal-Newton solve **with gap-safe sphere screening +
/// Anderson dual extrapolation on the surrogate** — celer's per-λ
/// pattern, ported into the prox-Newton inner subproblem.
///
/// Equivalent to [`prox_newton_solve`] when `lambda = None`; otherwise
/// each outer PN iter runs the same KKT-with-screening loop as
/// [`crate::solver::solve_path`] does at one λ, using the surrogate's
/// weighted-LS dual obj for gap computation. Screened features are
/// reset per PN outer iter (the surrogate changes each iter, so a
/// feature provably zero at one surrogate's optimum is not guaranteed
/// to stay zero at the next).
///
/// **Why the explicit `lambda` arg.** The penalty trait encodes λ
/// internally (it's baked into `prox_coord` and `value`), but the
/// safe-sphere bound `|grad_j| + r_safe·‖X_j‖₂ < λ·w_j` needs the
/// scalar λ separately to project the dual point and to test
/// feasibility. The path driver knows λ from the grid; the standalone
/// [`prox_newton_solve`] doesn't (and falls back to the unscreened
/// path by passing `None`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prox_newton_solve_screened(
    design: &dyn DesignMatrix,
    glm: &dyn GlmDatafit,
    penalty: &dyn Penalty,
    init_beta: Array1<f64>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
    lambda: Option<f64>,
) -> (Array1<f64>, ProxNewtonReport) {
    let p = design.n_features();
    debug_assert_eq!(init_beta.len(), p, "init_beta length must equal n_features");

    let mut warm = init_beta;
    let mut inner_iters = Vec::with_capacity(max_outer);
    let mut outer_converged = false;
    let mut outer_iters = 0usize;

    let weights: Array1<f64> = penalty.weights().to_owned();

    for outer in 0..max_outer {
        outer_iters = outer + 1;
        let surrogate = glm.surrogate_at(design, warm.view());

        let beta_old = warm.clone();
        let sw = surrogate
            .sample_weights()
            .expect("GlmDatafit surrogates always carry per-sample weights");
        let n_f = design.n_samples() as f64;

        let lips_arr = design.weighted_col_sq_norms(sw);
        let lips: Vec<f64> = lips_arr.iter().map(|&v| v / n_f).collect();

        let r0 = surrogate.init_residual(design, warm.view());
        let grad0 = surrogate.full_grad(design, r0.view());
        let n_support = warm.iter().filter(|&&b| b != 0.0).count();
        let ws_size = (n_support * 2).max(PROX_NEWTON_P0).min(p);
        let mut ws =
            priority_rule_screen_with_grad(grad0.view(), weights.view(), warm.view(), ws_size);

        // Saturation bypass: when the warm β has more than
        // PN_SCREENING_SATURATION_THRESHOLD × p nonzero entries, the
        // screening overhead (extra rmatvec for Anderson + O(p) prox
        // for safe-sphere test) outweighs the screening gain, because
        // active features can't be screened. Fall back to the legacy
        // KKT-only loop in that regime. Mirrors path.rs's M13.1.
        let saturated = (n_support as f64) > PN_SCREENING_SATURATION_THRESHOLD * (p as f64);
        let effective_lambda = if saturated { None } else { lambda };

        let inner_iter_total = match effective_lambda {
            // No screening — reproduce the legacy KKT-only loop bit-for-bit.
            None => run_kkt_only_loop(
                design, &surrogate, penalty, cd_config, &lips, &mut warm, &mut ws, p,
            ),
            Some(lam) => run_screened_loop(
                design, &surrogate, penalty, cd_config, &lips, &mut warm, &mut ws, p, lam,
            ),
        };

        inner_iters.push(inner_iter_total);

        let max_change = (0..p)
            .map(|j| (warm[j] - beta_old[j]).abs())
            .fold(0.0_f64, f64::max);
        if max_change < outer_tol {
            outer_converged = true;
            break;
        }
    }

    let final_loss = glm.loss(design, warm.view());
    (
        warm,
        ProxNewtonReport {
            outer_iters,
            outer_converged,
            inner_iters,
            final_loss,
        },
    )
}

/// Original KKT-only inner loop (no dual extrapolation, no gap-safe
/// screening). Kept as the legacy path so [`prox_newton_solve`] is
/// behaviorally unchanged when called without a λ.
#[allow(clippy::too_many_arguments)]
fn run_kkt_only_loop(
    design: &dyn DesignMatrix,
    surrogate: &crate::datafit::LeastSquares,
    penalty: &dyn Penalty,
    cd_config: &CdConfig,
    lips: &[f64],
    warm: &mut Array1<f64>,
    ws: &mut Vec<usize>,
    p: usize,
) -> usize {
    let mut inner_iter_total = 0usize;
    let mut expansion_pass = 0usize;
    loop {
        let beta_in = std::mem::take(warm);
        let (b_new, r_new, rep) = cd_solve_subset_weighted_ls_with_lips(
            beta_in, ws, design, surrogate, penalty, cd_config, lips,
        );
        *warm = b_new;
        inner_iter_total = inner_iter_total.saturating_add(rep.iter);

        let grad = surrogate.full_grad(design, r_new.view());
        let violators =
            find_kkt_violators_batched(penalty, warm.view(), grad.view(), lips, ws, cd_config.tol);
        if violators.is_empty() {
            break;
        }
        expansion_pass += 1;
        if expansion_pass >= KKT_EXPANSION_PASSES {
            *ws = (0..p).collect();
            let beta_in = std::mem::take(warm);
            let (b_new, _r_new, rep) = cd_solve_subset_weighted_ls_with_lips(
                beta_in, ws, design, surrogate, penalty, cd_config, lips,
            );
            *warm = b_new;
            inner_iter_total = inner_iter_total.saturating_add(rep.iter);
            break;
        }
        ws.extend(violators);
        ws.sort_unstable();
        ws.dedup();
    }
    inner_iter_total
}

/// Per-PN-iter screening loop. Mirrors `solver::path::solve_path`'s per-λ
/// outer KKT loop: gap-safe sphere screening using the surrogate's
/// weighted-LS dual obj, Anderson dual extrapolation on `(β, r)` pairs,
/// adaptive inner tolerance via the previous outer's prox-grad distance.
///
/// The screened mask resets per PN outer iter (surrogate-level screening
/// only — a feature provably zero at one surrogate's optimum is not
/// guaranteed to stay zero at the next surrogate). Persistent
/// across-PN-iter screening using the GLM-level dual obj from
/// [`GlmDatafit::glm_dual_obj`] is a follow-up; this implementation
/// gets most of the wall-clock benefit at much lower complexity.
#[allow(clippy::too_many_arguments)]
fn run_screened_loop(
    design: &dyn DesignMatrix,
    surrogate: &crate::datafit::LeastSquares,
    penalty: &dyn Penalty,
    cd_config: &CdConfig,
    lips: &[f64],
    warm: &mut Array1<f64>,
    ws: &mut Vec<usize>,
    p: usize,
    lambda: f64,
) -> usize {
    let mut inner_iter_total = 0usize;
    let mut expansion_pass = 0usize;
    let mut beta_history: Vec<Array1<f64>> = Vec::with_capacity(DUAL_HISTORY_MAX);
    let mut residual_history: Vec<Array1<f64>> = Vec::with_capacity(DUAL_HISTORY_MAX);
    let mut best_dual_obj: f64 = f64::NEG_INFINITY;
    let mut screened: Vec<bool> = vec![false; p];
    let mut prev_outer_pgd: f64 = f64::INFINITY;
    let mut inner_cd_cfg = cd_config.clone();

    loop {
        // Adaptive inner tolerance, celer/skglm style. Same constants as
        // the path solver (`path.rs:398-402`). Loose at the start
        // (10×config.cd.tol) when there's no prior PGD; tightens to the
        // user-requested tol as the outer loop converges.
        inner_cd_cfg.tol = if prev_outer_pgd.is_finite() {
            cd_config.tol.max(0.3 * prev_outer_pgd)
        } else {
            cd_config.tol * 10.0
        };

        let beta_in = std::mem::take(warm);
        let (b_new, r_new, rep) = cd_solve_subset_weighted_ls_with_lips(
            beta_in,
            ws,
            design,
            surrogate,
            penalty,
            &inner_cd_cfg,
            lips,
        );
        *warm = b_new;
        inner_iter_total = inner_iter_total.saturating_add(rep.iter);

        // Push current (β, r) into dual-extrapolation history.
        if beta_history.len() == DUAL_HISTORY_MAX {
            beta_history.remove(0);
            residual_history.remove(0);
        }
        beta_history.push(warm.clone());
        residual_history.push(r_new.clone());

        let extrapolation = if residual_history.len() >= 3 {
            anderson_extrapolate_pair(&residual_history, &beta_history)
        } else {
            None
        };

        let mut outer = compute_outer_state(
            design,
            surrogate,
            penalty,
            warm.view(),
            r_new.view(),
            ws,
            lips,
            lambda,
            cd_config.tol,
            extrapolation
                .as_ref()
                .map(|(r_acc, beta_acc)| (r_acc.view(), beta_acc.view())),
            &mut best_dual_obj,
        );
        // `outer.grad` isn't reused here (no cross-PN-iter grad cache);
        // the path solver's M13.2 cache is between λ's, not between
        // surrogates. Drop it explicitly via `take` to avoid the clone
        // path the compiler would otherwise pick.
        let _ = std::mem::take(&mut outer.grad);
        prev_outer_pgd = outer.max_pgd;

        // Apply gap-safe screening: pull provably-zero features out of
        // the working set permanently (for this PN iter).
        if !outer.safely_inactive.is_empty() {
            for &j in &outer.safely_inactive {
                screened[j] = true;
            }
            ws.retain(|&j| !screened[j]);
        }

        // Outer convergence check — gap-based OR PGD stationarity.
        // Mirrors `path.rs:499-505`.
        let converged = match outer.gap {
            Some(g) => g < cd_config.tol * cd_config.tol || outer.max_pgd < cd_config.tol,
            None => outer.max_pgd < cd_config.tol,
        };
        if converged {
            break;
        }

        if outer.violators.is_empty() {
            // WS is correct but the inner CD stopped sloppy. Next pass
            // will rerun with a tighter inner_tol via shrinking
            // prev_outer_pgd.
            continue;
        }

        expansion_pass += 1;
        if expansion_pass >= KKT_EXPANSION_PASSES {
            // Fall back to the full feature set, minus anything we've
            // already screened out. Same protective cap as the legacy
            // KKT-only path so a pathological surrogate can't blow the
            // outer-iter budget.
            *ws = (0..p).filter(|j| !screened[*j]).collect();
            let beta_in = std::mem::take(warm);
            let (b_new, _r_new, rep) = cd_solve_subset_weighted_ls_with_lips(
                beta_in,
                ws,
                design,
                surrogate,
                penalty,
                &inner_cd_cfg,
                lips,
            );
            *warm = b_new;
            inner_iter_total = inner_iter_total.saturating_add(rep.iter);
            break;
        }

        for j in outer.violators {
            if !screened[j] {
                ws.push(j);
            }
        }
        ws.sort_unstable();
        ws.dedup();
    }
    inner_iter_total
}

/// Single-λ proximal-Newton solve for any GLM that exposes a weighted-LS
/// surrogate via [`GlmDatafit`].
///
/// Public surface preserved (no signature change) — internally delegates
/// to [`prox_newton_solve_screened`] with `lambda = None`, which falls
/// back to the legacy KKT-only inner loop (no gap-safe screening, no
/// dual extrapolation). Callers that have an explicit λ on hand (the
/// path solver, M13 GLM-screening milestone) should route through
/// [`prox_newton_solve_screened`] directly to opt into screening.
#[allow(clippy::too_many_arguments)]
pub fn prox_newton_solve(
    design: &dyn DesignMatrix,
    glm: &dyn GlmDatafit,
    penalty: &dyn Penalty,
    init_beta: Array1<f64>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array1<f64>, ProxNewtonReport) {
    prox_newton_solve_screened(
        design, glm, penalty, init_beta, cd_config, max_outer, outer_tol, None,
    )
}

/// KKT violators using batched-gradient input.
///
/// `grad` is the full-feature gradient (one rmatvec, computed once per
/// outer-iter KKT pass) and `lips` is the precomputed coord-Lipschitz
/// cache. For each feature `j ∉ ws`, applies a prox-gradient step at
/// `(β_j, grad_j)` and reports `j` if the result would move `β_j` by
/// more than `tol`. Penalty-agnostic: uses the same `prox_coord` the
/// inner CD calls, so the boundary is consistent.
///
/// Per-feature cost is O(1) — no column reads in the verifier loop.
/// That makes the verifier ~1000× cheaper than the per-feature
/// `col_dot_weighted` variant it replaced.
fn find_kkt_violators_batched(
    penalty: &dyn Penalty,
    beta: ndarray::ArrayView1<'_, f64>,
    grad: ndarray::ArrayView1<'_, f64>,
    lips: &[f64],
    ws: &[usize],
    tol: f64,
) -> Vec<usize> {
    let p = grad.len();
    debug_assert_eq!(lips.len(), p);
    debug_assert_eq!(beta.len(), p);

    let mut violators = Vec::new();
    let mut ws_idx = 0usize;
    for j in 0..p {
        if ws_idx < ws.len() && ws[ws_idx] == j {
            ws_idx += 1;
            continue;
        }
        let lj = lips[j];
        if lj == 0.0 {
            continue;
        }
        let step = 1.0 / lj;
        let z = beta[j] - grad[j] * step;
        let prox_bj = penalty.prox_coord(j, z, step);
        if (prox_bj - beta[j]).abs() > tol {
            violators.push(j);
        }
    }
    violators
}

/// Fused IRLS + CD GLM solver — ncvreg's `src/glm.c::cdfit_glm` pattern.
///
/// Differs from [`prox_newton_solve`] by collapsing the per-outer-iter
/// "build full surrogate + run inner CD to tol" pattern into a single
/// fused loop where each iteration both refreshes the IRLS surrogate
/// components (`w`, `r`) at the current `eta` and performs ONE sweep
/// through the active set. This eliminates the upfront `O(n·p)` work
/// (`lips_arr` + `grad0`) that the classic solver pays per outer iter
/// — the per-feature `xwr` and `xwx` are computed lazily inside the
/// sweep, costing `O(n·|active|)` total. On the bench-shape
/// `logistic_mcp medium-sparse` problem this closes most of the
/// 6× wall-clock gap to ncvreg (target ~5s, vs classic skein 20s
/// and ncvreg 3.3s).
///
/// The function maintains state `(β, η, w, r)` where:
///   - `β` is the coefficient vector (updated in place)
///   - `η = X β` is the linear predictor (updated incrementally as
///     `η ← η + shift · X[:, j]` whenever a coordinate changes)
///   - `w` and `r` are the IRLS surrogate's per-sample weight and
///     working residual; refreshed from `η` at the start of each
///     iter via `GlmDatafit::refresh_surrogate_components`
///
/// Working-set strategy mirrors ncvreg's two-tier scan:
///   1. Inner loop iterates until `max_change < tol` on the current
///      active set `ws`.
///   2. After inner converges, scan all features outside `ws` for
///      KKT violators (using the same `find_kkt_violators_batched`
///      the classic solver uses).
///   3. Add violators to `ws` and re-enter step 1.
///   4. If no violators, the path-level optimum at this λ is reached.
///
/// **Precondition:** `glm.refresh_surrogate_components` must be
/// implemented (the default `unimplemented!()`). All of `BinomialLogit`,
/// `PoissonLog`, `CoxPH` provide it; other datafits do not and must
/// route through [`prox_newton_solve`] instead.
#[allow(clippy::too_many_arguments)]
pub fn prox_newton_fused_solve(
    design: &dyn DesignMatrix,
    glm: &dyn GlmDatafit,
    penalty: &dyn Penalty,
    init_beta: Array1<f64>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array1<f64>, ProxNewtonReport) {
    let n = design.n_samples();
    let p = design.n_features();
    let n_f = n as f64;
    debug_assert_eq!(init_beta.len(), p, "init_beta length must equal n_features");

    let mut beta = init_beta;
    let mut eta = design.matvec(beta.view());
    let mut w = Array1::<f64>::zeros(n);
    let mut r = Array1::<f64>::zeros(n);
    glm.refresh_surrogate_components(eta.view(), w.view_mut(), r.view_mut());

    let weights: Array1<f64> = penalty.weights().to_owned();

    // Seed the working set via the strong-rule screen on the initial
    // gradient at `warm`. Gradient = `-(1/n) · X^T (w · r)`, derived
    // lazily from one column-dot pass.
    let mut grad0 = Array1::<f64>::zeros(p);
    for j in 0..p {
        grad0[j] = -design.col_dot_weighted(j, w.view(), r.view()) / n_f;
    }
    let n_support = beta.iter().filter(|&&b| b != 0.0).count();
    let ws_size = (n_support * 2).max(PROX_NEWTON_P0).min(p);
    let mut ws = priority_rule_screen_with_grad(grad0.view(), weights.view(), beta.view(), ws_size);

    let mut total_iters = 0usize;
    let mut total_inner = 0usize;
    let mut converged = false;

    // Outer KKT-expansion loop (ncvreg's "strong set" tier).
    'outer: loop {
        // Inner fused loop: refresh surrogate, sweep active features,
        // check convergence.
        let mut inner_converged = false;
        for _ in 0..max_outer {
            total_iters += 1;
            total_inner += 1;
            glm.refresh_surrogate_components(eta.view(), w.view_mut(), r.view_mut());

            let mut max_change = 0.0_f64;
            for &j in &ws {
                let xwr = design.col_dot_weighted(j, w.view(), r.view());
                let xwx = design.col_sq_norm_weighted(j, w.view());
                let v = xwx / n_f;
                if v <= 0.0 {
                    continue;
                }
                let u = xwr / n_f + v * beta[j];
                // Translate to skein's `prox_coord(z, step)` API:
                //   skein z = β_j - step · grad_j
                //   step = 1 / v
                //   z = u / v
                let step = 1.0 / v;
                let z = u / v;
                let new_b = penalty.prox_coord(j, z, step);
                let shift = new_b - beta[j];
                if shift != 0.0 {
                    beta[j] = new_b;
                    // Incremental updates: r ← r − shift · X[:, j] and
                    // η ← η + shift · X[:, j]. Two column reads per
                    // changed coord — no combined primitive exists,
                    // but the column is cache-hot from the just-prior
                    // `col_dot_weighted` / `col_sq_norm_weighted`.
                    design.col_axpy(j, -shift, r.view_mut());
                    design.col_axpy(j, shift, eta.view_mut());
                    let change = shift.abs() * v.sqrt();
                    if change > max_change {
                        max_change = change;
                    }
                }
            }

            if max_change < cd_config.tol {
                inner_converged = true;
                break;
            }
        }

        if !inner_converged {
            // Hit max_outer without inner convergence on the current
            // ws. The classic solver's pattern is to keep iterating;
            // here we surface this as outer non-convergence and bail.
            break;
        }

        // KKT scan on features outside ws. Compute the full lips +
        // gradient ONLY at this expansion check (not per inner iter).
        let lips_arr = design.weighted_col_sq_norms(w.view());
        let lips: Vec<f64> = lips_arr.iter().map(|&v| v / n_f).collect();
        let mut grad_full = Array1::<f64>::zeros(p);
        for j in 0..p {
            grad_full[j] = -design.col_dot_weighted(j, w.view(), r.view()) / n_f;
        }
        let violators = find_kkt_violators_batched(
            penalty,
            beta.view(),
            grad_full.view(),
            &lips,
            &ws,
            outer_tol,
        );
        if violators.is_empty() {
            converged = true;
            break 'outer;
        }
        ws.extend(violators);
        ws.sort_unstable();
        ws.dedup();
    }

    let final_loss = glm.loss(design, beta.view());
    (
        beta,
        ProxNewtonReport {
            outer_iters: total_iters,
            outer_converged: converged,
            inner_iters: vec![1; total_inner],
            final_loss,
        },
    )
}

/// λ-path prox-Newton solve. Each row of the returned matrix is the β at
/// the corresponding λ; β warm-starts across the path. Auto-grid uses
/// `lambda_max` on the surrogate at `β = 0`.
#[allow(clippy::too_many_arguments)]
pub fn prox_newton_solve_path<F>(
    design: &dyn DesignMatrix,
    glm: &dyn GlmDatafit,
    make_penalty: F,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    explicit_lambdas: Option<Vec<f64>>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array2<f64>, ProxNewtonPathReport)
where
    F: Fn(f64) -> Box<dyn Penalty>,
{
    let p = design.n_features();

    let lambdas = match explicit_lambdas {
        Some(v) => v,
        None => {
            // λ_max from the surrogate at β = 0 (the entry point of the
            // KKT-at-zero argument is identical to LS when we evaluate it
            // against the local quadratic).
            let beta_zero = Array1::<f64>::zeros(p);
            let surrogate0 = glm.surrogate_at(design, beta_zero.view());
            let sample_pen = make_penalty(1.0);
            let lam_max = lambda_max(design, &surrogate0, sample_pen.weights());
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

    for (k, &lam) in lambdas.iter().enumerate() {
        let pen = make_penalty(lam);
        // Route through the screened variant — `lambda = Some(lam)` enables
        // gap-safe sphere screening + Anderson dual extrapolation on the
        // surrogate (celer's per-λ pattern). The standalone single-λ
        // `prox_newton_solve` keeps the legacy KKT-only path for callers
        // that don't have λ in hand.
        let (new_beta, report) = prox_newton_solve_screened(
            design,
            glm,
            &*pen,
            warm,
            cd_config,
            max_outer,
            outer_tol,
            Some(lam),
        );
        warm = new_beta;
        betas.row_mut(k).assign(&warm);
        outer_iters_out.push(report.outer_iters);
        outer_converged_out.push(report.outer_converged);
        let total_inner: usize = report.inner_iters.iter().sum();
        inner_iters_out.push(total_inner);
        final_losses_out.push(report.final_loss);
    }

    (
        betas,
        ProxNewtonPathReport {
            lambdas,
            outer_iters: outer_iters_out,
            outer_converged: outer_converged_out,
            inner_iters: inner_iters_out,
            final_losses: final_losses_out,
        },
    )
}

/// λ-path wrapper around [`prox_newton_fused_solve`]. Same signature
/// as [`prox_newton_solve_path`]; callers swap by name. Used by the
/// Python `solve_{logistic,poisson,cox}_{mcp,scad}_path` bindings
/// since M14f. ElasticNet GLM bindings stay on `prox_newton_solve_path`
/// (convex penalty, upfront `lips` amortizes fine).
#[allow(clippy::too_many_arguments)]
pub fn prox_newton_fused_solve_path<F>(
    design: &dyn DesignMatrix,
    glm: &dyn GlmDatafit,
    make_penalty: F,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    explicit_lambdas: Option<Vec<f64>>,
    cd_config: &CdConfig,
    max_outer: usize,
    outer_tol: f64,
) -> (Array2<f64>, ProxNewtonPathReport)
where
    F: Fn(f64) -> Box<dyn Penalty>,
{
    let p = design.n_features();

    let lambdas = match explicit_lambdas {
        Some(v) => v,
        None => {
            let beta_zero = Array1::<f64>::zeros(p);
            let surrogate0 = glm.surrogate_at(design, beta_zero.view());
            let sample_pen = make_penalty(1.0);
            let lam_max = lambda_max(design, &surrogate0, sample_pen.weights());
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

    for (k, &lam) in lambdas.iter().enumerate() {
        let pen = make_penalty(lam);
        let (new_beta, report) =
            prox_newton_fused_solve(design, glm, &*pen, warm, cd_config, max_outer, outer_tol);
        warm = new_beta;
        betas.row_mut(k).assign(&warm);
        outer_iters_out.push(report.outer_iters);
        outer_converged_out.push(report.outer_converged);
        let total_inner: usize = report.inner_iters.iter().sum();
        inner_iters_out.push(total_inner);
        final_losses_out.push(report.final_loss);
    }

    (
        betas,
        ProxNewtonPathReport {
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
    use crate::datafit::{BinomialLogit, CoxPH, Huber, PoissonLog};
    use crate::design::{DenseMatrix, Standardized};
    use crate::penalty::{Mcp, Scad};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    fn logistic_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>) {
        // Sparse-truth: only first 3 features active. 100 samples, 10 features.
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
        let eta = x.dot(&true_beta);
        // Generate y by sampling Bernoulli(sigmoid(η)) — use the xorshift
        // deterministic stream so the test is reproducible.
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let p_i = 1.0 / (1.0 + (-eta[i]).exp());
            // Uniform [0,1] from sample() ∈ [-1,1] mapped: (sample()+1)/2.
            let u = (sample() + 1.0) * 0.5;
            y[i] = if u < p_i { 1.0 } else { 0.0 };
        }
        (DenseMatrix::new(x), y, true_beta)
    }

    #[test]
    fn prox_newton_at_lambda_max_returns_zero() {
        let (design, y, _) = logistic_problem(1);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let beta_zero = Array1::<f64>::zeros(p);
        let surrogate0 = glm.surrogate_at(&design, beta_zero.view());
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &surrogate0, weights.view());

        let (beta, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lam_max, 100.0, p),
            beta_zero.clone(),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
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
    fn prox_newton_recovers_signal_at_small_lambda() {
        let (design, y, true_beta) = logistic_problem(2);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let (beta, report) = prox_newton_solve(
            &design,
            &glm,
            // Mcp at γ=1e6 ≈ lasso (convex inner, easier to converge).
            &Mcp::new(0.005, 1e6, p),
            Array1::<f64>::zeros(p),
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            20,
            1e-7,
        );
        assert!(
            report.outer_converged,
            "prox-Newton should converge in ≤ 20 outer iterations (got {})",
            report.outer_iters
        );
        for k in 0..3 {
            assert_eq!(
                beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch: β = {}",
                k,
                beta[k]
            );
        }
    }

    #[test]
    fn prox_newton_path_lambdas_decreasing_with_correct_shape() {
        let (design, y, _) = logistic_problem(3);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let (betas, report) = prox_newton_solve_path(
            &design,
            &glm,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            10,
            1e-2,
            None,
            &CdConfig {
                max_iter: 1000,
                tol: 1e-8,
                acceleration: None,
            },
            10,
            1e-7,
        );
        assert_eq!(betas.shape(), &[10, p]);
        assert_eq!(report.lambdas.len(), 10);
        for k in 1..report.lambdas.len() {
            assert!(report.lambdas[k] < report.lambdas[k - 1]);
        }
    }

    #[test]
    fn prox_newton_path_recovers_truth_at_smallest_lambda() {
        let (design, y, true_beta) = logistic_problem(4);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let (betas, report) = prox_newton_solve_path(
            &design,
            &glm,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            25,
            1e-3,
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
        for k in 0..3 {
            assert_eq!(
                last_beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch at smallest λ: β = {}",
                k,
                last_beta[k]
            );
        }
    }

    /// Sparse-truth Poisson regression problem: only first 3 features
    /// active. y ~ Poisson(exp(η)) sampled with Knuth's algorithm using
    /// a deterministic xorshift stream. Counts are typically 0..6 with
    /// occasional larger values.
    fn poisson_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>) {
        let n = 300;
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
        true_beta[0] = 0.7;
        true_beta[1] = -0.5;
        true_beta[2] = 0.4;
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
        (DenseMatrix::new(x), y, true_beta)
    }

    #[test]
    fn poisson_prox_newton_at_lambda_max_returns_zero() {
        let (design, y, _) = poisson_problem(1);
        let glm = PoissonLog::new(y);
        let p = design.n_features();
        let beta_zero = Array1::<f64>::zeros(p);
        let surrogate0 = glm.surrogate_at(&design, beta_zero.view());
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &surrogate0, weights.view());

        let (beta, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lam_max, 100.0, p),
            beta_zero.clone(),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta[j], 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn poisson_prox_newton_recovers_signal_at_small_lambda() {
        let (design, y, true_beta) = poisson_problem(2);
        let glm = PoissonLog::new(y);
        let p = design.n_features();
        // γ=1e6 ⇒ ≈ lasso: convex inner problem, easier to converge.
        let (beta, report) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(0.005, 1e6, p),
            Array1::<f64>::zeros(p),
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        assert!(
            report.outer_converged,
            "prox-Newton should converge in ≤ 30 outer iterations (got {})",
            report.outer_iters
        );
        for k in 0..3 {
            assert_eq!(
                beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch: β = {}",
                k,
                beta[k]
            );
        }
    }

    /// Sparse-truth Cox PH problem with exponential baseline hazard.
    /// Sample T_i ~ Exp(exp(η_i)), C_i ~ Exp(0.5); observe t = min(T,C),
    /// δ = 1[T ≤ C]. Sample stream is deterministic xorshift so the
    /// test is reproducible.
    fn cox_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>, Array1<f64>) {
        let n = 300;
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
        true_beta[0] = 0.7;
        true_beta[1] = -0.5;
        true_beta[2] = 0.3;
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
        (DenseMatrix::new(x), time, event, true_beta)
    }

    #[test]
    fn cox_prox_newton_at_lambda_max_returns_zero() {
        let (design, time, event, _) = cox_problem(1);
        let glm = CoxPH::new(time, event);
        let p = design.n_features();
        let beta_zero = Array1::<f64>::zeros(p);
        let surrogate0 = glm.surrogate_at(&design, beta_zero.view());
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &surrogate0, weights.view());

        let (beta, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lam_max, 100.0, p),
            beta_zero.clone(),
            &CdConfig {
                max_iter: 200,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta[j], 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn cox_prox_newton_recovers_signal_at_small_lambda() {
        let (design, time, event, true_beta) = cox_problem(2);
        let glm = CoxPH::new(time, event);
        let p = design.n_features();
        let (beta, report) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(0.005, 1e6, p),
            Array1::<f64>::zeros(p),
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            30,
            1e-7,
        );
        assert!(
            report.outer_converged,
            "prox-Newton should converge in ≤ 30 outer iterations (got {})",
            report.outer_iters
        );
        for k in 0..3 {
            assert_eq!(
                beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch: β = {}",
                k,
                beta[k]
            );
        }
    }

    #[test]
    fn cox_prox_newton_path_recovers_truth_at_smallest_lambda() {
        let (design, time, event, true_beta) = cox_problem(3);
        let glm = CoxPH::new(time, event);
        let p = design.n_features();
        let (betas, report) = prox_newton_solve_path(
            &design,
            &glm,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            25,
            1e-3,
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
        for k in 0..3 {
            assert_eq!(
                last_beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch at smallest λ: β = {}",
                k,
                last_beta[k]
            );
        }
    }

    /// Build a pre-scaled `DenseMatrix` reference: `X · diag(1/s)`. The
    /// solver run on this reference must match the run on the
    /// `Standardized<DenseMatrix>` wrapper with the same `s`, since both
    /// represent the same problem in standardized β-space.
    fn pre_scaled_dense(x: &Array2<f64>, scales: &Array1<f64>) -> DenseMatrix {
        let mut x_scaled = x.clone();
        for j in 0..x.ncols() {
            let s = scales[j];
            for i in 0..x.nrows() {
                x_scaled[[i, j]] /= s;
            }
        }
        DenseMatrix::new(x_scaled)
    }

    /// Logistic prox-Newton path on `Standardized<DenseMatrix>` matches
    /// the same solver on a pre-scaled `DenseMatrix` reference at every
    /// λ, within 1e-7. Validates that the prox-Newton outer loop
    /// composes transparently with the lazy column-scaling wrapper —
    /// the prerequisite for sparse + standardize on GLMs (M4.3 follow-up).
    #[test]
    fn logistic_prox_newton_path_through_standardized_matches_pre_scaled() {
        let (design_raw, y, _) = logistic_problem(7);
        let x = design_raw.view().to_owned();
        let p = x.ncols();
        let scales = Array1::from(vec![1.5, 0.7, 2.3, 0.9, 1.1, 1.8, 0.6, 2.0, 1.3, 0.8]);

        let dense_ref = pre_scaled_dense(&x, &scales);
        let std_design = Standardized::new(DenseMatrix::new(x), scales.clone());

        let glm_a = BinomialLogit::new(y.clone());
        let glm_b = BinomialLogit::new(y);

        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let make_pen = |lam: f64| -> Box<dyn Penalty> { Box::new(Mcp::new(lam, 3.0, p)) };

        let (betas_ref, report_ref) = prox_newton_solve_path(
            &dense_ref, &glm_a, make_pen, 12, 1e-2, None, &cd_cfg, 20, 1e-8,
        );
        let (betas_std, report_std) = prox_newton_solve_path(
            &std_design,
            &glm_b,
            make_pen,
            12,
            1e-2,
            None,
            &cd_cfg,
            20,
            1e-8,
        );

        assert_eq!(report_ref.lambdas.len(), report_std.lambdas.len());
        for k in 0..report_ref.lambdas.len() {
            assert_abs_diff_eq!(
                report_ref.lambdas[k],
                report_std.lambdas[k],
                epsilon = 1e-12
            );
        }
        assert_eq!(betas_ref.shape(), betas_std.shape());
        for k in 0..betas_ref.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_ref[[k, j]], betas_std[[k, j]], epsilon = 1e-7);
            }
        }
    }

    /// Poisson prox-Newton path through `Standardized<DenseMatrix>` vs
    /// pre-scaled reference. Same equivalence argument as the logistic
    /// case — the GLM surrogate is built off `design.matvec(β)`, which
    /// the wrapper redirects to `base.matvec(β/s)`.
    #[test]
    fn poisson_prox_newton_path_through_standardized_matches_pre_scaled() {
        let (design_raw, y, _) = poisson_problem(7);
        let x = design_raw.view().to_owned();
        let p = x.ncols();
        let scales = Array1::from(vec![1.4, 0.8, 2.1, 1.0, 0.9, 1.7, 0.7, 1.9, 1.2, 0.85]);

        let dense_ref = pre_scaled_dense(&x, &scales);
        let std_design = Standardized::new(DenseMatrix::new(x), scales.clone());

        let glm_a = PoissonLog::new(y.clone());
        let glm_b = PoissonLog::new(y);

        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let make_pen = |lam: f64| -> Box<dyn Penalty> { Box::new(Mcp::new(lam, 3.0, p)) };

        let (betas_ref, _) = prox_newton_solve_path(
            &dense_ref, &glm_a, make_pen, 10, 1e-2, None, &cd_cfg, 30, 1e-8,
        );
        let (betas_std, _) = prox_newton_solve_path(
            &std_design,
            &glm_b,
            make_pen,
            10,
            1e-2,
            None,
            &cd_cfg,
            30,
            1e-8,
        );

        assert_eq!(betas_ref.shape(), betas_std.shape());
        for k in 0..betas_ref.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_ref[[k, j]], betas_std[[k, j]], epsilon = 1e-7);
            }
        }
    }

    /// Cox prox-Newton path through `Standardized<DenseMatrix>` vs
    /// pre-scaled reference. Cox has no intercept augmentation, so the
    /// wrapper is applied directly to the user matrix.
    #[test]
    fn cox_prox_newton_path_through_standardized_matches_pre_scaled() {
        let (design_raw, time, event, _) = cox_problem(7);
        let x = design_raw.view().to_owned();
        let p = x.ncols();
        let scales = Array1::from(vec![1.6, 0.75, 2.0, 0.95, 1.1, 1.5, 0.65, 1.85, 1.25, 0.9]);

        let dense_ref = pre_scaled_dense(&x, &scales);
        let std_design = Standardized::new(DenseMatrix::new(x), scales.clone());

        let glm_a = CoxPH::new(time.clone(), event.clone());
        let glm_b = CoxPH::new(time, event);

        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let make_pen = |lam: f64| -> Box<dyn Penalty> { Box::new(Mcp::new(lam, 3.0, p)) };

        let (betas_ref, _) = prox_newton_solve_path(
            &dense_ref, &glm_a, make_pen, 10, 1e-2, None, &cd_cfg, 30, 1e-8,
        );
        let (betas_std, _) = prox_newton_solve_path(
            &std_design,
            &glm_b,
            make_pen,
            10,
            1e-2,
            None,
            &cd_cfg,
            30,
            1e-8,
        );

        assert_eq!(betas_ref.shape(), betas_std.shape());
        for k in 0..betas_ref.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_ref[[k, j]], betas_std[[k, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn poisson_prox_newton_path_recovers_truth_at_smallest_lambda() {
        let (design, y, true_beta) = poisson_problem(3);
        let glm = PoissonLog::new(y);
        let p = design.n_features();
        let (betas, report) = prox_newton_solve_path(
            &design,
            &glm,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            25,
            1e-3,
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
        for k in 0..3 {
            assert_eq!(
                last_beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch at smallest λ: β = {}",
                k,
                last_beta[k]
            );
        }
    }

    // ---- Huber regression (M3.7) -----------------------------------

    fn huber_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Array1<f64>) {
        // Sparse truth, then add a few large outliers to motivate Huber.
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
        let signal = x.dot(&true_beta);
        let mut y = signal;
        for i in 0..n {
            y[i] += 0.1 * sample();
        }
        // Inject 10 large outliers (5% contamination) at amplitude 20× noise.
        for i in (0..10).map(|k| k * 17 % n) {
            y[i] += 5.0 * sample().signum();
        }
        (DenseMatrix::new(x), y, true_beta)
    }

    #[test]
    fn huber_prox_newton_recovers_signal_at_small_lambda() {
        let (design, y, true_beta) = huber_problem(11);
        // δ ≈ 1.345·σ recovers the 95%-efficient setting at the normal.
        let glm = Huber::new(y, 1.345);
        let p = design.n_features();
        let (beta, report) = prox_newton_solve(
            &design,
            &glm,
            // Mcp at γ=1e6 ≈ lasso (convex inner, easier to converge).
            &Mcp::new(0.01, 1e6, p),
            Array1::<f64>::zeros(p),
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            20,
            1e-7,
        );
        assert!(
            report.outer_converged,
            "Huber prox-Newton should converge in ≤ 20 outer iters (got {})",
            report.outer_iters
        );
        for k in 0..3 {
            assert_eq!(
                beta[k].signum(),
                true_beta[k].signum(),
                "feature {} sign mismatch: β = {}",
                k,
                beta[k]
            );
        }
    }

    #[test]
    fn huber_prox_newton_at_lambda_max_returns_zero() {
        let (design, y, _) = huber_problem(12);
        let glm = Huber::new(y, 1.345);
        let p = design.n_features();
        // λ_max from the Huber surrogate at β = 0.
        let surr0 = glm.surrogate_at(&design, Array1::<f64>::zeros(p).view());
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &surr0, weights.view());
        let (beta, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lam_max, 1e6, p),
            Array1::<f64>::zeros(p),
            &CdConfig {
                max_iter: 1000,
                tol: 1e-8,
                acceleration: None,
            },
            10,
            1e-7,
        );
        for k in 0..p {
            assert_abs_diff_eq!(beta[k], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn huber_large_delta_matches_least_squares() {
        // δ chosen far above the largest residual ⇒ Huber ≡ LS, so the
        // prox-Newton fit should match a direct LS solve at the same λ.
        let (design, y, _) = huber_problem(13);
        let p = design.n_features();
        let glm = Huber::new(y.clone(), 1e3);
        let (beta_huber, report) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(0.01, 1e6, p),
            Array1::<f64>::zeros(p),
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            10,
            1e-8,
        );
        assert!(report.outer_converged);
        // Direct LS at the same λ via cd_solve.
        use crate::datafit::LeastSquares;
        use crate::solver::cd::cd_solve;
        let ls = LeastSquares::new(y);
        let (beta_ls, _) = cd_solve(
            &design,
            &ls,
            &Mcp::new(0.01, 1e6, p),
            &CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
        );
        for k in 0..p {
            assert_abs_diff_eq!(beta_huber[k], beta_ls[k], epsilon = 1e-5);
        }
    }

    /// Pins the M14e bloat fix. At small λ the IRLS surrogate
    /// `step = 1/L_jj` exceeds γ=3 on most features (saturated samples
    /// drive `w_i → W_FLOOR = 1e-4`, shrinking L_jj). Vanilla MCP's
    /// firm-threshold returns `z` unchanged in the wide saturation
    /// band `[γλ, γλ·step]` → features pile up at their warm value →
    /// active set bloats to ~80% of p. ncvreg's v-scaled MCP prox
    /// (shipped M14e in `prox::mcp_prox`) shrinks throughout this
    /// band, so the support stays close to the planted truth.
    fn logistic_problem_medium(seed: u64) -> (DenseMatrix, Array1<f64>) {
        let n = 500;
        let p = 100;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        for k in 0..10 {
            true_beta[k] = if k % 2 == 0 { 1.0 } else { -1.0 };
        }
        let eta = x.dot(&true_beta);
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let p_i = 1.0 / (1.0 + (-eta[i]).exp());
            let u = (sample() + 1.0) * 0.5;
            y[i] = if u < p_i { 1.0 } else { 0.0 };
        }
        (DenseMatrix::new(x), y)
    }

    #[test]
    fn logistic_mcp_path_active_set_stays_bounded_at_small_lambda() {
        let (design, y) = logistic_problem_medium(7);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();

        let (betas, report) = prox_newton_solve_path(
            &design,
            &glm,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            50,
            5e-2,
            None,
            &CdConfig {
                max_iter: 1000,
                tol: 1e-7,
                acceleration: Some(5),
            },
            50,
            1e-7,
        );

        // Allow a handful of transitional λs to not converge (the
        // first IRLS surrogate at the convex→non-convex boundary can
        // bounce). Pre-M14e the entire tail failed to converge; with
        // the v-scaled prox the tail converges cleanly and only the
        // crossing region might wobble.
        let unconverged = report.outer_converged.iter().filter(|&&c| !c).count();
        assert!(
            unconverged <= 5,
            "expected ≤ 5 un-converged λs (transitional); got {} (out of 50). converged: {:?}",
            unconverged,
            report.outer_converged
        );

        // True support is 10. The empirical post-M14e count on this
        // tiny problem is ~56; ncvreg gets a similar count at this
        // scale (noisier per-feature than the bench-shape n=10k/p=1k
        // problem where both algorithms converge to ~107 active).
        // Bound the assertion at 65 — generous headroom over the
        // observed ~56 — to gate against regressions to the pre-M14e
        // ~80+ baseline without false-failing on platform noise.
        let last_row = betas.row(betas.nrows() - 1);
        let active = last_row.iter().filter(|&&b| b != 0.0).count();
        assert!(
            active <= 65,
            "expected ≤ 65 active features at λ_min; got {} \
             (pre-M14e: ~80, ncvreg at this scale: similar to skein)",
            active
        );
    }

    /// SCAD analog of the MCP bloat-fix gate. Pre-M14e, SCAD had the
    /// same kind of degeneracy as MCP for GLM IRLS surrogates: the
    /// middle-branch denominator `1 − step/(a−1)` flips sign when
    /// `step ≥ a − 1` (≈ 2.7 for default `a = 3.7`), which IRLS step
    /// `1/L_jj` routinely exceeds when samples saturate. The if-else
    /// cascade also degenerated because `(1+step)·λ > a·λ` once
    /// `step > a − 1`, eliminating the middle (SCAD-quadratic)
    /// region entirely and forcing features above the lasso boundary
    /// to land in the identity branch (pinned at warm β unchanged).
    /// M14e's v-scaled SCAD prox in `prox::scad_prox` fixes both
    /// issues. On the bench-shape problem (n=10k, p=1k)
    /// logistic_scad now matches logistic_mcp almost exactly: 108
    /// vs 107 active, 19.4s vs 20.8s.
    #[test]
    fn logistic_scad_path_active_set_stays_bounded_at_small_lambda() {
        let (design, y) = logistic_problem_medium(11);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();

        let (betas, report) = prox_newton_solve_path(
            &design,
            &glm,
            |lam| Box::new(Scad::new(lam, 3.7, p)),
            50,
            5e-2,
            None,
            &CdConfig {
                max_iter: 1000,
                tol: 1e-7,
                acceleration: Some(5),
            },
            50,
            1e-7,
        );

        let unconverged = report.outer_converged.iter().filter(|&&c| !c).count();
        assert!(
            unconverged <= 5,
            "expected ≤ 5 un-converged λs; got {} (out of 50). converged: {:?}",
            unconverged,
            report.outer_converged
        );

        // Bound looser than MCP's (≤ 65) because SCAD's middle
        // quadratic region shrinks less aggressively by design — the
        // penalty curvature is gentler in the transition band. At
        // this small scale the empirical post-M14e count is ~72;
        // bound at 85 to gate against a regression toward the pre-M14e
        // ~p=100 baseline.
        let last_row = betas.row(betas.nrows() - 1);
        let active = last_row.iter().filter(|&&b| b != 0.0).count();
        assert!(
            active <= 85,
            "expected ≤ 85 active features at λ_min; got {} (pre-M14e: ~p=100)",
            active
        );
    }

    /// M14f cross-solver agreement: the fused IRLS+CD solver should
    /// converge to a β within tight tolerance of the classic solver on
    /// the same problem. The math problem solved is identical (v-scaled
    /// MCP via M14e); only the iteration strategy differs.
    #[test]
    fn fused_solve_matches_classic_logistic_mcp() {
        let (design, y, _) = logistic_problem(3);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let lambda = 0.05;
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta_classic, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lambda, 3.0, p),
            Array1::<f64>::zeros(p),
            &cfg,
            50,
            1e-8,
        );
        let (beta_fused, _) = prox_newton_fused_solve(
            &design,
            &glm,
            &Mcp::new(lambda, 3.0, p),
            Array1::<f64>::zeros(p),
            &cfg,
            500, // fused inner is cheaper per iter → can afford more
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta_fused[j], beta_classic[j], epsilon = 1e-4);
        }
    }

    #[test]
    fn fused_solve_matches_classic_poisson_mcp() {
        let (design, y, _) = poisson_problem(5);
        let glm = PoissonLog::new(y);
        let p = design.n_features();
        let lambda = 0.05;
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta_classic, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lambda, 3.0, p),
            Array1::<f64>::zeros(p),
            &cfg,
            50,
            1e-8,
        );
        let (beta_fused, _) = prox_newton_fused_solve(
            &design,
            &glm,
            &Mcp::new(lambda, 3.0, p),
            Array1::<f64>::zeros(p),
            &cfg,
            500,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta_fused[j], beta_classic[j], epsilon = 1e-4);
        }
    }

    #[test]
    fn fused_solve_matches_classic_cox_mcp() {
        let (design, time, event, _) = cox_problem(7);
        let glm = CoxPH::new(time, event);
        let p = design.n_features();
        let lambda = 0.05;
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let (beta_classic, _) = prox_newton_solve(
            &design,
            &glm,
            &Mcp::new(lambda, 3.0, p),
            Array1::<f64>::zeros(p),
            &cfg,
            50,
            1e-8,
        );
        let (beta_fused, _) = prox_newton_fused_solve(
            &design,
            &glm,
            &Mcp::new(lambda, 3.0, p),
            Array1::<f64>::zeros(p),
            &cfg,
            500,
            1e-8,
        );
        for j in 0..p {
            assert_abs_diff_eq!(beta_fused[j], beta_classic[j], epsilon = 1e-4);
        }
    }

    /// Mirror of `solver::path::tests::solve_path_screening_on_matches_screening_off_within_tol`
    /// for the GLM path. The screening-enabled
    /// `prox_newton_solve_screened(lambda=Some(λ))` path must produce
    /// numerically equivalent coefficients to the legacy KKT-only path
    /// (`lambda=None`) at tight tol. If a `gap_safe` formula slips below
    /// the double-precision floor, the screened loop will sweep
    /// `max_iter × n_lambdas` per test and starve the runner — exactly
    /// the pathology CLAUDE.md's solver-change pre-flight warns about.
    #[test]
    fn prox_newton_screening_matches_no_screening_within_tol() {
        let (design, y, _) = logistic_problem(7);
        let glm = BinomialLogit::new(y);
        let p = design.n_features();
        let cfg = CdConfig {
            max_iter: 200,
            tol: 1e-10,
            acceleration: None,
        };
        let lambdas = [0.10_f64, 0.05, 0.02, 0.01];
        let mut betas_off = Array2::<f64>::zeros((lambdas.len(), p));
        let mut warm_off = Array1::<f64>::zeros(p);
        for (k, &lam) in lambdas.iter().enumerate() {
            // Force the legacy KKT-only path (`lambda = None`).
            let (b, _) = prox_newton_solve_screened(
                &design,
                &glm,
                &Mcp::new(lam, 1e6, p), // γ → ∞ ⇒ pure L1 — well-conditioned
                warm_off,
                &cfg,
                30,
                1e-9,
                None,
            );
            warm_off = b.clone();
            betas_off.row_mut(k).assign(&b);
        }
        let mut betas_on = Array2::<f64>::zeros((lambdas.len(), p));
        let mut warm_on = Array1::<f64>::zeros(p);
        for (k, &lam) in lambdas.iter().enumerate() {
            // Screened path (`lambda = Some(lam)` → gap-safe screening
            // + Anderson dual extrapolation on the surrogate enabled).
            let (b, _) = prox_newton_solve_screened(
                &design,
                &glm,
                &Mcp::new(lam, 1e6, p),
                warm_on,
                &cfg,
                30,
                1e-9,
                Some(lam),
            );
            warm_on = b.clone();
            betas_on.row_mut(k).assign(&b);
        }
        for k in 0..lambdas.len() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_on[[k, j]], betas_off[[k, j]], epsilon = 5e-6);
            }
        }
    }
}
