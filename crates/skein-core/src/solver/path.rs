//! λ-path solver for separable penalties.
//!
//! Computes `lambda_max` (the smallest λ at which β = 0 is optimal), builds
//! a geometric grid down to `lambda_min_ratio · lambda_max`, and warm-starts
//! CD across the grid. The user supplies a `make_penalty` builder so we can
//! re-instantiate the penalty per λ without coupling this module to any one
//! penalty type.
//!
//! Currently assumes LS-style scaling for the gradient (`∂_j L = X_jᵀ r / n`
//! at β = 0). When non-LS datafits land in M3, `lambda_max` will move behind
//! a `Datafit::coord_grad_at_zero` accessor and become datafit-agnostic.

use crate::datafit::Datafit;
use crate::design::DesignMatrix;
use crate::penalty::Penalty;
use crate::solver::cd::{cd_solve_subset, solve_small, CdConfig, CdReport};
use ndarray::{Array1, Array2, ArrayView1};

/// Smallest λ at which β = 0 satisfies the KKT conditions for a separable
/// L1-like penalty (lasso, MCP, SCAD: subdifferential at 0 is `λ w_j [-1, 1]`).
///
/// Computed at β = 0, so the residual is `-y` and `∂_j L = -X_jᵀ y / n`.
/// Result: `max_{j: w_j > 0} |X_jᵀ y| / (n · w_j)`.
///
/// Features with `w_j = 0` are unpenalized; β = 0 is not an optimum for them
/// so they don't contribute to `lambda_max`. Caller should fit unpenalized
/// features first when present (not yet implemented — v0.1 scope).
pub fn lambda_max(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    weights: ArrayView1<f64>,
) -> f64 {
    let p = design.n_features();
    let zero_beta = Array1::<f64>::zeros(p);
    let r0 = datafit.init_residual(design, zero_beta.view());
    let mut max_g = 0.0_f64;
    for j in 0..p {
        let w = weights[j];
        if w <= 0.0 {
            continue;
        }
        let g = datafit.coord_grad(design, j, r0.view()).abs() / w;
        if g > max_g {
            max_g = g;
        }
    }
    max_g
}

/// Geometric grid of `n_lambdas` values from `lambda_max` down to
/// `lambda_min_ratio · lambda_max` (both endpoints included).
///
/// `n_lambdas == 1` returns `[lambda_max]`. `lambda_min_ratio` must be in
/// `(0, 1]`.
pub fn lambda_grid(lambda_max: f64, n_lambdas: usize, lambda_min_ratio: f64) -> Vec<f64> {
    if n_lambdas == 0 {
        return Vec::new();
    }
    if n_lambdas == 1 {
        return vec![lambda_max];
    }
    let log_max = lambda_max.ln();
    let log_min = (lambda_min_ratio * lambda_max).ln();
    let denom = (n_lambdas - 1) as f64;
    (0..n_lambdas)
        .map(|k| {
            let t = k as f64 / denom;
            (log_max + t * (log_min - log_max)).exp()
        })
        .collect()
}

/// Per-λ working-set screening strategy.
///
/// `Strong` is the Tibshirani sequential strong rule, applicable to any
/// separable L1-like penalty (including non-convex MCP/SCAD). `GapSafe` is
/// the Fercoq–Gramfort–Salmon sphere screen, *provably* tighter than
/// `Strong` but only valid for LS with a convex separable penalty (lasso /
/// large-γ MCP). The KKT-verification loop runs after either rule, so a
/// mismatched penalty/datafit/screening combo is corrected, just less
/// efficiently than its native rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Screening {
    Off,
    #[default]
    Strong,
    GapSafe,
}

#[derive(Debug, Clone)]
pub struct PathConfig {
    /// Number of λ values to sweep when `lambdas` is `None`.
    pub n_lambdas: usize,
    /// `lambda_min / lambda_max` when `lambdas` is `None`.
    pub lambda_min_ratio: f64,
    /// Explicit λ values; overrides `n_lambdas` / `lambda_min_ratio`.
    pub lambdas: Option<Vec<f64>>,
    /// CD config used at each λ.
    pub cd: CdConfig,
    /// Working-set screening strategy. Default `Strong`.
    pub screening: Screening,
    /// Initial working-set size when `Screening::Strong` is used and
    /// the previous-λ support is empty (cold start at λ_max). Mirrors
    /// celer's `p0` and skglm's `p0`. Larger values approach the
    /// "full sweep" behaviour the strong rule used to fall back to;
    /// smaller values trade off more KKT-verify passes for cheaper
    /// cold-start sweeps. Default `10` matches skglm.
    pub p0: usize,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            n_lambdas: 100,
            lambda_min_ratio: 1e-3,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::default(),
            p0: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathReport {
    pub lambdas: Vec<f64>,
    pub iters: Vec<usize>,
    pub converged: Vec<bool>,
    pub final_objs: Vec<f64>,
    /// Final working-set size at each λ (post KKT-verification loop).
    pub working_set_sizes: Vec<usize>,
    /// Number of outer KKT-loop passes at each λ. `1` means the strong rule's
    /// initial set was already KKT-correct and no expansion was needed.
    pub kkt_passes: Vec<usize>,
}

/// Solve a separable-penalty problem along a λ-path with warm starts.
///
/// Returns coefficients of shape `(n_lambdas, n_features)`; row `k` is the
/// solution at `report.lambdas[k]`. The grid is decreasing in λ.
pub fn solve_path<F>(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    make_penalty: F,
    config: &PathConfig,
) -> (Array2<f64>, PathReport)
where
    F: Fn(f64) -> Box<dyn Penalty>,
{
    let p = design.n_features();

    let lambdas = match &config.lambdas {
        Some(v) => v.clone(),
        None => {
            // Penalty weights are λ-independent; sample at any λ to read them.
            let sample = make_penalty(1.0);
            let lam_max = lambda_max(design, datafit, sample.weights());
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

    let mut warm = Array1::<f64>::zeros(p);
    let mut prev_residual: Option<Array1<f64>> = None;

    for (k, &lam) in lambdas.iter().enumerate() {
        let pen = make_penalty(lam);
        let weights: Array1<f64> = pen.weights().to_owned();

        // Initial working set per screening strategy.
        // - Off: full feature set, single pass.
        // - Strong: priority-based active-set rule (celer/skglm pattern).
        //   Ranks features by KKT-violation magnitude `|grad_j| / w_j` and
        //   picks the top `max(p0, 2 × |support|)`. Active and
        //   unpenalised features are pinned in. At λ_max with no prior
        //   support this gives a `p0`-sized WS instead of the previous
        //   "fall back to full feature set"; the KKT verifier catches
        //   anything missed.
        // - GapSafe: gap-safe sphere screen; works at every λ, including
        //   the first (uses the cold-start residual = -y).
        let mut ws: Vec<usize> = match config.screening {
            Screening::Off => (0..p).collect(),
            Screening::Strong => {
                // Use the previous λ's residual when available; cold-start
                // at λ_0 falls back to `r = init_residual(0)` = `−y` for
                // LS. Skipping the cold-residual matvec for k > 0 is the
                // whole point of warm-starting.
                let r_owned;
                let r_view = if let Some(pr) = &prev_residual {
                    pr.view()
                } else {
                    r_owned = datafit.init_residual(design, warm.view());
                    prev_residual = Some(r_owned);
                    prev_residual.as_ref().unwrap().view()
                };
                priority_rule_screen(
                    design,
                    datafit,
                    r_view,
                    weights.view(),
                    warm.view(),
                    config.p0,
                )
            }
            Screening::GapSafe => {
                let res_view = if k == 0 {
                    // Cold start at this λ; warm β = 0 ⇒ residual = -y.
                    let r = datafit.init_residual(design, warm.view());
                    prev_residual = Some(r);
                    prev_residual.as_ref().unwrap().view()
                } else {
                    prev_residual.as_ref().unwrap().view()
                };
                gap_safe_screen(design, datafit, res_view, warm.view(), weights.view(), lam)
            }
        };

        // Cache per-coord Lipschitz constants once per λ. For unweighted
        // LS this is an O(1) table lookup per call, but weighted-LS /
        // GLM datafits compute it via a column scan, so caching avoids
        // O(p · n) work when we evaluate the prox-gradient distance over
        // the full feature set in the verifier loop below.
        let coord_lipschitz: Vec<f64> = (0..p)
            .map(|j| datafit.coord_lipschitz(design, j))
            .collect();

        let mut passes = 0usize;
        let mut inner_cd_cfg = config.cd.clone();
        let mut prev_outer_pgd: f64 = f64::INFINITY;

        // Dual extrapolation history (celer pattern). Stores the last
        // `K_MAX` (β, r) pairs from this λ's outer KKT loop. Anderson
        // on the residual sequence yields a candidate dual point whose
        // dual obj — when feasible — is often a tighter lower bound
        // than the naive `θ_naive = -r/n`, shrinking the gap. We
        // maintain β alongside r so the dual obj formula
        // `D = ‖r‖²/n · scale·(1−scale/2) − scale · βᵀ grad` still
        // applies after extrapolation: linear combinations of (β, r)
        // with coefficients summing to 1 stay self-consistent
        // (`r_acc = X β_acc − y` automatically), so we don't need to
        // expose y on the `Datafit` trait.
        const DUAL_HISTORY_MAX: usize = 6;
        let mut beta_history: Vec<Array1<f64>> = Vec::with_capacity(DUAL_HISTORY_MAX);
        let mut residual_history: Vec<Array1<f64>> = Vec::with_capacity(DUAL_HISTORY_MAX);
        let mut best_dual_obj: f64 = f64::NEG_INFINITY;

        let (final_residual, last_report): (Array1<f64>, CdReport) = loop {
            passes += 1;

            // Adaptive inner tolerance, celer/skglm style. We use the
            // prox-gradient distance from the previous outer pass —
            // same units as `config.cd.tol`, penalty-agnostic, identical
            // to skglm's `dist_fix_point_cd`. As `prev_outer_pgd`
            // shrinks toward `tol`, `inner_tol` converges to `tol` and
            // the final β meets the user's request. First pass has no
            // prior PGD, so relax 10× — enough to skip the last sweep
            // on warm-started λs while keeping the cold-start λ_max
            // iteration tight enough to not need extra outer passes.
            inner_cd_cfg.tol = if prev_outer_pgd.is_finite() {
                config.cd.tol.max(0.3 * prev_outer_pgd)
            } else {
                config.cd.tol * 10.0
            };

            let (new_beta, r, report) =
                cd_solve_subset(warm, &ws, design, datafit, &*pen, &inner_cd_cfg);
            warm = new_beta;

            // Without screening, the WS is the full set; one pass and
            // we trust the inner CD's convergence.
            if matches!(config.screening, Screening::Off) {
                break (r, report);
            }

            // Push current (β, r) into dual-extrapolation history. Cap
            // to DUAL_HISTORY_MAX entries (drop the oldest when full).
            if beta_history.len() == DUAL_HISTORY_MAX {
                beta_history.remove(0);
                residual_history.remove(0);
            }
            beta_history.push(warm.clone());
            residual_history.push(r.clone());

            // Try Anderson on the residual sequence. When successful,
            // gives `(β_acc, r_acc)` consistent with `r = Xβ − y`.
            let extrapolation = if residual_history.len() >= 3 {
                anderson_extrapolate_pair(&residual_history, &beta_history)
            } else {
                None
            };

            let outer = compute_outer_state(
                design,
                datafit,
                &*pen,
                warm.view(),
                r.view(),
                &ws,
                &coord_lipschitz,
                lam,
                config.cd.tol,
                extrapolation.as_ref().map(|(r_acc, beta_acc)| (r_acc.view(), beta_acc.view())),
                &mut best_dual_obj,
            );
            prev_outer_pgd = outer.max_pgd;

            // Outer convergence: prox-gradient stationarity (penalty-
            // agnostic, in coefficient units, commensurable with
            // `config.cd.tol`). The duality gap (now potentially
            // tightened by dual extrapolation) is computed as a side
            // channel for future use (gap-safe screening); switching
            // the outer stop to gap-based is a separate change with
            // its own tol-units conversion to work out against the
            // existing test corpus.
            if outer.max_pgd < config.cd.tol {
                break (r, report);
            }

            if outer.violators.is_empty() {
                // Working set is correct (no inactive feature wants to
                // become active) but the inner CD stopped sloppy. The
                // next pass will rerun with a tighter `inner_tol`
                // because `prev_outer_pgd` shrank.
                continue;
            }
            ws.extend(outer.violators);
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
    }

    (
        betas,
        PathReport {
            lambdas,
            iters,
            converged,
            final_objs,
            working_set_sizes,
            kkt_passes: kkt_passes_out,
        },
    )
}

/// Priority-based active-set rule (celer/skglm pattern).
///
/// For each feature, computes a violation score
/// `s_j = |X_jᵀ r / n| / w_j` (= `|grad_j| / w_j`); active features
/// (`β_j ≠ 0`) and unpenalised features (`w_j ≤ 0`) get `s_j = ∞` so
/// they're always pinned in. Returns the indices of the
/// `max(p0, 2 × |support|)` features with the largest scores, capped
/// to `n_features`.
///
/// **Why this replaces the strong rule**:
///
/// The Tibshirani strong rule passes-or-rejects features against a
/// threshold `(2 λ_k − λ_{k-1}) · w_j`, which has two degenerate
/// behaviours skein used to fall back from: (a) at λ_max no previous
/// λ exists, so the rule degrades to "include everything"; (b) the
/// threshold can stay loose enough to leave many irrelevant features
/// in the WS during the saturated tail. The priority rule fixes both:
/// at the cold-start λ_max we use a small `p0` WS and let the KKT
/// verifier pull in violators, and the WS size grows monotonically
/// with the support, matching the actual active-set dynamics.
///
/// One BLAS gemv to compute the gradient; one O(p) scan to rank.
/// Argpartition (`select_nth_unstable_by`) avoids a full sort.
fn priority_rule_screen(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    residual: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    beta: ArrayView1<'_, f64>,
    p0: usize,
) -> Vec<usize> {
    let p = design.n_features();
    let n_support = beta.iter().filter(|&&b| b != 0.0).count();
    let ws_size = (n_support * 2).max(p0).min(p);

    if ws_size == 0 {
        return Vec::new();
    }
    if ws_size >= p {
        return (0..p).collect();
    }

    let grad = datafit.full_grad(design, residual);
    // `score[j]` is the priority; INFINITY for pinned features, else
    // `|grad_j| / w_j`. We sort by score descending and take the top
    // `ws_size`.
    let mut scored: Vec<(usize, f64)> = (0..p)
        .map(|j| {
            let w = weights[j];
            let score = if w <= 0.0 || beta[j] != 0.0 {
                f64::INFINITY
            } else {
                grad[j].abs() / w
            };
            (j, score)
        })
        .collect();

    // Argpartition: descending order by score. The top `ws_size`
    // entries (by some permutation) end up in `scored[..ws_size]`.
    scored.select_nth_unstable_by(ws_size - 1, |a, b| {
        // Reverse so largest score sorts first; treat NaN as smallest
        // (it shouldn't appear, but be defensive).
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ws: Vec<usize> = scored[..ws_size].iter().map(|&(j, _)| j).collect();
    ws.sort_unstable();
    ws
}

/// Gap-safe sphere screen (Fercoq–Gramfort–Salmon 2015) for LS + separable
/// convex penalty.
///
/// At a feasible primal-dual pair built from `(β, residual = Xβ − y)`, the
/// duality gap and its safe radius `√(2G/n)` give a sphere around the
/// optimal dual that lets us safely discard features whose dual constraint
/// is *strictly* satisfied with margin. This rule is provably tighter than
/// the sequential strong rule on convex problems.
///
/// Currently-active features (`β_j ≠ 0`) are kept regardless, mirroring
/// the strong-rule convention.
fn gap_safe_screen(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    residual: ArrayView1<'_, f64>,
    beta: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    lambda: f64,
) -> Vec<usize> {
    let p = design.n_features();
    let n = design.n_samples() as f64;

    // Full gradient ∂L/∂β. LS overrides this with one matvec; default
    // impl loops over coord_grad.
    let g = datafit.full_grad(design, residual);

    // Project θ_init = -r/n into the dual-feasible set by scaling so
    // |X_jᵀ θ| ≤ λ w_j for every penalized feature.
    let mut max_ratio = 0.0_f64;
    for j in 0..p {
        let w = weights[j];
        if w <= 0.0 {
            continue;
        }
        let ratio = g[j].abs() / (lambda * w);
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    let scale = if max_ratio > 1.0 {
        1.0 / max_ratio
    } else {
        1.0
    };

    // Primal: (1/2n)‖r‖² + λ Σ w_j |β_j|. Unpenalized features (w_j ≤ 0)
    // don't contribute to the L1 term.
    let r_sq: f64 = residual.iter().map(|v| v * v).sum();
    let pen_value: f64 = (0..p).map(|j| weights[j].max(0.0) * beta[j].abs()).sum();
    let primal_obj = r_sq / (2.0 * n) + lambda * pen_value;

    // Dual obj derived from y = Xβ − r so we don't need direct access to y:
    //   D(θ) = ‖r‖²/n · scale · (1 − scale/2) − (βᵀg) · scale
    let beta_dot_g: f64 = (0..p).map(|j| beta[j] * g[j]).sum();
    let dual_obj = r_sq / n * scale * (1.0 - scale / 2.0) - beta_dot_g * scale;

    let gap = (primal_obj - dual_obj).max(0.0);
    let safe_r = (2.0 * gap / n).sqrt();

    let mut ws = Vec::new();
    for j in 0..p {
        let w = weights[j];
        if w <= 0.0 || beta[j] != 0.0 {
            ws.push(j);
            continue;
        }
        let xj_norm = design.col_sq_norm(j).sqrt();
        if g[j].abs() * scale + safe_r * xj_norm >= lambda * w {
            ws.push(j);
        }
    }
    ws
}

/// Per-pass outer state — violators, prox-gradient distance, and the
/// (best-known so far) duality gap when the datafit/penalty pair
/// supports it.
///
/// Returned fields:
///
/// - `violators`: features `j ∉ in_ws` whose one-step prox-gradient
///   update would move `β_j` by more than `tol` (i.e. they're far
///   enough from a stationary fixed point that they should join the
///   working set).
/// - `max_pgd`: `max_j |β_j − prox_j(β_j − grad_j / lc_j, 1 / lc_j)|`
///   over **all** features. `max_pgd ≤ tol` means β is at a
///   prox-gradient stationary point.
/// - `gap`: duality gap if computable (`Some(g)`), else `None`.
///   Computable iff `Datafit::lasso_dual_obj(...)` returns `Some`
///   *and* `Penalty::weights()` defines a meaningful L1-effective
///   constraint (`weights[j] > 0` for at least one penalised
///   feature). Currently fires for LS + elastic-net-family penalties.
/// - `lambda_bound`: `max_j |grad_j| / w_j` over penalised features.
///   This is what we compare to `λ` for dual feasibility scaling
///   (`scale = min(1, λ / lambda_bound)`); the path solver also
///   uses it to drive adaptive inner tolerance even when the gap
///   isn't computable.
///
/// One BLAS gemv (`full_grad`) plus `p` per-coord prox calls and a
/// few O(p) reductions for the gap. Asymptotically the same cost as
/// the gradient-only KKT verifier the path solver started with.
struct OuterState {
    violators: Vec<usize>,
    max_pgd: f64,
    gap: Option<f64>,
    #[allow(dead_code)] // surfaced for callers that want the dual-feasibility ratio
    lambda_bound: f64,
}

fn compute_outer_state(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn Penalty,
    beta: ArrayView1<'_, f64>,
    residual: ArrayView1<'_, f64>,
    in_ws: &[usize],
    coord_lipschitz: &[f64],
    lambda: f64,
    tol: f64,
    extrapolation: Option<(ArrayView1<'_, f64>, ArrayView1<'_, f64>)>,
    best_dual_obj: &mut f64,
) -> OuterState {
    let p = design.n_features();
    debug_assert_eq!(coord_lipschitz.len(), p);
    let weights = penalty.weights();
    let grad = datafit.full_grad(design, residual);

    // First pass: prox-gradient distance + violators + dual feasibility
    // bound (`max_j |grad_j| / w_j` over penalised features).
    let mut violators = Vec::new();
    let mut max_pgd = 0.0_f64;
    let mut lambda_bound = 0.0_f64;
    let mut ws_idx = 0usize;
    for j in 0..p {
        let in_ws_flag = ws_idx < in_ws.len() && in_ws[ws_idx] == j;
        if in_ws_flag {
            ws_idx += 1;
        }
        let w = weights[j];
        if w > 0.0 && w.is_finite() {
            let r = grad[j].abs() / w;
            if r > lambda_bound {
                lambda_bound = r;
            }
        }
        let lj = coord_lipschitz[j];
        if lj == 0.0 {
            continue;
        }
        let step = 1.0 / lj;
        let z = beta[j] - grad[j] * step;
        let prox_bj = penalty.prox_coord(j, z, step);
        let d = (prox_bj - beta[j]).abs();
        if d > max_pgd {
            max_pgd = d;
        }
        if !in_ws_flag && d > tol {
            violators.push(j);
        }
    }

    // Dual gap, when the datafit + penalty supports the lasso-form
    // duality. `scale = min(1, λ / lambda_bound)` projects the naive
    // dual point `θ_naive` (= -r/n for LS) into the feasibility set
    // `{θ : ‖Xᵀθ‖_∞ ≤ λ · w_j}`; the dual obj at the scaled point is
    // a valid lower bound on primal optimum.
    //
    // If the caller supplied an Anderson-extrapolated `(r_acc, β_acc)`
    // pair, we *also* evaluate the dual obj there (with its own
    // feasibility scaling) and take whichever is larger as the best
    // known dual. This is celer's trick: the Anderson trajectory
    // often lies closer to the dual optimum than the most recent
    // iterate, so the projected dual obj is bigger → tighter gap.
    //
    // `best_dual_obj` is monotone-non-decreasing across outer passes
    // (we keep the previous best and only replace if we found
    // something better this pass). At the cost of one extra rmatvec
    // when the extrapolation is supplied.
    let scale_naive = if lambda_bound > lambda {
        lambda / lambda_bound
    } else {
        1.0
    };

    let dual_correction_naive = penalty.dual_correction(beta);
    let dual_naive = datafit
        .lasso_dual_obj(design, beta, residual, grad.view(), scale_naive)
        .map(|d| d - dual_correction_naive);

    let dual_extrapolated = match (extrapolation, dual_naive) {
        (Some((r_acc, beta_acc)), Some(_)) => {
            // Compute extrapolated grad + feasibility ratio.
            let grad_acc = datafit.full_grad(design, r_acc);
            let mut lambda_bound_acc = 0.0_f64;
            for j in 0..p {
                let w = weights[j];
                if w > 0.0 && w.is_finite() {
                    let r_ratio = grad_acc[j].abs() / w;
                    if r_ratio > lambda_bound_acc {
                        lambda_bound_acc = r_ratio;
                    }
                }
            }
            let scale_acc = if lambda_bound_acc > lambda {
                lambda / lambda_bound_acc
            } else {
                1.0
            };
            datafit
                .lasso_dual_obj(design, beta_acc, r_acc, grad_acc.view(), scale_acc)
                .map(|d| d - penalty.dual_correction(beta_acc))
        }
        _ => None,
    };

    // best_dual_obj tracks the largest D we've ever seen across this
    // λ's outer passes. Update with both this pass's naive and (if
    // available) extrapolated points.
    if let Some(d) = dual_naive {
        if d > *best_dual_obj {
            *best_dual_obj = d;
        }
    }
    if let Some(d) = dual_extrapolated {
        if d > *best_dual_obj {
            *best_dual_obj = d;
        }
    }

    let gap = if best_dual_obj.is_finite() && dual_naive.is_some() {
        let primal = datafit.value(residual) + penalty.value(beta);
        Some((primal - *best_dual_obj).max(0.0))
    } else {
        None
    };

    OuterState {
        violators,
        max_pgd,
        gap,
        lambda_bound,
    }
}

/// Anderson extrapolation on a *pair* of parallel sequences.
///
/// Solves the K × K Anderson normal equations on the first sequence
/// (the "primary" one — typically the residual `r`, mirroring celer's
/// `last_K_R`), then applies the same coefficients to both. Returns
/// `(seq_a_acc, seq_b_acc)` where each is `seq[K] − U_seq · c` with
/// `c` normalised so `Σc = 1`.
///
/// The pair-form matters when the two sequences are linearly related
/// — for the path solver, `r_i = X β_i − y`, so the extrapolated
/// `(r_acc, β_acc)` automatically satisfies `r_acc = X β_acc − y`
/// and the dual obj formula in `lasso_dual_obj` (which assumes
/// residual ↔ parameter consistency) is still valid. Coefficients
/// summing to 1 cancel the `y` term — that's the algebraic guarantee.
///
/// Both sequences must be the same length and have ≥ 3 entries.
/// Returns `None` if the normal equations are numerically singular
/// (degenerate input — typical when the trajectory has stalled).
fn anderson_extrapolate_pair(
    seq_a: &[Array1<f64>],
    seq_b: &[Array1<f64>],
) -> Option<(Array1<f64>, Array1<f64>)> {
    debug_assert_eq!(seq_a.len(), seq_b.len(), "parallel sequences must align");
    if seq_a.len() < 3 {
        return None;
    }
    let n_diff = seq_a.len() - 1;
    let p_a = seq_a[0].len();
    let p_b = seq_b[0].len();

    // U_a (p_a × K) — primary sequence's difference matrix; drives the
    // normal equations.
    let mut u_a = Array2::<f64>::zeros((p_a, n_diff));
    for i in 0..n_diff {
        for j in 0..p_a {
            u_a[[j, i]] = seq_a[i + 1][j] - seq_a[i][j];
        }
    }

    let mut m = u_a.t().dot(&u_a);
    // Same Tikhonov regularisation as `cd::anderson_extrapolate`: the
    // normal-equation matrix is severely ill-conditioned on a
    // converging trajectory; ` reg = 1e-10 · max_diag` keeps the
    // system solvable while biasing the result negligibly when
    // well-conditioned. The acceptance check (whether `D` improves)
    // guards against bad extrapolations.
    let max_diag = (0..n_diff).map(|i| m[[i, i]]).fold(0.0_f64, f64::max);
    if max_diag < 1e-30 {
        return None;
    }
    let reg = 1e-10 * max_diag;
    for i in 0..n_diff {
        m[[i, i]] += reg;
    }
    let rhs = Array1::<f64>::ones(n_diff);
    let c_unnorm = solve_small(m, rhs)?;
    let sum: f64 = c_unnorm.sum();
    if !sum.is_finite() || sum.abs() < 1e-14 {
        return None;
    }
    let c = &c_unnorm / sum;

    // Apply c to both sequences. Same Anderson formula as for a single
    // sequence (`last - U · c`), independently for each.
    let last_a = seq_a.last().unwrap();
    let a_acc = last_a - &u_a.dot(&c);

    let mut u_b = Array2::<f64>::zeros((p_b, n_diff));
    for i in 0..n_diff {
        for j in 0..p_b {
            u_b[[j, i]] = seq_b[i + 1][j] - seq_b[i][j];
        }
    }
    let last_b = seq_b.last().unwrap();
    let b_acc = last_b - &u_b.dot(&c);

    Some((a_acc, b_acc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::{DenseMatrix, SparseCSC};
    use crate::penalty::{Mcp, Scad};
    use crate::solver::cd::cd_solve;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    /// Deterministic small problem without pulling in `rand`.
    /// xorshift64 fills X and a small noise vector.
    fn toy_problem(seed: u64) -> (DenseMatrix, Array1<f64>) {
        let n = 20;
        let p = 5;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let true_beta = array![1.0, 0.0, 0.0, -2.0, 0.0];
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        (DenseMatrix::new(x), y)
    }

    // ---- lambda_max ------------------------------------------------------

    #[test]
    fn lambda_max_uniform_weights_matches_max_correlation() {
        let (design, y) = toy_problem(1);
        let datafit = LeastSquares::new(y.clone());
        let p = design.n_features();
        let n = design.n_samples() as f64;
        let weights = Array1::<f64>::ones(p);

        let lam = lambda_max(&design, &datafit, weights.view());

        let mut expected = 0.0_f64;
        for j in 0..p {
            let g = design.col_dot(j, y.view()).abs() / n;
            if g > expected {
                expected = g;
            }
        }
        assert_abs_diff_eq!(lam, expected, epsilon = 1e-12);
    }

    #[test]
    fn lambda_max_scales_inversely_with_per_feature_weight() {
        let (design, y) = toy_problem(2);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();

        let uniform = lambda_max(&design, &datafit, Array1::<f64>::ones(p).view());

        let mut w = Array1::<f64>::ones(p);
        w[0] = 0.5; // halving weight doubles the effective threshold for feature 0
        let weighted = lambda_max(&design, &datafit, w.view());

        // Weighted lambda_max can only grow when a single weight is reduced.
        assert!(weighted >= uniform - 1e-12);
    }

    #[test]
    fn lambda_max_ignores_zero_weight_features() {
        let (design, y) = toy_problem(3);
        let datafit = LeastSquares::new(y.clone());
        let p = design.n_features();
        let n = design.n_samples() as f64;

        // Pick the argmax-corr feature and set its weight to zero. lambda_max
        // should now come from the second-largest correlation, not the first.
        let mut grads: Vec<(usize, f64)> = (0..p)
            .map(|j| (j, design.col_dot(j, y.view()).abs() / n))
            .collect();
        grads.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let argmax = grads[0].0;
        let second = grads[1].1;

        let mut w = Array1::<f64>::ones(p);
        w[argmax] = 0.0;
        let lam = lambda_max(&design, &datafit, w.view());

        assert_abs_diff_eq!(lam, second, epsilon = 1e-12);
    }

    // ---- lambda_grid -----------------------------------------------------

    #[test]
    fn lambda_grid_geometric_endpoints_and_constant_ratio() {
        let grid = lambda_grid(1.0, 5, 0.01);
        assert_eq!(grid.len(), 5);
        assert_abs_diff_eq!(grid[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(*grid.last().unwrap(), 0.01, epsilon = 1e-12);
        for k in 1..grid.len() {
            assert!(grid[k] < grid[k - 1]);
        }
        let ratio = grid[1] / grid[0];
        for k in 1..grid.len() {
            assert_abs_diff_eq!(grid[k] / grid[k - 1], ratio, epsilon = 1e-10);
        }
    }

    #[test]
    fn lambda_grid_single_point_returns_lambda_max() {
        let grid = lambda_grid(0.5, 1, 0.01);
        assert_eq!(grid, vec![0.5]);
    }

    // ---- solve_path: KKT at boundary -------------------------------------

    #[test]
    fn at_lambda_max_path_solution_is_zero() {
        // λ_max is defined by the convex KKT condition at β=0: |X_jᵀy/n| ≤ λw_j.
        // For MCP that condition is sufficient only in the convex regime γ > step.
        // Per-coord step ≈ n / ‖X_j‖² ≈ 3 for our toy problem, so γ = 100
        // keeps every coord well inside the convex regime.
        let (design, y) = toy_problem(4);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &datafit, weights.view());

        let cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![lam_max]),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 100.0, p)),
            &cfg,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn above_lambda_max_path_solution_is_zero() {
        let (design, y) = toy_problem(5);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &datafit, weights.view());

        let cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![1.5 * lam_max]),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, _) = solve_path(
            &design,
            &datafit,
            // γ in the convex regime — see comment on `at_lambda_max_…`.
            |lam| Box::new(Mcp::new(lam, 100.0, p)),
            &cfg,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn just_below_lambda_max_only_top_correlated_feature_enters() {
        // Orthogonal design + isolated correlation peak ⇒ only argmax feature
        // enters at λ slightly below λ_max. Uses MCP with γ huge so it
        // behaves like lasso (avoids nonconvex regime ambiguity at threshold).
        let n = 10;
        let p = 5;
        let mut x = Array2::<f64>::zeros((n, p));
        for j in 0..p {
            x[[j, j]] = 1.0;
        }
        // X_jᵀ y / n = y_j / n. Set widely spaced magnitudes so the second
        // feature stays well below 0.99 · λ_max.
        let mut y = Array1::<f64>::zeros(n);
        y[0] = 3.0;
        y[1] = 1.0;
        y[2] = 0.5;
        y[3] = 0.2;
        y[4] = 0.1;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let weights = Array1::<f64>::ones(p);

        let lam_max = lambda_max(&design, &datafit, weights.view());
        assert_abs_diff_eq!(lam_max, 0.3, epsilon = 1e-12);

        let cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![0.99 * lam_max]),
            cd: CdConfig {
                max_iter: 1000,
                tol: 1e-12,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            &cfg,
        );
        assert!(
            betas[[0, 0]].abs() > 1e-4,
            "argmax feature should be active, got β_0 = {}",
            betas[[0, 0]]
        );
        for j in 1..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-8);
        }
    }

    // ---- solve_path: shape, grid, custom λ -------------------------------

    #[test]
    fn path_output_shape_matches_n_lambdas_by_n_features() {
        let (design, y) = toy_problem(7);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();

        let cfg = PathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            &cfg,
        );
        assert_eq!(betas.shape(), &[8, p]);
        assert_eq!(report.lambdas.len(), 8);
        assert_eq!(report.iters.len(), 8);
        assert_eq!(report.converged.len(), 8);
        assert_eq!(report.final_objs.len(), 8);
    }

    #[test]
    fn auto_path_starts_at_lambda_max_and_decreases() {
        let (design, y) = toy_problem(8);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &datafit, weights.view());

        let cfg = PathConfig {
            n_lambdas: 10,
            lambda_min_ratio: 1e-3,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (_, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            &cfg,
        );
        assert_abs_diff_eq!(report.lambdas[0], lam_max, epsilon = 1e-10);
        for k in 1..report.lambdas.len() {
            assert!(report.lambdas[k] < report.lambdas[k - 1]);
        }
    }

    #[test]
    fn user_supplied_lambdas_are_honored_verbatim() {
        let (design, y) = toy_problem(10);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let custom = vec![1.0, 0.5, 0.25, 0.1];

        let cfg = PathConfig {
            n_lambdas: 0, // ignored when `lambdas` is Some
            lambda_min_ratio: 0.0,
            lambdas: Some(custom.clone()),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            &cfg,
        );
        assert_eq!(report.lambdas, custom);
        assert_eq!(betas.shape(), &[4, p]);
    }

    // ---- solve_path: warm-start equivalence & monotone support -----------

    #[test]
    fn warm_start_path_matches_cold_solve_at_smallest_lambda() {
        // Compares warm-started path β at smallest λ against a cold CD at the
        // same λ. The two trajectories must coincide in the convex regime —
        // we use γ = 100 (well above the per-coord step ≈ 3 of this toy
        // problem) to stay there. At γ = 3 (borderline non-convex), screening
        // and warm starts can land on a different stationary point than cold
        // CD, which is correct nonconvex behavior, just not what this test
        // is asserting.
        let (design, y) = toy_problem(11);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();

        let cfg = PathConfig {
            n_lambdas: 20,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 2000,
                tol: 1e-10,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 100.0, p)),
            &cfg,
        );
        let last_lambda = *report.lambdas.last().unwrap();
        let last_beta_path = betas.row(betas.nrows() - 1).to_owned();

        let pen = Mcp::new(last_lambda, 100.0, p);
        let (cold_beta, _) = cd_solve(
            &design,
            &datafit,
            &pen,
            &CdConfig {
                max_iter: 10_000,
                tol: 1e-10,
                acceleration: Some(5),
            },
        );
        for j in 0..p {
            assert_abs_diff_eq!(last_beta_path[j], cold_beta[j], epsilon = 1e-4);
        }
    }

    #[test]
    fn lasso_regime_path_support_is_monotone_nondecreasing() {
        // MCP with γ very large ≈ lasso ⇒ support grows as λ shrinks.
        let (design, y) = toy_problem(12);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();

        let cfg = PathConfig {
            n_lambdas: 15,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            &cfg,
        );
        let active =
            |k: usize| -> Vec<usize> { (0..p).filter(|&j| betas[[k, j]].abs() > 1e-6).collect() };
        for k in 1..betas.nrows() {
            let prev = active(k - 1);
            let cur = active(k);
            for j in &prev {
                assert!(
                    cur.contains(j),
                    "feature {} dropped from λ index {} to {}",
                    j,
                    k - 1,
                    k
                );
            }
        }
    }

    // ---- solve_path: KKT verification at every λ ------------------------

    #[test]
    fn lasso_path_solutions_satisfy_kkt() {
        let (design, y) = toy_problem(13);
        let datafit = LeastSquares::new(y.clone());
        let p = design.n_features();
        let n = design.n_samples() as f64;

        let cfg = PathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 5e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 10_000,
                tol: 1e-12,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)), // ≈ lasso
            &cfg,
        );
        for k in 0..report.lambdas.len() {
            let lam = report.lambdas[k];
            let beta = betas.row(k).to_owned();
            let r = &design.matvec(beta.view()) - &y;
            for j in 0..p {
                let g = design.col_dot(j, r.view()) / n;
                if beta[j].abs() > 1e-6 {
                    assert_abs_diff_eq!(g, -lam * beta[j].signum(), epsilon = 1e-3);
                } else {
                    assert!(
                        g.abs() <= lam + 1e-3,
                        "KKT violation @ λ_{}={}, j={}: |g|={} > λ",
                        k,
                        lam,
                        j,
                        g.abs()
                    );
                }
            }
        }
    }

    // ---- solve_path: degenerate columns ---------------------------------

    #[test]
    fn zero_columns_stay_zero_along_path() {
        let n = 30;
        let p = 4;
        let mut x = Array2::<f64>::from_elem((n, p), 0.5);
        x.column_mut(2).fill(0.0); // dead feature
        let y = Array1::<f64>::from_shape_fn(n, |i| (i as f64).sin());
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);

        let cfg = PathConfig {
            n_lambdas: 5,
            lambda_min_ratio: 1e-3,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            &cfg,
        );
        for k in 0..betas.nrows() {
            assert_abs_diff_eq!(betas[[k, 2]], 0.0, epsilon = 1e-12);
        }
    }

    // ---- SCAD smoke: same path API drives a different penalty -----------

    #[test]
    fn scad_path_runs_and_above_lambda_max_is_zero() {
        let (design, y) = toy_problem(14);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let weights = Array1::<f64>::ones(p);
        let lam_max = lambda_max(&design, &datafit, weights.view());

        let cfg = PathConfig {
            n_lambdas: 1,
            lambda_min_ratio: 1.0,
            lambdas: Some(vec![1.5 * lam_max]),
            cd: CdConfig::default(),
            screening: Screening::Strong,
            p0: 10,
        };
        let (betas, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Scad::new(lam, 3.7, p)),
            &cfg,
        );
        for j in 0..p {
            assert_abs_diff_eq!(betas[[0, j]], 0.0, epsilon = 1e-10);
        }
    }

    // ---- screening: behavior when off, and equivalence on/off -----------

    #[test]
    fn solve_path_screening_off_uses_full_working_set() {
        let (design, y) = toy_problem(20);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let cfg = PathConfig {
            n_lambdas: 5,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig::default(),
            screening: Screening::Off,
            p0: 10,
        };
        let (_, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            &cfg,
        );
        for &ws in &report.working_set_sizes {
            assert_eq!(ws, p);
        }
        for &kk in &report.kkt_passes {
            assert_eq!(kk, 1);
        }
    }

    #[test]
    fn solve_path_screening_on_matches_screening_off_within_tol() {
        // Both code paths must converge to the same β at every λ.
        let (design, y) = toy_problem(21);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let mk_cfg = |s: Screening| PathConfig {
            n_lambdas: 10,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: cd_cfg.clone(),
            screening: s,
            p0: 10,
        };
        let (b_off, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            &mk_cfg(Screening::Off),
        );
        let (b_on, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            &mk_cfg(Screening::Strong),
        );
        assert_eq!(b_off.shape(), b_on.shape());
        for k in 0..b_off.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(b_off[[k, j]], b_on[[k, j]], epsilon = 1e-6);
            }
        }
    }

    // ---- screening: actually drops features on a sparse-truth problem ---

    #[test]
    fn solve_path_screening_drops_inactive_features_on_sparse_problem() {
        // p = 20 features, only 3 active in truth. After the first few λ on
        // a decreasing path, the strong rule should screen out most of the
        // truly-inactive features.
        let n = 50;
        let p = 20;
        let mut state: u64 = 99;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 1.5;
        true_beta[1] = -2.0;
        true_beta[2] = 0.8;
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);

        // Use a small `p0` so the priority rule's lower bound for the
        // working set isn't dictated by the seed-WS size on this 20-
        // feature toy problem. Real-size problems use the default
        // `p0 = 10`.
        let cfg = PathConfig {
            n_lambdas: 15,
            lambda_min_ratio: 5e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: Some(5),
            },
            screening: Screening::Strong,
            p0: 3,
        };
        let (_, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 3.0, p)),
            &cfg,
        );
        // Mid-path, the priority rule should have kept the working set
        // close to the support (3 truly-active features), not the full
        // p=20.
        let mid = report.working_set_sizes.len() / 2;
        let mid_ws = report.working_set_sizes[mid];
        assert!(
            mid_ws < p / 2,
            "working set at mid-path should be < p/2 = {} (got {})",
            p / 2,
            mid_ws,
        );
    }

    // ---- gap-safe screening ---------------------------------------------

    #[test]
    fn gap_safe_path_matches_strong_rule_path_within_tol_on_lasso() {
        // Both screening rules must converge to the same β at every λ for
        // a convex problem. Discrepancies > tol would mean a bug in the
        // gap-safe rule (or its KKT verifier short-circuit).
        let (design, y) = toy_problem(40);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let cd_cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let mk_cfg = |s: Screening| PathConfig {
            n_lambdas: 12,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: cd_cfg.clone(),
            screening: s,
            p0: 10,
        };
        let (b_strong, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)), // ≈ lasso
            &mk_cfg(Screening::Strong),
        );
        let (b_gap, _) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            &mk_cfg(Screening::GapSafe),
        );
        for k in 0..b_strong.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(b_strong[[k, j]], b_gap[[k, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn gap_safe_drops_inactive_features_on_sparse_lasso_problem() {
        // p = 20 features, 3 active in truth. With a convex penalty,
        // gap-safe should screen most inactive features at mid-path.
        let n = 50;
        let p = 20;
        let mut state: u64 = 73;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 1.5;
        true_beta[1] = -2.0;
        true_beta[2] = 0.8;
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);

        let cfg = PathConfig {
            n_lambdas: 15,
            lambda_min_ratio: 5e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: Some(5),
            },
            screening: Screening::GapSafe,
            p0: 10,
        };
        let (_, report) = solve_path(
            &design,
            &datafit,
            |lam| Box::new(Mcp::new(lam, 1e6, p)),
            &cfg,
        );
        let mid = report.working_set_sizes.len() / 2;
        let mid_ws = report.working_set_sizes[mid];
        assert!(
            mid_ws < p / 2,
            "gap-safe ws at mid-path should be < p/2 = {} (got {})",
            p / 2,
            mid_ws,
        );
    }

    /// Convert a dense matrix to CSC by listing all non-zero entries.
    /// Treats every value as non-zero; this is the worst case for sparse
    /// (no compression) but lets us prove the SparseCSC backend gives
    /// bit-equivalent solver output for the same X.
    fn dense_to_csc(x: &Array2<f64>) -> SparseCSC {
        let n = x.nrows();
        let p = x.ncols();
        let mut data = Vec::with_capacity(n * p);
        let mut indices = Vec::with_capacity(n * p);
        let mut indptr = Vec::with_capacity(p + 1);
        indptr.push(0_usize);
        for j in 0..p {
            for i in 0..n {
                let v = x[[i, j]];
                if v != 0.0 {
                    data.push(v);
                    indices.push(i);
                }
            }
            indptr.push(data.len());
        }
        SparseCSC::new(
            n,
            Array1::from(data),
            Array1::from(indices),
            Array1::from(indptr),
        )
    }

    /// Sparse with random sparsity pattern. Same dense-vs-sparse equivalence
    /// argument: both backends compute the same X·β / Xᵀr / ‖X[:,j]‖²
    /// (modulo floating-point summation order, which we tolerate via
    /// `epsilon`), so the solver output should match.
    fn random_sparse_problem(seed: u64, density: f64) -> (Array2<f64>, Array1<f64>) {
        let n = 40;
        let p = 12;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| {
            // Uniform [0,1] via (sample + 1) / 2; zero out below density.
            let u = (sample() + 1.0) * 0.5;
            if u < density {
                sample()
            } else {
                0.0
            }
        });
        let true_beta = array![1.0, 0.0, -2.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        (x, y)
    }

    // ---- SparseCSC ↔ DenseMatrix solver equivalence ----------------------

    #[test]
    fn sparse_path_matches_dense_path_within_tol() {
        let (x_dense_arr, y) = random_sparse_problem(13, 0.4);
        let dense = DenseMatrix::new(x_dense_arr.clone());
        let sparse = dense_to_csc(&x_dense_arr);
        let p = dense.n_features();

        let cfg = PathConfig {
            n_lambdas: 10,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-12,
                acceleration: Some(5),
            },
            screening: Screening::Off,
            p0: 10,
        };
        let datafit_d = LeastSquares::new(y.clone());
        let datafit_s = LeastSquares::new(y.clone());
        // Lasso-like (γ=1e6) so the problem is convex and both backends
        // should converge to the same global optimum (up to FP order).
        let make_pen = |lam: f64| -> Box<dyn crate::Penalty> { Box::new(Mcp::new(lam, 1e6, p)) };

        let (betas_d, _) = solve_path(&dense, &datafit_d, make_pen, &cfg);
        let (betas_s, _) = solve_path(&sparse, &datafit_s, make_pen, &cfg);

        assert_eq!(betas_d.shape(), betas_s.shape());
        for k in 0..betas_d.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_d[[k, j]], betas_s[[k, j]], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn sparse_cd_matches_dense_cd_at_single_lambda() {
        let (x_dense_arr, y) = random_sparse_problem(17, 0.3);
        let dense = DenseMatrix::new(x_dense_arr.clone());
        let sparse = dense_to_csc(&x_dense_arr);
        let p = dense.n_features();

        let datafit_d = LeastSquares::new(y.clone());
        let datafit_s = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let pen = Mcp::new(0.05, 1e6, p);

        let (beta_d, _) = cd_solve(&dense, &datafit_d, &pen, &cfg);
        let (beta_s, _) = cd_solve(&sparse, &datafit_s, &pen, &cfg);

        for j in 0..p {
            assert_abs_diff_eq!(beta_d[j], beta_s[j], epsilon = 1e-8);
        }
    }
}
