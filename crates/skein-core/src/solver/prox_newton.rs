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
use crate::solver::path::{lambda_grid, lambda_max, priority_rule_screen_with_grad};
use ndarray::{Array1, Array2};

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

/// Single-λ proximal-Newton solve for any GLM that exposes a weighted-LS
/// surrogate via [`GlmDatafit`].
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

        // Batched BLAS gemv (`X².t().dot(w)`) on dense designs — falls
        // back to the per-column manual fold for sparse / mmap backends
        // via the default `DesignMatrix::weighted_col_sq_norms`.
        let lips_arr = design.weighted_col_sq_norms(sw);
        let lips: Vec<f64> = lips_arr.iter().map(|&v| v / n_f).collect();

        let r0 = surrogate.init_residual(design, warm.view());
        let grad0 = surrogate.full_grad(design, r0.view());
        let n_support = warm.iter().filter(|&&b| b != 0.0).count();
        let ws_size = (n_support * 2).max(PROX_NEWTON_P0).min(p);
        let mut ws =
            priority_rule_screen_with_grad(grad0.view(), weights.view(), warm.view(), ws_size);

        let mut inner_iter_total = 0usize;
        let mut expansion_pass = 0usize;

        // KKT-protected WS loop. CD restricted to `ws` → KKT verify on
        // the full feature set using one `full_grad` rmatvec and the
        // cached Lj. Violators (if any) expand the WS; cap and fall
        // back to the full set so a pathological surrogate can't blow
        // the outer-iter budget.
        loop {
            let (b_new, r_new, rep) = cd_solve_subset_weighted_ls_with_lips(
                warm, &ws, design, &surrogate, penalty, cd_config, &lips,
            );
            warm = b_new;
            inner_iter_total = inner_iter_total.saturating_add(rep.iter);

            let grad = surrogate.full_grad(design, r_new.view());
            let violators = find_kkt_violators_batched(
                penalty,
                warm.view(),
                grad.view(),
                &lips,
                &ws,
                cd_config.tol,
            );
            if violators.is_empty() {
                break;
            }
            expansion_pass += 1;
            if expansion_pass >= KKT_EXPANSION_PASSES {
                ws = (0..p).collect();
                let (b_new, _r_new, rep) = cd_solve_subset_weighted_ls_with_lips(
                    warm, &ws, design, &surrogate, penalty, cd_config, &lips,
                );
                warm = b_new;
                inner_iter_total = inner_iter_total.saturating_add(rep.iter);
                break;
            }
            ws.extend(violators);
            ws.sort_unstable();
            ws.dedup();
        }

        inner_iters.push(inner_iter_total);
        let _ = expansion_pass; // currently observed via tests only

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
        let (new_beta, report) =
            prox_newton_solve(design, glm, &*pen, warm, cd_config, max_outer, outer_tol);
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
}
