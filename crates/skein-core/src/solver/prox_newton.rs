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
//! Inner solve uses `cd_solve_warm` (no per-outer-iter screening yet);
//! adding screening in the prox-Newton inner is M3.x.

use crate::datafit::GlmDatafit;
use crate::design::DesignMatrix;
use crate::penalty::Penalty;
use crate::solver::cd::{cd_solve_warm, CdConfig};
use crate::solver::path::{lambda_grid, lambda_max};
use ndarray::{Array1, Array2};

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

    for outer in 0..max_outer {
        outer_iters = outer + 1;
        let surrogate = glm.surrogate_at(design, warm.view());
        let beta_old = warm.clone();
        let (new_beta, inner_report) = cd_solve_warm(warm, design, &surrogate, penalty, cd_config);
        warm = new_beta;
        inner_iters.push(inner_report.iter);

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
    use crate::penalty::Mcp;
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
}
