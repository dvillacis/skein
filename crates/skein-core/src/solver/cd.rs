//! Cyclic coordinate descent for separable (scalar) penalties + LS datafit.
//!
//! Intentionally minimal: no working set, no acceleration. Its job is to
//! validate the trait surface end-to-end. The production solver lives in a
//! follow-up.

use crate::datafit::{Datafit, LeastSquares};
use crate::design::DesignMatrix;
use crate::penalty::Penalty;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct CdConfig {
    pub max_iter: usize,
    /// Absolute coefficient-space tolerance. CD declares convergence when
    /// `max_j |β_j_new − β_j_old|` over a full sweep falls below this. This
    /// is the CD fixed-point condition (penalty-agnostic) and is generally
    /// tighter than a relative-objective criterion at the same numerical
    /// value: the objective can plateau while β still drifts toward its
    /// optimum.
    pub tol: f64,
    /// Type-II Anderson acceleration on the iterate sequence. `None` runs
    /// pure CD; `Some(K)` (K ≥ 2) attempts a K-step extrapolation every K
    /// sweeps and accepts it only if the objective decreases. Accepted
    /// candidates resync the residual via a fresh `init_residual` matvec.
    pub acceleration: Option<usize>,
}

impl Default for CdConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            acceleration: Some(5),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CdReport {
    pub iter: usize,
    pub converged: bool,
    pub final_obj: f64,
}

pub fn cd_solve(
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn Penalty,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let p = design.n_features();
    let (beta, _, report) =
        cd_solve_warm_with_residual(Array1::<f64>::zeros(p), design, datafit, penalty, config);
    (beta, report)
}

/// CD with a caller-supplied initial β. Used by the path solver to warm-start
/// down a λ-grid. Thin wrapper over [`cd_solve_subset`] with the full feature
/// set.
pub fn cd_solve_warm(
    beta_init: Array1<f64>,
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn Penalty,
    config: &CdConfig,
) -> (Array1<f64>, CdReport) {
    let (beta, _, report) =
        cd_solve_warm_with_residual(beta_init, design, datafit, penalty, config);
    (beta, report)
}

/// CD over the full feature set; returns the final residual alongside `β`.
/// The path solver calls this so it can re-use the residual the inner CD
/// already maintains (via incremental `r += δ · X[:, j]` updates), instead of
/// recomputing `r = Xβ − y` from scratch after every call — that recompute
/// was an `O(np)` matvec per λ that the caller already had the answer to.
pub fn cd_solve_warm_with_residual(
    beta_init: Array1<f64>,
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn Penalty,
    config: &CdConfig,
) -> (Array1<f64>, Array1<f64>, CdReport) {
    let p = design.n_features();
    let features: Vec<usize> = (0..p).collect();
    cd_solve_subset(beta_init, &features, design, datafit, penalty, config)
}

/// CD restricted to a subset of features. Coordinates not in `features` are
/// held fixed at their values in `beta_init`; their contribution to `Xβ` is
/// captured because the residual is initialized from the full `β`.
///
/// Returns `(β, r, report)` where `r = Xβ − y` (or the datafit's analogous
/// residual) is the final residual maintained by the in-loop axpy updates.
/// Callers in the path solver use `r` directly for KKT verification rather
/// than recomputing it via a fresh matvec.
///
/// Empty `features` returns immediately (no work to do, considered converged).
pub fn cd_solve_subset(
    beta_init: Array1<f64>,
    features: &[usize],
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn Penalty,
    config: &CdConfig,
) -> (Array1<f64>, Array1<f64>, CdReport) {
    let p = design.n_features();
    debug_assert_eq!(beta_init.len(), p, "beta_init length must equal n_features");
    let mut beta = beta_init;
    let mut r = datafit.init_residual(design, beta.view());

    if features.is_empty() {
        let obj = datafit.value(r.view()) + penalty.value(beta.view());
        return (
            beta,
            r,
            CdReport {
                iter: 0,
                converged: true,
                final_obj: obj,
            },
        );
    }

    let mut report = CdReport {
        iter: 0,
        converged: false,
        final_obj: 0.0,
    };

    let acceleration = config.acceleration.filter(|&k| k >= 2);
    let mut history: Vec<Array1<f64>> = Vec::new();
    if acceleration.is_some() {
        history.push(beta.clone());
    }

    for it in 0..config.max_iter {
        let mut max_delta = 0.0_f64;
        for &j in features {
            let lj = datafit.coord_lipschitz(design, j);
            if lj == 0.0 {
                continue;
            }
            let grad_j = datafit.coord_grad(design, j, r.view());
            let z = beta[j] - grad_j / lj;
            let step = 1.0 / lj;
            let new_bj = penalty.prox_coord(j, z, step);
            let delta = new_bj - beta[j];
            if delta != 0.0 {
                // r += δ · X[:, j] — zero-alloc via DesignMatrix::col_axpy.
                design.col_axpy(j, delta, r.view_mut());
                beta[j] = new_bj;
                let abs_delta = delta.abs();
                if abs_delta > max_delta {
                    max_delta = abs_delta;
                }
            }
        }

        let obj = datafit.value(r.view()) + penalty.value(beta.view());
        report.iter = it + 1;
        report.final_obj = obj;
        if max_delta < config.tol {
            report.converged = true;
            break;
        }

        // Anderson runs after the convergence check so it never affects
        // termination — only the next iteration's CD trajectory.
        if let Some(period) = acceleration {
            history.push(beta.clone());
            if history.len() > period + 1 {
                history.remove(0);
            }
            if history.len() == period + 1 {
                if let Some(beta_acc) = anderson_extrapolate(&history) {
                    let r_acc = datafit.init_residual(design, beta_acc.view());
                    let obj_acc = datafit.value(r_acc.view()) + penalty.value(beta_acc.view());
                    if obj_acc < obj {
                        beta = beta_acc;
                        r = r_acc;
                        history.clear();
                        history.push(beta.clone());
                    }
                }
            }
        }
    }

    (beta, r, report)
}

/// Specialised CD inner solve for a weighted [`LeastSquares`] surrogate,
/// restricted to a working subset of features.
///
/// The prox-Newton wrapper calls this for every outer iteration of every
/// GLM (Poisson, logistic, …); the surrogate has per-sample weights
/// `w_i = μ_i` that are *constant* for the whole inner call. The generic
/// `cd_solve_subset` path doesn't know that, so it routes every coord
/// update through `LeastSquares::coord_grad` (which allocates an n-sized
/// `w · r` buffer per call) and `LeastSquares::coord_lipschitz` (which
/// re-scans column `j` per call). This function exploits the constancy:
///
/// * Coordinate-Lipschitz constants `L_j = (1/n) Σ w_i x_{ij}²` are
///   precomputed once via `DesignMatrix::col_sq_norm_weighted` for every
///   feature in `features` and read from the cache for each update.
/// * Coordinate gradients `g_j = (1/n) Σ w_i x_{ij} r_i` route through
///   `DesignMatrix::col_dot_weighted`, a fused weighted dot with no
///   intermediate allocation.
///
/// Returns `(β, r, report)`. Coordinates outside `features` are held at
/// their values in `beta_init`; their contribution to the residual is
/// captured because `r` is initialised from the full `β`. The wrapper
/// uses `r` to KKT-check features outside the working set without
/// recomputing it from scratch.
///
/// Falls back to `cd_solve_subset` when the surrogate has no sample
/// weights — the unweighted LS path is already at memory bandwidth via
/// `col_sq_norm`'s lookup cache, and the generic loop is the single
/// source of truth for that case.
pub fn cd_solve_subset_weighted_ls(
    beta_init: Array1<f64>,
    features: &[usize],
    design: &dyn DesignMatrix,
    ls: &LeastSquares,
    penalty: &dyn Penalty,
    config: &CdConfig,
) -> (Array1<f64>, Array1<f64>, CdReport) {
    let sw = match ls.sample_weights() {
        Some(w) => w,
        None => return cd_solve_subset(beta_init, features, design, ls, penalty, config),
    };
    let p = design.n_features();
    let n_f = design.n_samples() as f64;
    let mut lips = vec![0.0_f64; p];
    for &j in features {
        lips[j] = design.col_sq_norm_weighted(j, sw) / n_f;
    }
    cd_solve_subset_weighted_ls_with_lips(
        beta_init, features, design, ls, penalty, config, &lips,
    )
}

/// Variant of [`cd_solve_subset_weighted_ls`] that receives a precomputed
/// Lipschitz cache `lips[j] = (1/n) Σ w_i x_{ij}²`. The prox-Newton
/// outer loop builds this cache once per outer iter and reuses it for
/// both the CD inner solve AND the KKT verifier.
///
/// Maintains a weighted-residual cache `wr = w · r` alongside `r` so
/// the coordinate gradient `(1/n) Σ w_i x_{ij} r_i` is computed as a
/// plain BLAS `col_dot(j, wr)` instead of the manual triple-product
/// loop in `col_dot_weighted`. The trade-off: every nonzero update
/// pays one extra weighted axpy (`wr += δ · w · X[:, j]`), but the
/// gradient queries — one per coord per sweep, the hot path — drop
/// from a manual loop to a BLAS ddot.
///
/// `lips.len()` must equal `p`. The CD loop reads `lips[j]` for `j ∈ features`;
/// out-of-WS entries are ignored.
pub fn cd_solve_subset_weighted_ls_with_lips(
    beta_init: Array1<f64>,
    features: &[usize],
    design: &dyn DesignMatrix,
    ls: &LeastSquares,
    penalty: &dyn Penalty,
    config: &CdConfig,
    lips: &[f64],
) -> (Array1<f64>, Array1<f64>, CdReport) {
    let sw = ls
        .sample_weights()
        .expect("cd_solve_subset_weighted_ls_with_lips is only valid for weighted-LS surrogates");

    let p = design.n_features();
    let n = design.n_samples();
    debug_assert_eq!(beta_init.len(), p, "beta_init length must equal n_features");
    debug_assert_eq!(lips.len(), p, "lips cache must have length n_features");

    let mut beta = beta_init;
    let mut r = ls.init_residual(design, beta.view());
    // wr[i] = w[i] * r[i]. Maintained incrementally with each nonzero
    // coordinate update; rebuilt from scratch after an Anderson reset.
    let mut wr = Array1::<f64>::zeros(n);
    for i in 0..n {
        wr[i] = sw[i] * r[i];
    }

    if features.is_empty() {
        let obj = ls.value(r.view()) + penalty.value(beta.view());
        return (
            beta,
            r,
            CdReport {
                iter: 0,
                converged: true,
                final_obj: obj,
            },
        );
    }

    let n_f = n as f64;
    let acceleration = config.acceleration.filter(|&k| k >= 2);
    let mut history: Vec<Array1<f64>> = Vec::new();
    if acceleration.is_some() {
        history.push(beta.clone());
    }

    let mut report = CdReport {
        iter: 0,
        converged: false,
        final_obj: 0.0,
    };

    for it in 0..config.max_iter {
        let mut max_delta = 0.0_f64;
        for &j in features {
            let lj = lips[j];
            if lj == 0.0 {
                continue;
            }
            // BLAS ddot against the cached weighted residual.
            let grad_j = design.col_dot(j, wr.view()) / n_f;
            let z = beta[j] - grad_j / lj;
            let step = 1.0 / lj;
            let new_bj = penalty.prox_coord(j, z, step);
            let delta = new_bj - beta[j];
            if delta != 0.0 {
                // Update both r and wr so the next coord's gradient
                // query stays consistent. col_axpy uses BLAS daxpy;
                // col_axpy_weighted is a manual weighted axpy, paid
                // only per *nonzero* update — the strong-rule WS
                // makes that vastly fewer than per-coord gradient
                // queries.
                design.col_axpy(j, delta, r.view_mut());
                design.col_axpy_weighted(j, delta, sw, wr.view_mut());
                beta[j] = new_bj;
                let abs_delta = delta.abs();
                if abs_delta > max_delta {
                    max_delta = abs_delta;
                }
            }
        }

        let obj = ls.value(r.view()) + penalty.value(beta.view());
        report.iter = it + 1;
        report.final_obj = obj;
        if max_delta < config.tol {
            report.converged = true;
            break;
        }

        if let Some(period) = acceleration {
            history.push(beta.clone());
            if history.len() > period + 1 {
                history.remove(0);
            }
            if history.len() == period + 1 {
                if let Some(beta_acc) = anderson_extrapolate(&history) {
                    let r_acc = ls.init_residual(design, beta_acc.view());
                    let obj_acc = ls.value(r_acc.view()) + penalty.value(beta_acc.view());
                    if obj_acc < obj {
                        beta = beta_acc;
                        r = r_acc;
                        // Rebuild wr from scratch after the Anderson
                        // jump; incremental maintenance would have
                        // missed the non-CD coordinate move.
                        for i in 0..n {
                            wr[i] = sw[i] * r[i];
                        }
                        history.clear();
                        history.push(beta.clone());
                    }
                }
            }
        }
    }

    (beta, r, report)
}

/// Type-II Anderson extrapolation on a sequence of CD iterates.
///
/// Given `K + 1` iterates `β^(0), …, β^(K)`, builds the difference matrix
/// `U ∈ ℝ^{p × K}` with columns `u_i = β^(i) − β^(i−1)` and returns
/// `β^(K) − U c`, where `c` solves `M c = 1` with `M = UᵀU`, then is
/// rescaled so `Σc = 1`. Returns `None` when there are fewer than 3
/// iterates or when `M` is numerically singular (degenerate input).
fn anderson_extrapolate(iterates: &[Array1<f64>]) -> Option<Array1<f64>> {
    if iterates.len() < 3 {
        return None;
    }
    let p = iterates[0].len();
    let n_diff = iterates.len() - 1;

    let mut u = Array2::<f64>::zeros((p, n_diff));
    for i in 0..n_diff {
        for j in 0..p {
            u[[j, i]] = iterates[i + 1][j] - iterates[i][j];
        }
    }

    let mut m = u.t().dot(&u);
    // Tikhonov regularization: U^T U is typically severely ill-conditioned
    // because successive CD iterates trace a low-rank subspace, so its
    // smaller singular values collapse below 1e-14 within a few steps.
    // Adding `reg · I` keeps the system solvable while biasing the result
    // negligibly when the problem is well-conditioned. The acceptance check
    // (obj decrease) guards against any harm from a biased extrapolation.
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

    // Infallible: the early return at the top of the function guarantees
    // `iterates.len() >= 3`, so `last()` is `Some`.
    let last = iterates
        .last()
        .expect("iterates non-empty: len() ≥ 3 checked above");
    let uc = u.dot(&c);
    Some(last - &uc)
}

/// Solve a small `n × n` system `A x = b` by Gaussian elimination with
/// partial pivoting. Returns `None` when the system is numerically singular.
/// Intended for the K × K Anderson normal equations (K small).
pub(crate) fn solve_small(mut a: Array2<f64>, mut b: Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n);
    debug_assert_eq!(b.len(), n);

    // Relative pivot threshold scaled by the largest input magnitude. This
    // matters for the Anderson normal equations, whose entries can be
    // ~1e-5 on a converging trajectory and would trip an absolute 1e-14
    // threshold even after Tikhonov regularization.
    let initial_scale = (0..n)
        .map(|i| a[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let pivot_thresh = 1e-14 * initial_scale;

    for i in 0..n {
        // Partial pivoting on column i.
        let mut piv = i;
        let mut piv_val = a[[i, i]].abs();
        for k in (i + 1)..n {
            let v = a[[k, i]].abs();
            if v > piv_val {
                piv = k;
                piv_val = v;
            }
        }
        if piv_val < pivot_thresh {
            return None;
        }
        if piv != i {
            for c in 0..n {
                let tmp = a[[i, c]];
                a[[i, c]] = a[[piv, c]];
                a[[piv, c]] = tmp;
            }
            b.swap(i, piv);
        }
        // Eliminate below the pivot.
        let pivot = a[[i, i]];
        for k in (i + 1)..n {
            let factor = a[[k, i]] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in i..n {
                a[[k, c]] -= factor * a[[i, c]];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution.
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[[i, c]] * x[c];
        }
        x[i] = s / a[[i, i]];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::DenseMatrix;
    use crate::penalty::Mcp;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn cd_recovers_zero_solution_under_strong_penalty() {
        // Tiny problem, large λ ⇒ optimal β = 0.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![0.1, 0.0, -0.1];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let penalty = Mcp::new(10.0, 3.0, 2);
        let (beta, report) = cd_solve(&design, &datafit, &penalty, &CdConfig::default());
        assert!(report.iter > 0);
        assert_abs_diff_eq!(beta[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(beta[1], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn cd_finds_signal_under_small_penalty() {
        // β* ≈ (1, 0): column 0 is correlated with y, column 1 is noise.
        let x = array![[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let y = array![1.0, 1.0, 1.0, 1.0];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let penalty = Mcp::new(0.01, 3.0, 2);
        let (beta, _) = cd_solve(&design, &datafit, &penalty, &CdConfig::default());
        assert!(beta[0] > 0.5);
        assert_abs_diff_eq!(beta[1], 0.0, epsilon = 1e-8);
    }

    // ---- cd_solve_subset -----------------------------------------------

    #[test]
    fn cd_solve_subset_holds_excluded_features_fixed() {
        let x = array![
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.8, 1.0],
            [0.1, 0.4, 0.6]
        ];
        let y = array![1.0, 0.5, 0.3, 0.2];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let penalty = Mcp::new(0.01, 1e6, 3);
        let beta_init = array![0.5, 1.0, -0.5];
        let features = vec![0, 2]; // hold β[1] fixed
        let (beta_out, _, _) = cd_solve_subset(
            beta_init.clone(),
            &features,
            &design,
            &datafit,
            &penalty,
            &CdConfig::default(),
        );
        assert_abs_diff_eq!(beta_out[1], beta_init[1], epsilon = 1e-12);
    }

    #[test]
    fn cd_solve_subset_full_features_matches_cd_solve() {
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![0.1, 0.0, -0.1];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let penalty = Mcp::new(0.05, 3.0, 2);
        let cfg = CdConfig {
            max_iter: 1000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let (beta_full, _) = cd_solve(&design, &datafit, &penalty, &cfg);
        let (beta_subset, _, _) = cd_solve_subset(
            ndarray::Array1::<f64>::zeros(2),
            &[0, 1],
            &design,
            &datafit,
            &penalty,
            &cfg,
        );
        for j in 0..2 {
            assert_abs_diff_eq!(beta_full[j], beta_subset[j], epsilon = 1e-10);
        }
    }

    // ---- KKT-based stopping ----------------------------------------------

    #[test]
    fn cd_at_zero_optimum_converges_at_first_iteration() {
        // Massive λ ⇒ β = 0 is optimal. Cold start ⇒ no coordinate moves on
        // the first sweep ⇒ max coord-update is 0 ⇒ KKT-stopping terminates
        // at iter = 1. Under the old relative-objective criterion the loop
        // had to run twice to compare two equal objectives, which is what
        // this test pins down.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8]];
        let y = array![0.1, 0.0, -0.1];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let penalty = Mcp::new(100.0, 3.0, 2);
        let (beta, report) = cd_solve(&design, &datafit, &penalty, &CdConfig::default());
        assert_eq!(report.iter, 1);
        assert!(report.converged);
        for j in 0..2 {
            assert_abs_diff_eq!(beta[j], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn cd_kkt_residual_after_convergence_is_below_tol() {
        // After CD declares convergence, the per-coordinate prox-gradient
        // step at every j should be small — that *is* the CD fixed-point
        // condition. We allow a 10× factor because the criterion is checked
        // pre-update on the last sweep; the post-stop β can still be up to
        // ~tol away in any one coord on the next-would-be sweep.
        let x = array![
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.2, 0.8, 0.7],
            [0.1, 0.4, 0.9]
        ];
        let y = array![1.0, 0.5, 0.3, 0.2];
        let design = DenseMatrix::new(x.clone());
        let datafit = LeastSquares::new(y.clone());
        let penalty = Mcp::new(0.05, 1e6, 3);
        let tol = 1e-8;
        let cfg = CdConfig {
            max_iter: 10_000,
            tol,
            acceleration: Some(5),
        };
        let (beta, report) = cd_solve(&design, &datafit, &penalty, &cfg);
        assert!(report.converged, "CD should have converged within max_iter");

        let r = &x.dot(&beta) - &y;
        let n = design.n_samples() as f64;
        let mut max_residual = 0.0_f64;
        for j in 0..3 {
            let lj = datafit.coord_lipschitz(&design, j);
            if lj == 0.0 {
                continue;
            }
            let grad_j = design.col_dot(j, r.view()) / n;
            let z = beta[j] - grad_j / lj;
            let step = 1.0 / lj;
            let new_bj = penalty.prox_coord(j, z, step);
            let delta = (new_bj - beta[j]).abs();
            if delta > max_residual {
                max_residual = delta;
            }
        }
        assert!(
            max_residual < tol * 10.0,
            "KKT residual {} should be < {}",
            max_residual,
            tol * 10.0
        );
    }

    #[test]
    fn cd_solve_subset_handles_empty_features() {
        let x = array![[1.0, 0.5], [0.5, 1.0]];
        let y = array![0.5, 0.5];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let penalty = Mcp::new(0.1, 3.0, 2);
        let beta_init = array![0.7, -0.3];
        let (beta_out, _, report) = cd_solve_subset(
            beta_init.clone(),
            &[],
            &design,
            &datafit,
            &penalty,
            &CdConfig::default(),
        );
        assert_eq!(beta_out, beta_init);
        assert!(report.converged);
        assert_eq!(report.iter, 0);
    }

    // ---- Anderson acceleration ------------------------------------------

    #[test]
    fn anderson_extrapolate_returns_none_for_singular_input() {
        // Identical iterates ⇒ U = 0 ⇒ M = 0 ⇒ singular.
        let iterates: Vec<Array1<f64>> = (0..4).map(|_| array![1.0, 2.0, 3.0]).collect();
        assert!(anderson_extrapolate(&iterates).is_none());
    }

    #[test]
    fn anderson_extrapolate_returns_none_for_too_few_iterates() {
        let iterates: Vec<Array1<f64>> = vec![array![1.0, 2.0], array![1.5, 2.5]];
        assert!(anderson_extrapolate(&iterates).is_none());
    }

    #[test]
    fn anderson_extrapolate_returns_some_for_well_conditioned_input() {
        // Two-mode geometric convergence in 2D: β^k = β* + r1^k v1 + r2^k v2.
        // U has full column rank, M is positive definite.
        let beta_star = array![1.0, 2.0];
        let v1 = array![1.0, 0.0];
        let v2 = array![0.0, 1.0];
        let r1 = 0.7_f64;
        let r2 = 0.3_f64;
        let iterates: Vec<Array1<f64>> = (0..3)
            .map(|k| &beta_star + &(&v1 * r1.powi(k)) + &(&v2 * r2.powi(k)))
            .collect();
        let acc = anderson_extrapolate(&iterates).expect("should solve");
        assert_eq!(acc.len(), 2);
        assert!(acc[0].is_finite() && acc[1].is_finite());
    }

    #[test]
    fn cd_with_anderson_matches_unaccelerated_lasso_solution() {
        // On a convex problem (lasso ≈ MCP at huge γ), accelerated and pure
        // CD must converge to the same β. Acceptance check guards against
        // bad extrapolations, so divergence here would mean a bug.
        let x = array![
            [1.0, 0.5, 0.3, 0.1],
            [0.5, 1.0, 0.2, 0.4],
            [0.2, 0.8, 1.0, 0.7],
            [0.1, 0.4, 0.9, 0.3],
            [0.3, 0.6, 0.5, 1.0]
        ];
        let y = array![1.0, 0.5, 0.3, 0.2, 0.6];
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let p = 4;
        let penalty = Mcp::new(0.02, 1e6, p);

        let cfg_off = CdConfig {
            max_iter: 50_000,
            tol: 1e-12,
            acceleration: None,
        };
        let cfg_on = CdConfig {
            max_iter: 50_000,
            tol: 1e-12,
            acceleration: Some(5),
        };
        let (b_off, _) = cd_solve(&design, &datafit, &penalty, &cfg_off);
        let (b_on, _) = cd_solve(&design, &datafit, &penalty, &cfg_on);
        for j in 0..p {
            assert_abs_diff_eq!(b_off[j], b_on[j], epsilon = 1e-7);
        }
    }

    // ---- weighted LS (sample_weights) ----------------------------------

    #[test]
    fn weighted_ls_with_uniform_weights_one_matches_unweighted() {
        // sample_weights = ones(n) must produce identical β to no-weights.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![1.0, 0.5, 0.3, 0.2];
        let design = DenseMatrix::new(x);
        let n = 4;
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let penalty = Mcp::new(0.05, 1e6, 2);

        let plain = LeastSquares::new(y.clone());
        let weighted = LeastSquares::with_sample_weights(y, Array1::<f64>::ones(n));

        let (b_plain, _) = cd_solve(&design, &plain, &penalty, &cfg);
        let (b_weighted, _) = cd_solve(&design, &weighted, &penalty, &cfg);
        for j in 0..2 {
            assert_abs_diff_eq!(b_plain[j], b_weighted[j], epsilon = 1e-10);
        }
    }

    #[test]
    fn weighted_ls_doubled_uniform_weights_equals_halved_lambda() {
        // With sample_weights = c·ones(n), the weighted-LS gradient is c·LS-gradient,
        // so the optimum at λ matches plain LS at λ/c. Verify with c=2.
        let x = array![[1.0, 0.5], [0.5, 1.0], [0.2, 0.8], [0.1, 0.4]];
        let y = array![1.0, 0.5, 0.3, 0.2];
        let design = DenseMatrix::new(x);
        let n = 4;
        let lambda = 0.05;
        let cfg = CdConfig {
            max_iter: 10_000,
            tol: 1e-14,
            acceleration: None,
        };

        // Solve plain LS at λ/2.
        let plain = LeastSquares::new(y.clone());
        let pen_half = Mcp::new(lambda / 2.0, 1e6, 2);
        let (b_plain, _) = cd_solve(&design, &plain, &pen_half, &cfg);

        // Solve weighted LS (c = 2) at λ.
        let weighted = LeastSquares::with_sample_weights(y, Array1::<f64>::from_elem(n, 2.0));
        let pen_full = Mcp::new(lambda, 1e6, 2);
        let (b_weighted, _) = cd_solve(&design, &weighted, &pen_full, &cfg);

        // Tolerance 1e-5 (not 1e-6) absorbs the tiny FP-summation-order
        // difference between the plain and weighted gradient code paths;
        // both still converge to the same fixed point.
        for j in 0..2 {
            assert_abs_diff_eq!(b_plain[j], b_weighted[j], epsilon = 1e-5);
        }
    }

    #[test]
    fn weighted_ls_per_sample_weighting_changes_solution() {
        // Nontrivial per-sample weights should produce a different β from
        // the unweighted case (sanity that the weights actually flow into
        // the gradient and Lipschitz, not just `value`).
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.5], [0.5, 1.0]];
        let y = array![2.0, 0.5, 1.0, 1.0];
        let design = DenseMatrix::new(x);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let penalty = Mcp::new(0.02, 1e6, 2);

        let plain = LeastSquares::new(y.clone());
        let (b_plain, _) = cd_solve(&design, &plain, &penalty, &cfg);

        // Heavily weight the first sample (β should pull toward fitting it).
        let w = array![10.0, 1.0, 1.0, 1.0];
        let weighted = LeastSquares::with_sample_weights(y, w);
        let (b_weighted, _) = cd_solve(&design, &weighted, &penalty, &cfg);

        let max_diff: f64 = (0..2)
            .map(|j| (b_plain[j] - b_weighted[j]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-3,
            "expected non-uniform sample weights to change β; max diff = {}",
            max_diff
        );
    }

    // ---- cd_solve_subset_weighted_ls ----------------------------------

    fn weighted_ls_problem() -> (DenseMatrix, LeastSquares) {
        // Reproducibly randomised n=80, p=20 weighted-LS problem with
        // heterogeneous weights spanning four orders of magnitude — the
        // regime the Poisson surrogate lands in once μ = exp(η) varies
        // across samples.
        let n = 80;
        let p = 20;
        let mut state: u64 = 0xC0FFEE_C0DE_5678;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| next());
        let y = Array1::<f64>::from_shape_fn(n, |_| next());
        let w = Array1::<f64>::from_shape_fn(n, |_| {
            // 1e-2 … 1e2 — same dynamic range as PoissonLog surrogate
            // weights after a few prox-Newton outer iters.
            let u = (next() + 1.0) * 0.5;
            10.0_f64.powf(4.0 * u - 2.0)
        });
        let design = DenseMatrix::new(x);
        let ls = LeastSquares::with_sample_weights(y, w);
        (design, ls)
    }

    #[test]
    fn cd_solve_subset_weighted_ls_full_set_matches_generic_at_tight_tol() {
        // With `features = (0..p).collect()`, the fast path must match
        // `cd_solve_warm` on a weighted-LS problem within ULP-level
        // slack (multiply ordering differs: `w · x · v` vs `(w·v) · x`).
        let (design, ls) = weighted_ls_problem();
        let p = design.n_features();
        let penalty = Mcp::new(0.05, 3.0, p);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: Some(5),
        };
        let beta_init = Array1::<f64>::zeros(p);
        let features: Vec<usize> = (0..p).collect();

        let (b_ref, rep_ref) =
            cd_solve_warm(beta_init.clone(), &design, &ls, &penalty, &cfg);
        let (b_fast, _r, rep_fast) = cd_solve_subset_weighted_ls(
            beta_init, &features, &design, &ls, &penalty, &cfg,
        );

        for j in 0..p {
            assert_abs_diff_eq!(b_ref[j], b_fast[j], epsilon = 1e-9);
        }
        assert_eq!(rep_ref.converged, rep_fast.converged);
        assert!(
            (rep_ref.iter as i64 - rep_fast.iter as i64).abs() <= 2,
            "iteration counts diverged: ref={} fast={}",
            rep_ref.iter,
            rep_fast.iter
        );
    }

    #[test]
    fn cd_solve_subset_weighted_ls_falls_back_when_unweighted() {
        // No sample weights ⇒ delegate to the generic subset path
        // verbatim (no caching savings to be had — `col_sq_norm` is
        // already a table lookup for unweighted LS).
        let x = array![[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.8, 1.0]];
        let y = array![1.0, 0.5, 0.3];
        let design = DenseMatrix::new(x);
        let ls = LeastSquares::new(y);
        let penalty = Mcp::new(0.01, 3.0, 3);
        let cfg = CdConfig::default();
        let beta_init = Array1::<f64>::zeros(3);
        let features: Vec<usize> = (0..3).collect();

        let (b_ref, _, _) = cd_solve_subset(
            beta_init.clone(),
            &features,
            &design,
            &ls,
            &penalty,
            &cfg,
        );
        let (b_fast, _, _) = cd_solve_subset_weighted_ls(
            beta_init, &features, &design, &ls, &penalty, &cfg,
        );
        for j in 0..3 {
            assert_eq!(
                b_ref[j], b_fast[j],
                "unweighted fallback must be bit-identical at j={}",
                j
            );
        }
    }
}
