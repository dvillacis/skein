//! Graphical lasso (Friedman/Hastie/Tibshirani 2008) with weighted L1,
//! MCP, or SCAD penalty on the off-diagonal entries of the precision
//! matrix `Θ = Σ⁻¹`.
//!
//! Algorithm — cycle over columns `k`, peel `W = [[W₁₁, w₁₂], [w₁₂ᵀ,
//! w₂₂]]`, solve the weighted-lasso subproblem
//!
//! ```text
//!     β̂_k = argmin_β  (1/2) βᵀ W₁₁ β − sᵀ_{¬k,k} β
//!                     + λ Σ_{j ≠ k} w_{kj} · p(|β_j|),
//! ```
//!
//! and update `w₁₂ = W₁₁ β̂_k` (with the symmetric counterpart). The
//! inner solve runs on a [`GramDesign`] + [`GramLeastSquares`], so the
//! existing [`cd_solve`] handles it unchanged — no glasso-specific CD
//! kernel. Disable Anderson acceleration on the inner solve:
//! [`GramLeastSquares::value`] returns the gradient norm as a proxy
//! rather than the true quadratic objective.
//!
//! `Θ` is reconstructed at convergence from the columns of `β̂` via the
//! block-matrix inverse formula
//!
//! ```text
//!     Θ_{kk} = 1 / (W_{kk} − w_{¬k,k}ᵀ β̂_k),
//!     Θ_{¬k,k} = −Θ_{kk} · β̂_k.
//! ```

use crate::datafit::GramLeastSquares;
use crate::design::{DesignMatrix, GramDesign};
use crate::penalty::ScalarPenaltyFactory;
use crate::solver::{cd_solve, CdConfig};
use ndarray::{Array1, Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct GlassoConfig {
    /// Maximum number of full sweeps over columns.
    pub max_outer_iter: usize,
    /// Convergence on `‖W_new − W_old‖_∞` (max absolute change in the
    /// working covariance estimate between sweeps).
    pub outer_tol: f64,
    /// Added once to the diagonal of `W` at initialisation: `W_kk =
    /// S_kk + diag_offset`. The standard sklearn / Friedman convention
    /// sets this equal to the L1 strength `α` to keep `W` positive
    /// definite throughout. Default `0.0`.
    pub diag_offset: f64,
    /// CD configuration for the per-column inner solve. Anderson must
    /// be `None` because [`GramLeastSquares::value`] is a proxy
    /// (see its docs).
    pub inner: CdConfig,
    /// Warm-start `W`. If `None`, `W = S` (plus `diag_offset` on the
    /// diagonal).
    pub warm_start: Option<Array2<f64>>,
}

impl Default for GlassoConfig {
    fn default() -> Self {
        Self {
            max_outer_iter: 100,
            outer_tol: 1e-4,
            diag_offset: 0.0,
            inner: CdConfig {
                max_iter: 200,
                tol: 1e-6,
                acceleration: None,
            },
            warm_start: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlassoReport {
    pub outer_iter: usize,
    pub converged: bool,
    pub max_w_delta: f64,
}

/// Solve the (weighted, possibly nonconvex) graphical lasso problem.
///
/// `sample_cov` is the `(p, p)` symmetric sample covariance `S`.
/// `edge_weights[i, j]` is the per-edge multiplier on `λ · p(|Θ_{ij}|)`;
/// `None` is uniform weight 1. The diagonal of `edge_weights` is
/// ignored (the diagonal of `Θ` is never penalised). `penalty_factory`
/// builds a per-edge weighted scalar penalty (L1, MCP, or SCAD).
///
/// Returns `(precision Θ, covariance W, report)`.
pub fn glasso_solve(
    sample_cov: ArrayView2<f64>,
    edge_weights: Option<ArrayView2<f64>>,
    penalty_factory: &dyn ScalarPenaltyFactory,
    config: &GlassoConfig,
) -> (Array2<f64>, Array2<f64>, GlassoReport) {
    let p = sample_cov.nrows();
    assert_eq!(sample_cov.ncols(), p, "sample_cov must be square");
    if let Some(ew) = edge_weights {
        assert_eq!(
            (ew.nrows(), ew.ncols()),
            (p, p),
            "edge_weights must be (p, p)"
        );
    }

    // Initialise W.
    let mut w = match &config.warm_start {
        Some(init) => {
            assert_eq!(
                init.dim(),
                (p, p),
                "warm_start must have the same shape as sample_cov"
            );
            init.clone()
        }
        None => sample_cov.to_owned(),
    };
    if config.warm_start.is_none() && config.diag_offset != 0.0 {
        for k in 0..p {
            w[[k, k]] = sample_cov[[k, k]] + config.diag_offset;
        }
    }

    // β̂_k stored at row k (column k's lasso coefficients, with
    // diagonal entry 0). Length-`p` row keeps indexing trivial.
    let mut betas = Array2::<f64>::zeros((p, p));

    let mut report = GlassoReport {
        outer_iter: 0,
        converged: false,
        max_w_delta: f64::INFINITY,
    };

    if p == 1 {
        // Degenerate case: Θ = 1 / W_11.
        let mut theta = Array2::<f64>::zeros((1, 1));
        theta[[0, 0]] = 1.0 / w[[0, 0]];
        report.outer_iter = 0;
        report.converged = true;
        report.max_w_delta = 0.0;
        return (theta, w, report);
    }

    let mut w_sub = Array2::<f64>::zeros((p - 1, p - 1));
    let mut diag_sub = Array1::<f64>::zeros(p - 1);
    let mut s_sub = Array1::<f64>::zeros(p - 1);
    let mut weight_slice = Array1::<f64>::ones(p - 1);

    for it in 0..config.max_outer_iter {
        let w_prev = w.clone();

        for k in 0..p {
            // Peel `W_{-k,-k}`, `S_{-k,k}`, and the row-k slice of edge
            // weights into the pre-allocated buffers.
            fill_peeled(&mut w_sub, &mut diag_sub, w.view(), k);
            fill_peeled_col(&mut s_sub, sample_cov, k);
            match edge_weights {
                Some(ew) => fill_peeled_col(&mut weight_slice, ew, k),
                None => weight_slice.fill(1.0),
            }

            // Solve the inner weighted lasso on the gram form.
            let design = GramDesign::new(w_sub.clone());
            let datafit = GramLeastSquares::new(diag_sub.clone(), s_sub.clone());
            let penalty = penalty_factory.build(weight_slice.clone());
            let (beta, _) = cd_solve(&design, &datafit, &*penalty, &config.inner);

            // Store β̂_k.
            for (ii, j) in (0..p).filter(|&j| j != k).enumerate() {
                betas[[k, j]] = beta[ii];
            }
            // Update column k of W (and the symmetric row k):
            // `w_{¬k,k} = W_{¬k,¬k} · β̂_k`. Diagonal `W_{kk}` is
            // fixed throughout — Friedman convention.
            let w_dot_beta = design.matvec(beta.view());
            for (ii, j) in (0..p).filter(|&j| j != k).enumerate() {
                w[[j, k]] = w_dot_beta[ii];
                w[[k, j]] = w_dot_beta[ii];
            }
        }

        let mut max_delta = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                let d = (w[[i, j]] - w_prev[[i, j]]).abs();
                if d > max_delta {
                    max_delta = d;
                }
            }
        }
        report.outer_iter = it + 1;
        report.max_w_delta = max_delta;
        if max_delta < config.outer_tol {
            report.converged = true;
            break;
        }
    }

    // Reconstruct Θ from W and the column-wise β̂'s.
    let theta = reconstruct_precision(w.view(), betas.view());
    (theta, w, report)
}

fn fill_peeled(
    out_block: &mut Array2<f64>,
    out_diag: &mut Array1<f64>,
    src: ndarray::ArrayView2<f64>,
    k: usize,
) {
    let p = src.nrows();
    let mut ii = 0;
    for i in 0..p {
        if i == k {
            continue;
        }
        let mut jj = 0;
        for j in 0..p {
            if j == k {
                continue;
            }
            out_block[[ii, jj]] = src[[i, j]];
            jj += 1;
        }
        out_diag[ii] = src[[i, i]];
        ii += 1;
    }
}

fn fill_peeled_col(out: &mut Array1<f64>, src: ndarray::ArrayView2<f64>, k: usize) {
    let p = src.nrows();
    let mut ii = 0;
    for i in 0..p {
        if i == k {
            continue;
        }
        out[ii] = src[[i, k]];
        ii += 1;
    }
}

fn reconstruct_precision(
    w: ndarray::ArrayView2<f64>,
    betas: ndarray::ArrayView2<f64>,
) -> Array2<f64> {
    let p = w.nrows();
    let mut theta = Array2::<f64>::zeros((p, p));
    for k in 0..p {
        // Θ_{kk} = 1 / (W_{kk} − w_{¬k,k}ᵀ β̂_k) = 1 / (W_{kk} − Σ_{j≠k} W_{jk} β̂_{k,j}).
        let mut inner = w[[k, k]];
        for j in 0..p {
            if j != k {
                inner -= w[[j, k]] * betas[[k, j]];
            }
        }
        // Numerical safety: clamp away from zero. A truly singular case
        // means the algorithm hasn't converged.
        let theta_kk = if inner.abs() < 1e-30 {
            1.0 / 1e-30
        } else {
            1.0 / inner
        };
        theta[[k, k]] = theta_kk;
        for j in 0..p {
            if j != k {
                theta[[j, k]] = -theta_kk * betas[[k, j]];
            }
        }
    }
    // Symmetrise — column-wise reconstruction yields two estimates of
    // each off-diagonal (from each direction of the peel), average them.
    for i in 0..p {
        for j in (i + 1)..p {
            let m = 0.5 * (theta[[i, j]] + theta[[j, i]]);
            theta[[i, j]] = m;
            theta[[j, i]] = m;
        }
    }
    theta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::penalty::{LassoFactory, McpFactory};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Diagonal `S` with zero off-diagonals: the sparse precision is
    /// also diagonal with `Θ_kk = 1 / S_kk`, and any positive `λ`
    /// preserves that.
    #[test]
    fn diagonal_input_yields_diagonal_precision() {
        let s = array![[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]];
        let factory = LassoFactory { lambda: 0.1 };
        let cfg = GlassoConfig {
            diag_offset: 0.1,
            ..GlassoConfig::default()
        };
        let (theta, _, report) = glasso_solve(s.view(), None, &factory, &cfg);
        assert!(report.converged);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    // Θ_ii = 1 / (S_ii + diag_offset)
                    assert_abs_diff_eq!(theta[[i, j]], 1.0 / (s[[i, i]] + 0.1), epsilon = 1e-6);
                } else {
                    assert_abs_diff_eq!(theta[[i, j]], 0.0, epsilon = 1e-6);
                }
            }
        }
    }

    /// Tight λ should drive every off-diagonal of Θ to zero.
    #[test]
    fn large_lambda_zeros_off_diagonals() {
        // Build a moderately correlated S.
        let s = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]];
        let lam = 5.0;
        let factory = LassoFactory { lambda: lam };
        let cfg = GlassoConfig {
            diag_offset: lam,
            outer_tol: 1e-6,
            ..GlassoConfig::default()
        };
        let (theta, _, report) = glasso_solve(s.view(), None, &factory, &cfg);
        assert!(report.converged);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        theta[[i, j]].abs() < 1e-6,
                        "expected near-zero off-diag at ({},{}): got {}",
                        i,
                        j,
                        theta[[i, j]]
                    );
                }
            }
        }
    }

    /// At very small λ, glasso should recover an off-diagonal pattern
    /// close to `S⁻¹` (regularisation negligible). Smoke-checks the
    /// algorithm is doing precision estimation, not noise.
    #[test]
    fn small_lambda_approximates_inverse_covariance() {
        // Pick a small SPD S with a clean inverse.
        // S = [[2, 0.5], [0.5, 1]] → det = 1.75 → S⁻¹ = (1/1.75) [[1, -0.5], [-0.5, 2]].
        let s = array![[2.0, 0.5], [0.5, 1.0]];
        let lam = 1e-4;
        let factory = LassoFactory { lambda: lam };
        let cfg = GlassoConfig {
            diag_offset: lam,
            outer_tol: 1e-8,
            max_outer_iter: 200,
            inner: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            ..GlassoConfig::default()
        };
        let (theta, _, report) = glasso_solve(s.view(), None, &factory, &cfg);
        assert!(report.converged);
        // Expected (ignoring tiny diag_offset):
        //   Θ_00 ≈ 1 / 1.75 ≈ 0.5714,  Θ_11 ≈ 2 / 1.75 ≈ 1.1428,
        //   Θ_01 = Θ_10 ≈ -0.5 / 1.75 ≈ -0.2857.
        assert_abs_diff_eq!(theta[[0, 0]], 1.0 / 1.75, epsilon = 5e-3);
        assert_abs_diff_eq!(theta[[1, 1]], 2.0 / 1.75, epsilon = 5e-3);
        assert_abs_diff_eq!(theta[[0, 1]], -0.5 / 1.75, epsilon = 5e-3);
        assert_abs_diff_eq!(theta[[1, 0]], theta[[0, 1]], epsilon = 1e-12);
    }

    /// MCP factory must wire through to the inner CD and produce a
    /// valid (symmetric) Θ. Smoke test, not a parity test.
    #[test]
    fn mcp_runs_to_completion() {
        let s = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]];
        let factory = McpFactory {
            lambda: 0.2,
            gamma: 3.0,
        };
        let cfg = GlassoConfig {
            diag_offset: 0.2,
            ..GlassoConfig::default()
        };
        let (theta, _, report) = glasso_solve(s.view(), None, &factory, &cfg);
        assert!(report.converged);
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert_abs_diff_eq!(theta[[i, j]], theta[[j, i]], epsilon = 1e-12);
            }
        }
    }

    /// Edge weights of zero on a specific edge must not penalise it
    /// (the inner lasso treats that coordinate as free).
    #[test]
    fn zero_edge_weight_means_no_penalty_on_that_edge() {
        let s = array![[1.0, 0.5], [0.5, 1.0]];
        let lam = 10.0; // huge λ — would zero everything by default
        let weights = array![[0.0, 0.0], [0.0, 0.0]]; // no penalty anywhere
        let factory = LassoFactory { lambda: lam };
        let cfg = GlassoConfig {
            diag_offset: lam,
            outer_tol: 1e-8,
            inner: CdConfig {
                max_iter: 5000,
                tol: 1e-10,
                acceleration: None,
            },
            ..GlassoConfig::default()
        };
        let (theta, _, _) = glasso_solve(s.view(), Some(weights.view()), &factory, &cfg);
        // With no L1 penalty, Θ ≈ (S + diag_offset · I)⁻¹.
        // S + 10·I = [[11, 0.5], [0.5, 11]], det ≈ 120.75
        // Θ ≈ (1/120.75)·[[11, -0.5], [-0.5, 11]]
        assert_abs_diff_eq!(theta[[0, 1]], -0.5 / 120.75, epsilon = 1e-4);
    }
}
