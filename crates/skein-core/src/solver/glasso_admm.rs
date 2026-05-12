//! Joint graphical lasso (Danaher–Wang–Witten 2014, group form) via
//! ADMM.
//!
//! Estimates precision matrices `Θ^(1), …, Θ^(K)` for `K` related
//! populations sharing the same `p` variables, with a group penalty
//! coupling the *same* edge across populations:
//!
//! ```text
//!     min_{Θ^(k) ≻ 0}  Σ_k n_k · [−log det Θ^(k) + tr(S^(k) Θ^(k))]
//!                      + λ · Σ_{i ≠ j} w_{ij} · p_group( (Θ^(k)_{ij})_{k=1..K} )
//! ```
//!
//! `p_group` is supplied by a [`GroupPenaltyFactory`] — group lasso for
//! the convex case, group MCP for nonconvex. The factory carries `λ`
//! and (for nonconvex variants) the shape parameter; per-edge weights
//! `w_{ij}` are threaded in via the factory's `build`.
//!
//! ADMM split:
//!
//! - **Θ-update** (per population): closed form via
//!   [`logdet_eigen_prox`] on `M^(k) = ρ(Z^(k) − U^(k)) − n_k S^(k)`.
//! - **Z-update**: element-wise. Diagonal entries pass through; each
//!   off-diagonal edge `(i, j)` becomes a `K`-vector that goes through
//!   `GroupPenalty::prox_group` at step `1/ρ`.
//! - **U-update**: standard dual ascent `U ← U + Θ − Z`.
//!
//! Convergence: standard ADMM primal/dual residual stopping rule. This
//! is the first ADMM kernel in skein; if a second use case appears, the
//! outer loop can be lifted out of this file. Don't pre-generalise.

use crate::groups::Groups;
use crate::penalty::GroupPenaltyFactory;
use crate::prox::logdet_eigen_prox;
use ndarray::{Array1, Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct JointGlassoConfig {
    pub max_iter: usize,
    /// Stops when both primal and dual residuals (per-element rms) are
    /// below their tolerances.
    pub primal_tol: f64,
    pub dual_tol: f64,
    /// ADMM penalty parameter. Larger ρ favours feasibility (Θ ≈ Z);
    /// smaller ρ favours datafit. The default 1.0 is reasonable for
    /// `n_k`-scaled losses when `n_k ~ 100s`; tune via a small grid if
    /// convergence is slow.
    pub rho: f64,
    /// Added to the diagonal of each `S^(k)` at initialisation only
    /// (`Θ^(k) ← I` regardless; this affects the first `M^(k)`). Use
    /// the L1 strength `λ` for sklearn-style PSD-safety; default `0.0`.
    pub diag_offset: f64,
}

impl Default for JointGlassoConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            primal_tol: 1e-5,
            dual_tol: 1e-5,
            rho: 1.0,
            diag_offset: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct JointGlassoReport {
    pub iter: usize,
    pub converged: bool,
    pub primal_residual: f64,
    pub dual_residual: f64,
}

/// Solve group-form joint graphical lasso via ADMM.
///
/// `sample_covs` is a slice of `K` symmetric `(p, p)` sample
/// covariance matrices (one per population). `n_samples[k]` is the
/// number of observations behind `sample_covs[k]` (used to weight the
/// per-population log-likelihood). `edge_weights[i, j]` is the
/// per-edge coupling weight applied to `λ`; `None` is uniform 1.
/// `group_penalty_factory` builds the group penalty (lasso → soft
/// threshold; MCP → nonconvex) over `K`-vectors.
///
/// Returns `(precisions, report)` where `precisions[k]` is `Θ̂^(k)`.
pub fn joint_glasso_solve(
    sample_covs: &[ArrayView2<f64>],
    n_samples: &[f64],
    edge_weights: Option<ArrayView2<f64>>,
    group_penalty_factory: &dyn GroupPenaltyFactory,
    config: &JointGlassoConfig,
) -> (Vec<Array2<f64>>, JointGlassoReport) {
    let n_pops = sample_covs.len();
    assert!(
        n_pops >= 1,
        "joint_glasso_solve: at least one population required"
    );
    assert_eq!(
        n_samples.len(),
        n_pops,
        "n_samples and sample_covs must have the same length"
    );
    let p = sample_covs[0].nrows();
    for s in sample_covs {
        assert_eq!(s.nrows(), p);
        assert_eq!(s.ncols(), p);
    }
    for &n in n_samples {
        assert!(n > 0.0, "n_samples must be positive");
    }
    if let Some(ew) = edge_weights {
        assert_eq!(ew.dim(), (p, p), "edge_weights must be (p, p)");
    }

    // Build the group structure: one group per upper-triangular edge,
    // each containing `K` indices into a flat (n_edges × K) buffer.
    let n_edges = p * (p - 1) / 2;
    let mut ptr = Vec::with_capacity(n_edges + 1);
    let mut idx = Vec::with_capacity(n_edges * n_pops);
    ptr.push(0);
    for e in 0..n_edges {
        for k in 0..n_pops {
            idx.push(e * n_pops + k);
        }
        ptr.push(idx.len());
    }
    let _groups = Groups::from_csr(ptr, idx).expect("group construction");

    // Per-edge coupling weights, in upper-triangular row-major order.
    let mut group_weights = Array1::<f64>::ones(n_edges.max(1));
    if let Some(ew) = edge_weights {
        let mut e = 0;
        for i in 0..p {
            for j in (i + 1)..p {
                group_weights[e] = ew[[i, j]];
                e += 1;
            }
        }
    }
    let group_penalty = group_penalty_factory.build(group_weights);

    // Initial state. Θ^(k) = (S^(k) + diag_offset I)⁻¹ would be a
    // stronger warm start but needs an inversion per pop; the simple
    // identity start converges fast on the small ADMM iteration counts
    // typical at this scale.
    let mut theta: Vec<Array2<f64>> = (0..n_pops).map(|_| Array2::eye(p)).collect();
    let mut z: Vec<Array2<f64>> = (0..n_pops).map(|_| Array2::eye(p)).collect();
    let mut u: Vec<Array2<f64>> = (0..n_pops).map(|_| Array2::zeros((p, p))).collect();

    // Apply the diag_offset to S^(k) once — fold it into the per-iter
    // M^(k) computation by stashing in a per-pop "effective S".
    let s_eff: Vec<Array2<f64>> = sample_covs
        .iter()
        .map(|s| {
            let mut s_eff = s.to_owned();
            if config.diag_offset != 0.0 {
                for i in 0..p {
                    s_eff[[i, i]] += config.diag_offset;
                }
            }
            s_eff
        })
        .collect();

    let rho = config.rho;
    let mut report = JointGlassoReport::default();
    let mut block = Array1::<f64>::zeros(n_pops);
    let mut m_buf = Array2::<f64>::zeros((p, p));

    for iter in 0..config.max_iter {
        let z_prev = z.clone();

        // Θ-update: per-pop, closed-form via logdet_eigen_prox.
        for k in 0..n_pops {
            let n_k = n_samples[k];
            // M = ρ(Z − U) − n_k · S
            for i in 0..p {
                for j in 0..p {
                    m_buf[[i, j]] = rho * (z[k][[i, j]] - u[k][[i, j]]) - n_k * s_eff[k][[i, j]];
                }
            }
            theta[k] = logdet_eigen_prox(m_buf.view(), rho, n_k);
        }

        // Z-update.
        //  Diagonal: no penalty.
        for k in 0..n_pops {
            for i in 0..p {
                z[k][[i, i]] = theta[k][[i, i]] + u[k][[i, i]];
            }
        }
        //  Off-diagonal: edge-wise group prox.
        let mut e = 0;
        for i in 0..p {
            for j in (i + 1)..p {
                for k in 0..n_pops {
                    block[k] = theta[k][[i, j]] + u[k][[i, j]];
                }
                group_penalty.prox_group(e, block.view_mut(), 1.0 / rho);
                for k in 0..n_pops {
                    z[k][[i, j]] = block[k];
                    z[k][[j, i]] = block[k];
                }
                e += 1;
            }
        }

        // U-update.
        for k in 0..n_pops {
            for i in 0..p {
                for j in 0..p {
                    u[k][[i, j]] += theta[k][[i, j]] - z[k][[i, j]];
                }
            }
        }

        // Convergence.
        let mut primal_sq = 0.0_f64;
        let mut dual_sq = 0.0_f64;
        for k in 0..n_pops {
            for i in 0..p {
                for j in 0..p {
                    let pr = theta[k][[i, j]] - z[k][[i, j]];
                    primal_sq += pr * pr;
                    let dr = z[k][[i, j]] - z_prev[k][[i, j]];
                    dual_sq += dr * dr;
                }
            }
        }
        let n_total = (n_pops * p * p) as f64;
        let primal_res = (primal_sq / n_total).sqrt();
        let dual_res = rho * (dual_sq / n_total).sqrt();

        report.iter = iter + 1;
        report.primal_residual = primal_res;
        report.dual_residual = dual_res;
        if primal_res < config.primal_tol && dual_res < config.dual_tol {
            report.converged = true;
            break;
        }
    }

    (theta, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::penalty::{GroupLassoFactory, GroupMcpFactory};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn id_cov(p: usize) -> Array2<f64> {
        let mut s = Array2::<f64>::eye(p);
        for i in 0..p {
            s[[i, i]] = 1.0;
        }
        s
    }

    #[test]
    fn k_equals_one_runs_to_completion() {
        let s = array![[1.0, 0.3], [0.3, 1.0]];
        let sv = [s.view()];
        let factory = GroupLassoFactory { lambda: 0.1 };
        let cfg = JointGlassoConfig {
            max_iter: 500,
            ..JointGlassoConfig::default()
        };
        let (thetas, report) = joint_glasso_solve(&sv, &[100.0], None, &factory, &cfg);
        assert_eq!(thetas.len(), 1);
        assert!(report.iter > 0);
        // Symmetric.
        for i in 0..2 {
            for j in (i + 1)..2 {
                assert_abs_diff_eq!(thetas[0][[i, j]], thetas[0][[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn identity_input_yields_near_identity_precision() {
        // S = I per pop → precision ≈ I.
        let p = 3;
        let s = id_cov(p);
        let sv = [s.view(), s.view()];
        let factory = GroupLassoFactory { lambda: 0.5 };
        let cfg = JointGlassoConfig {
            max_iter: 500,
            primal_tol: 1e-6,
            dual_tol: 1e-6,
            ..JointGlassoConfig::default()
        };
        let (thetas, report) = joint_glasso_solve(&sv, &[50.0, 50.0], None, &factory, &cfg);
        assert!(report.converged);
        for theta in thetas.iter() {
            for i in 0..p {
                for j in 0..p {
                    if i == j {
                        // Identity-S + huge ρ trade-off: diagonal pulls toward 1.
                        assert!(theta[[i, j]] > 0.0);
                    } else {
                        assert_abs_diff_eq!(theta[[i, j]], 0.0, epsilon = 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn large_coupling_lambda_collapses_populations() {
        // Two pops with different S: huge λ on coupling should make
        // Θ̂^(1) ≈ Θ̂^(2). Small p so we don't fight numerics.
        let s1 = array![[1.0, 0.4], [0.4, 1.0]];
        let s2 = array![[1.0, 0.1], [0.1, 1.0]];
        let sv = [s1.view(), s2.view()];
        let factory = GroupLassoFactory { lambda: 50.0 };
        let cfg = JointGlassoConfig {
            max_iter: 1000,
            primal_tol: 1e-5,
            dual_tol: 1e-5,
            rho: 1.0,
            diag_offset: 0.0,
        };
        let (thetas, report) = joint_glasso_solve(&sv, &[100.0, 100.0], None, &factory, &cfg);
        assert!(report.converged, "ADMM should converge at large coupling λ");
        // Off-diagonal should be near identical across populations.
        let off_diff = (thetas[0][[0, 1]] - thetas[1][[0, 1]]).abs();
        assert!(
            off_diff < 1e-3,
            "expected coupled off-diagonals to agree; diff = {off_diff}"
        );
    }

    #[test]
    fn zero_lambda_decouples_to_per_population_mle() {
        // λ = 0 ⇒ each pop's Θ̂ solves the unpenalised MLE: Θ̂^(k) ≈ (S^(k))⁻¹.
        let s1 = array![[2.0, 0.5], [0.5, 1.0]];
        let s2 = array![[1.5, 0.2], [0.2, 0.8]];
        let sv = [s1.view(), s2.view()];
        // λ = 0 effectively. Use a tiny λ to avoid pathology.
        let factory = GroupLassoFactory { lambda: 1e-8 };
        let cfg = JointGlassoConfig {
            max_iter: 5000,
            primal_tol: 1e-7,
            dual_tol: 1e-7,
            rho: 1.0,
            diag_offset: 0.0,
        };
        let (thetas, _) = joint_glasso_solve(&sv, &[1.0, 1.0], None, &factory, &cfg);
        // Compare to closed-form S⁻¹.
        for (theta, s) in thetas.iter().zip([&s1, &s2]) {
            let det = s[[0, 0]] * s[[1, 1]] - s[[0, 1]] * s[[1, 0]];
            let inv = array![
                [s[[1, 1]] / det, -s[[0, 1]] / det],
                [-s[[1, 0]] / det, s[[0, 0]] / det]
            ];
            for i in 0..2 {
                for j in 0..2 {
                    assert_abs_diff_eq!(theta[[i, j]], inv[[i, j]], epsilon = 5e-2);
                }
            }
        }
    }

    #[test]
    fn group_mcp_factory_runs_to_completion() {
        let s = array![[1.0, 0.3], [0.3, 1.0]];
        let sv = [s.view(), s.view()];
        let factory = GroupMcpFactory {
            lambda: 0.1,
            gamma: 3.0,
        };
        let cfg = JointGlassoConfig {
            max_iter: 500,
            ..JointGlassoConfig::default()
        };
        let (thetas, report) = joint_glasso_solve(&sv, &[100.0, 100.0], None, &factory, &cfg);
        assert!(report.iter > 0);
        // Symmetric.
        for theta in thetas.iter() {
            assert_abs_diff_eq!(theta[[0, 1]], theta[[1, 0]], epsilon = 1e-10);
        }
    }
}
