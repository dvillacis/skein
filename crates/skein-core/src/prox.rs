//! Scalar and group proximal operators for nonconvex penalties.
//!
//! Conventions:
//! - `step` is the prox step size `t`. We solve `argmin_x { (1/(2t))(x-z)² + p(x) }`.
//! - `weight` multiplies the penalty: effective penalty is `weight · p(x)`.
//!   This is the single hook every weighted variant (adaptive, per-feature,
//!   per-group) plugs into.
//!
//! Also includes [`logdet_eigen_prox`] — the closed-form prox of the
//! log-det barrier `−c log det Θ + tr(S Θ)` with a Frobenius proximal
//! term. Used by the joint graphical-lasso ADMM solver. Backed by a
//! self-contained Jacobi eigendecomposition ([`jacobi_eigh`]) so the
//! crate doesn't have to depend on `ndarray-linalg` / LAPACK for what
//! is, in this use case, a small-matrix routine.

use ndarray::{Array1, Array2, ArrayView2};

/// Soft-threshold: prox of `weight·λ·|x|` at step `step`.
#[inline]
pub fn soft_threshold(z: f64, step: f64, lambda: f64, weight: f64) -> f64 {
    let thr = step * lambda * weight;
    if z > thr {
        z - thr
    } else if z < -thr {
        z + thr
    } else {
        0.0
    }
}

/// MCP scalar prox.
///
/// Penalty: `p(x; λ, γ) = λ|x| - x²/(2γ)` for `|x| ≤ γλ`, else `γλ²/2`.
/// Assumes the typical regime `γ > step`; for `γ ≤ step` the prox is
/// non-unique and we return the hard-threshold convention.
pub fn mcp_prox(z: f64, step: f64, lambda: f64, gamma: f64, weight: f64) -> f64 {
    let lam = lambda * weight;
    let abs_z = z.abs();
    if abs_z >= gamma * lam {
        return z;
    }
    if gamma > step {
        let s = (abs_z - step * lam).max(0.0);
        z.signum() * s / (1.0 - step / gamma)
    } else {
        let cutoff = (2.0 * step * gamma * lam * lam / 2.0).sqrt();
        if abs_z > cutoff {
            z
        } else {
            0.0
        }
    }
}

/// SCAD scalar prox with shape parameter `a > 2`.
pub fn scad_prox(z: f64, step: f64, lambda: f64, a: f64, weight: f64) -> f64 {
    let lam = lambda * weight;
    let abs_z = z.abs();
    if abs_z <= (1.0 + step) * lam {
        soft_threshold(z, step, lambda, weight)
    } else if abs_z <= a * lam {
        let denom = 1.0 - step / (a - 1.0);
        let num = abs_z - step * a * lam / (a - 1.0);
        z.signum() * num / denom
    } else {
        z
    }
}

/// Elastic-net scalar prox.
///
/// Penalty: `p(x; λ, α) = α λ |x| + (1-α) λ x² / 2`, weighted by
/// `weight`. Mixes lasso (α=1) and ridge (α=0); α ∈ (0, 1) gives the
/// classical glmnet elastic net. The prox is closed-form: soft-
/// threshold the L1 part, then divide by the ridge shrinkage factor.
///
/// Reduces exactly to `soft_threshold` at α=1 and to `z / (1 + step ·
/// λ · weight)` at α=0.
pub fn elastic_net_prox(z: f64, step: f64, lambda: f64, alpha: f64, weight: f64) -> f64 {
    let l1_thr = step * alpha * lambda * weight;
    let ridge_shrink = 1.0 + step * (1.0 - alpha) * lambda * weight;
    let st = if z > l1_thr {
        z - l1_thr
    } else if z < -l1_thr {
        z + l1_thr
    } else {
        0.0
    };
    st / ridge_shrink
}

/// Group prox of `weight · ‖x_g‖₂` at step `step` (block soft-threshold).
/// Modifies `block` in place.
pub fn group_soft_threshold(block: &mut [f64], step: f64, lambda: f64, weight: f64) {
    let norm: f64 = block.iter().map(|x| x * x).sum::<f64>().sqrt();
    let thr = step * lambda * weight;
    if norm <= thr {
        for x in block.iter_mut() {
            *x = 0.0;
        }
    } else {
        let scale = 1.0 - thr / norm;
        for x in block.iter_mut() {
            *x *= scale;
        }
    }
}

/// Group elastic-net prox at step `step`.
///
/// Penalty: `α λ w ‖x‖₂ + (1-α) λ w ‖x‖₂² / 2`. The mixed group-L2 + ridge
/// objective is rotationally symmetric in `x` aside from the data-fidelity
/// term, so the solution lies on the ray from the origin through `z`. Let
/// `r = ‖x‖`; minimizing `(1/(2t))(r-‖z‖)² + αλwr + (1-α)λwr²/2` over
/// `r ≥ 0` gives a block soft-threshold followed by ridge shrinkage —
/// the exact group analogue of [`elastic_net_prox`].
///
/// Reduces to [`group_soft_threshold`] at α=1 and to per-block ridge
/// shrinkage `x / (1 + step·λ·weight)` at α=0.
pub fn group_elastic_net_prox(block: &mut [f64], step: f64, lambda: f64, alpha: f64, weight: f64) {
    let norm: f64 = block.iter().map(|x| x * x).sum::<f64>().sqrt();
    let l1_thr = step * alpha * lambda * weight;
    let ridge_shrink = 1.0 + step * (1.0 - alpha) * lambda * weight;
    if norm <= l1_thr {
        for x in block.iter_mut() {
            *x = 0.0;
        }
    } else {
        let scale = (1.0 - l1_thr / norm) / ridge_shrink;
        for x in block.iter_mut() {
            *x *= scale;
        }
    }
}

/// Symmetric eigendecomposition `A = Q diag(d) Qᵀ` via cyclic Jacobi
/// rotations (Numerical Recipes §11.1). Returns `(d, Q)`.
///
/// Self-contained — avoids a LAPACK dependency for the small (`p`
/// typically ≤ a few hundred) symmetric eigendecompositions the joint
/// graphical lasso ADMM solver needs. Iterates until the sum of
/// absolute off-diagonal entries falls below `tol · ‖A‖_F` or
/// `max_sweeps` is reached.
///
/// Panics if `A` is not square.
pub fn jacobi_eigh(a: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    assert_eq!(a.ncols(), n, "jacobi_eigh: input must be square");
    let mut a = a.to_owned();
    let mut v = Array2::<f64>::eye(n);
    if n <= 1 {
        let d = Array1::from_iter(a.diag().iter().copied());
        return (d, v);
    }

    let max_sweeps = 100_usize;
    let tol = 1e-14;
    let frob = {
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                s += a[[i, j]] * a[[i, j]];
            }
        }
        s.sqrt()
    };
    let stop = tol * frob.max(1.0);

    for _sweep in 0..max_sweeps {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[[i, j]].abs();
            }
        }
        if off < stop {
            break;
        }

        for p in 0..(n - 1) {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[[p, p]];
                let aqq = a[[q, q]];
                let h = aqq - app;
                let t = if h.abs() > 1e100 * apq.abs() {
                    apq / h
                } else {
                    let theta = 0.5 * h / apq;
                    let mag = 1.0 / (theta.abs() + (1.0 + theta * theta).sqrt());
                    if theta < 0.0 {
                        -mag
                    } else {
                        mag
                    }
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                a[[p, p]] = app - t * apq;
                a[[q, q]] = aqq + t * apq;
                a[[p, q]] = 0.0;
                a[[q, p]] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[[r, p]];
                    let arq = a[[r, q]];
                    let new_rp = arp - s * (arq + tau * arp);
                    let new_rq = arq + s * (arp - tau * arq);
                    a[[r, p]] = new_rp;
                    a[[p, r]] = new_rp;
                    a[[r, q]] = new_rq;
                    a[[q, r]] = new_rq;
                }
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = vrp - s * (vrq + tau * vrp);
                    v[[r, q]] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    let d = Array1::from_iter((0..n).map(|i| a[[i, i]]));
    (d, v)
}

/// Closed-form prox of the log-det barrier with a Frobenius proximal
/// term:
///
/// ```text
///     Θ̂ = argmin_{Θ ≻ 0}  −c · log det Θ + tr(S Θ)
///                          + (ρ/2) · ‖Θ − V‖_F²,
/// ```
///
/// solved via the symmetric eigendecomposition of `M = ρV − cS`. The
/// stationarity condition `ρΘ − cΘ⁻¹ = M` is diagonalised by `M = Q D
/// Qᵀ`, giving `Θ = Q diag(θ_i) Qᵀ` with
///
/// ```text
///     θ_i = (d_i + √(d_i² + 4 ρ c)) / (2 ρ).
/// ```
///
/// `m_scaled` is the precomputed `M = ρ V − c S`. The caller is
/// responsible for forming it (cheap, and lets the ADMM outer loop
/// reuse buffers).
///
/// Panics on `rho ≤ 0`, `c ≤ 0`, or a non-square input.
pub fn logdet_eigen_prox(m_scaled: ArrayView2<f64>, rho: f64, c: f64) -> Array2<f64> {
    assert!(rho > 0.0, "logdet_eigen_prox: rho must be positive");
    assert!(c > 0.0, "logdet_eigen_prox: c must be positive");
    let n = m_scaled.nrows();
    assert_eq!(
        m_scaled.ncols(),
        n,
        "logdet_eigen_prox: input must be square"
    );

    // Symmetrise defensively — the ADMM outer loop sometimes feeds a
    // numerically near-symmetric matrix and we want to feed exactly-
    // symmetric data into jacobi_eigh.
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym[[i, j]] = 0.5 * (m_scaled[[i, j]] + m_scaled[[j, i]]);
        }
    }

    let (d, q) = jacobi_eigh(sym.view());
    let mut theta_diag = Array1::<f64>::zeros(n);
    let four_rho_c = 4.0 * rho * c;
    let two_rho = 2.0 * rho;
    for i in 0..n {
        let di = d[i];
        // Positive root of ρ θ² − d_i θ − c = 0 ⇒ θ > 0 always.
        theta_diag[i] = (di + (di * di + four_rho_c).sqrt()) / two_rho;
    }

    // Θ = Q · diag(θ) · Qᵀ. Form `q_scaled = Q · diag(θ)` then multiply
    // by `Qᵀ`.
    let mut q_scaled = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            q_scaled[[i, j]] = q[[i, j]] * theta_diag[j];
        }
    }
    q_scaled.dot(&q.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn soft_threshold_zero_below() {
        assert_abs_diff_eq!(soft_threshold(0.3, 1.0, 0.5, 1.0), 0.0);
    }

    #[test]
    fn soft_threshold_shrinks() {
        assert_abs_diff_eq!(soft_threshold(1.0, 1.0, 0.3, 1.0), 0.7, epsilon = 1e-12);
        assert_abs_diff_eq!(soft_threshold(-1.0, 1.0, 0.3, 1.0), -0.7, epsilon = 1e-12);
    }

    #[test]
    fn mcp_identity_outside_kink() {
        // |z| > γλ ⇒ prox is identity
        assert_abs_diff_eq!(mcp_prox(5.0, 1.0, 0.5, 3.0, 1.0), 5.0);
    }

    #[test]
    fn mcp_zero_at_origin() {
        assert_abs_diff_eq!(mcp_prox(0.1, 1.0, 0.5, 3.0, 1.0), 0.0);
    }

    #[test]
    fn mcp_weight_scales_threshold() {
        // weight=2 should zero out z=0.7 since effective λ = 1.0
        assert_abs_diff_eq!(mcp_prox(0.7, 1.0, 0.5, 3.0, 2.0), 0.0);
    }

    #[test]
    fn scad_matches_soft_threshold_in_lasso_regime() {
        // |z| ≤ (1+t)λ ⇒ SCAD prox = soft threshold
        let z = 0.6;
        let step = 1.0;
        let lambda = 0.5;
        let a = 3.7;
        assert_abs_diff_eq!(
            scad_prox(z, step, lambda, a, 1.0),
            soft_threshold(z, step, lambda, 1.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn elastic_net_alpha_one_matches_soft_threshold() {
        // α = 1 is pure lasso, so the EN prox must equal soft_threshold
        // bit-for-bit at every step / λ / weight.
        for &z in &[-2.0_f64, -0.6, -0.3, 0.0, 0.3, 0.6, 2.0] {
            for &step in &[0.5_f64, 1.0, 2.0] {
                for &lam in &[0.1_f64, 0.5, 1.0] {
                    for &w in &[0.5_f64, 1.0, 2.0] {
                        assert_abs_diff_eq!(
                            elastic_net_prox(z, step, lam, 1.0, w),
                            soft_threshold(z, step, lam, w),
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn elastic_net_alpha_zero_is_pure_ridge_shrinkage() {
        // α = 0 is pure ridge: prox(z) = z / (1 + step · λ · weight),
        // no soft-thresholding at all.
        for &z in &[-2.0_f64, -0.1, 0.0, 0.1, 2.0] {
            for &step in &[0.5_f64, 1.0] {
                for &lam in &[0.1_f64, 1.0] {
                    for &w in &[0.5_f64, 1.0] {
                        let expected = z / (1.0 + step * lam * w);
                        assert_abs_diff_eq!(
                            elastic_net_prox(z, step, lam, 0.0, w),
                            expected,
                            epsilon = 1e-12
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn elastic_net_zeroes_below_threshold() {
        // Same threshold rule as soft_threshold: |z| ≤ step·α·λ·w → 0.
        // After scaling-down by the ridge factor, 0 stays 0.
        let alpha = 0.5;
        let step = 1.0;
        let lam = 0.4;
        let w = 1.0;
        // step·α·λ·w = 0.2 → z = 0.15 lands at 0.
        assert_abs_diff_eq!(
            elastic_net_prox(0.15, step, lam, alpha, w),
            0.0,
            epsilon = 1e-12
        );
        // Just above the L1 threshold: small but non-zero.
        let z = 0.25;
        let expected = (z - step * alpha * lam * w) / (1.0 + step * (1.0 - alpha) * lam * w);
        assert_abs_diff_eq!(
            elastic_net_prox(z, step, lam, alpha, w),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn elastic_net_weight_scales_both_terms() {
        // Doubling the weight should double both the L1 threshold and
        // the ridge shrinkage rate.
        let z = 1.0;
        let step = 1.0;
        let lam = 0.3;
        let alpha = 0.4;
        let p_w1 = elastic_net_prox(z, step, lam, alpha, 1.0);
        // With w=2, threshold doubles to 0.24; ridge factor goes from
        // 1.18 to 1.36. Hand-compute:
        let l1_thr = step * alpha * lam * 2.0; // 0.24
        let ridge = 1.0 + step * (1.0 - alpha) * lam * 2.0; // 1.36
        let expected = (z - l1_thr) / ridge;
        assert_abs_diff_eq!(
            elastic_net_prox(z, step, lam, alpha, 2.0),
            expected,
            epsilon = 1e-12
        );
        // And not equal to the w=1 case (sanity-check the test isn't tautological).
        assert!((p_w1 - expected).abs() > 1e-3);
    }

    #[test]
    fn group_soft_threshold_zeroes_small_blocks() {
        let mut block = vec![0.1, 0.1, 0.1];
        group_soft_threshold(&mut block, 1.0, 1.0, 1.0);
        assert!(block.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn group_soft_threshold_shrinks_large_blocks() {
        let mut block = vec![3.0, 4.0]; // norm = 5
        group_soft_threshold(&mut block, 1.0, 1.0, 1.0);
        // scale = 1 - 1/5 = 0.8
        assert_abs_diff_eq!(block[0], 2.4, epsilon = 1e-12);
        assert_abs_diff_eq!(block[1], 3.2, epsilon = 1e-12);
    }

    #[test]
    fn group_elastic_net_alpha_one_matches_group_soft_threshold() {
        // α = 1 ⇒ pure group lasso: must equal group_soft_threshold elementwise.
        for &(b0, b1) in &[(3.0_f64, 4.0), (-1.0, 0.5), (0.05, 0.05)] {
            for &step in &[0.5_f64, 1.0, 2.0] {
                for &lam in &[0.1_f64, 0.5, 2.0] {
                    for &w in &[0.5_f64, 1.0, 2.0] {
                        let mut a = vec![b0, b1];
                        let mut b = vec![b0, b1];
                        group_elastic_net_prox(&mut a, step, lam, 1.0, w);
                        group_soft_threshold(&mut b, step, lam, w);
                        assert_abs_diff_eq!(a[0], b[0], epsilon = 1e-12);
                        assert_abs_diff_eq!(a[1], b[1], epsilon = 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn group_elastic_net_alpha_zero_is_pure_block_ridge() {
        // α = 0 ⇒ pure ridge per block: each entry shrunk by
        // 1 / (1 + step·λ·weight). No thresholding.
        for &(b0, b1) in &[(3.0_f64, 4.0), (-0.1, 2.0), (0.0, 1.0)] {
            for &step in &[0.5_f64, 1.0] {
                for &lam in &[0.1_f64, 1.0] {
                    for &w in &[0.5_f64, 1.0, 3.0] {
                        let mut block = vec![b0, b1];
                        group_elastic_net_prox(&mut block, step, lam, 0.0, w);
                        let factor = 1.0 + step * lam * w;
                        assert_abs_diff_eq!(block[0], b0 / factor, epsilon = 1e-12);
                        assert_abs_diff_eq!(block[1], b1 / factor, epsilon = 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn group_elastic_net_zeroes_small_blocks() {
        // ‖block‖ = 0.1, threshold = step·α·λ·w = 0.5·1·1·1 = 0.5 ⇒ zero.
        let mut block = vec![0.06, 0.08]; // norm = 0.1
        group_elastic_net_prox(&mut block, 0.5, 1.0, 1.0, 1.0);
        assert!(block.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn group_elastic_net_shrinks_large_blocks() {
        // norm = 5; α = 0.5, step = 1, λ = 1, w = 1.
        // l1_thr = 0.5; ridge_shrink = 1 + 0.5 = 1.5.
        // scale = (1 - 0.5/5) / 1.5 = 0.9 / 1.5 = 0.6.
        let mut block = vec![3.0, 4.0];
        group_elastic_net_prox(&mut block, 1.0, 1.0, 0.5, 1.0);
        assert_abs_diff_eq!(block[0], 1.8, epsilon = 1e-12);
        assert_abs_diff_eq!(block[1], 2.4, epsilon = 1e-12);
    }

    // ---- Jacobi eigendecomp + log-det prox -----------------------------

    #[test]
    fn jacobi_eigh_diagonal_returns_diagonal() {
        let a = ndarray::array![[2.0_f64, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, -1.0]];
        let (d, _q) = jacobi_eigh(a.view());
        let mut sorted: Vec<f64> = d.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_abs_diff_eq!(sorted[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sorted[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sorted[2], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn jacobi_eigh_reconstructs_input() {
        let a = ndarray::array![
            [4.0_f64, 1.0, 0.2, 0.1],
            [1.0, 3.0, 0.3, 0.2],
            [0.2, 0.3, 2.5, 0.4],
            [0.1, 0.2, 0.4, 1.7]
        ];
        let (d, q) = jacobi_eigh(a.view());
        // Reconstruct: Q diag(d) Qᵀ should equal A.
        let mut q_diag = q.clone();
        for j in 0..4 {
            for i in 0..4 {
                q_diag[[i, j]] *= d[j];
            }
        }
        let recon = q_diag.dot(&q.t());
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(recon[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
        // Q should be orthogonal.
        let qtq = q.t().dot(&q);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn logdet_eigen_prox_satisfies_stationarity() {
        // ρ Θ − c Θ⁻¹ = M  must hold at the returned Θ.
        let m = ndarray::array![[3.0_f64, 0.5], [0.5, 2.0]];
        let rho = 1.5;
        let c = 2.0;
        let theta = logdet_eigen_prox(m.view(), rho, c);
        // Compute Θ⁻¹ via 2×2 closed form.
        let det = theta[[0, 0]] * theta[[1, 1]] - theta[[0, 1]] * theta[[1, 0]];
        let theta_inv = ndarray::array![
            [theta[[1, 1]] / det, -theta[[0, 1]] / det],
            [-theta[[1, 0]] / det, theta[[0, 0]] / det],
        ];
        let lhs = &(&theta * rho) - &(&theta_inv * c);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(lhs[[i, j]], m[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn logdet_eigen_prox_is_positive_definite() {
        // For any M, the prox solution must be PD (positive root chosen).
        let m = ndarray::array![[-1.0_f64, 0.0], [0.0, -2.0]];
        let theta = logdet_eigen_prox(m.view(), 1.0, 1.0);
        // Diagonals must be positive; determinant positive.
        assert!(theta[[0, 0]] > 0.0);
        assert!(theta[[1, 1]] > 0.0);
        let det = theta[[0, 0]] * theta[[1, 1]] - theta[[0, 1]] * theta[[1, 0]];
        assert!(det > 0.0);
    }

    #[test]
    fn group_elastic_net_weight_scales_both_terms() {
        // Doubling weight: l1 threshold doubles, ridge factor's λw term doubles.
        let block_init = [1.0_f64, 2.0]; // norm = √5 ≈ 2.236
        let step = 1.0;
        let lam = 0.3;
        let alpha = 0.4;

        let mut b1 = block_init.to_vec();
        group_elastic_net_prox(&mut b1, step, lam, alpha, 1.0);

        let mut b2 = block_init.to_vec();
        group_elastic_net_prox(&mut b2, step, lam, alpha, 2.0);

        // Hand compute w=2:
        let norm = (5.0_f64).sqrt();
        let l1_thr = step * alpha * lam * 2.0; // 0.24
        let ridge = 1.0 + step * (1.0 - alpha) * lam * 2.0; // 1.36
        let scale = (1.0 - l1_thr / norm) / ridge;
        assert_abs_diff_eq!(b2[0], block_init[0] * scale, epsilon = 1e-12);
        assert_abs_diff_eq!(b2[1], block_init[1] * scale, epsilon = 1e-12);
        // And b1 ≠ b2.
        assert!((b1[0] - b2[0]).abs() > 1e-3);
    }
}
