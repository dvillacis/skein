//! Scalar and group proximal operators for nonconvex penalties.
//!
//! Conventions:
//! - `step` is the prox step size `t`. We solve `argmin_x { (1/(2t))(x-z)² + p(x) }`.
//! - `weight` multiplies the penalty: effective penalty is `weight · p(x)`.
//!   This is the single hook every weighted variant (adaptive, per-feature,
//!   per-group) plugs into.

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
}
