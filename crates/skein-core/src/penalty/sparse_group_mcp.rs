use super::GroupPenalty;
use crate::groups::Groups;
use crate::prox::mcp_prox;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Sparse-group MCP penalty (Breheny & Huang 2015 §3).
///
/// `λ · (α · Σ_k v_{g,k} · MCP(|β_{g,k}|; γ) + (1−α) · w_g · MCP(‖β_g‖_2; γ))`
///
/// where the per-coordinate term contributes within-group sparsity and
/// the per-group L2 term contributes whole-group sparsity. Both MCP
/// shrinkage functions share the same nonconvexity parameter `γ`.
///
/// The per-group prox is exact (Breheny & Huang 2015 Proposition 1):
/// apply the closed-form scalar MCP prox to each coordinate of the
/// block with threshold `α · λ · v_{g,k}`, then apply the closed-form
/// group-MCP prox to the result with threshold `(1−α) · λ · w_g`.
/// The composition is exact because the per-coord penalty is sign-
/// invariant + coordinatewise separable and the L2 penalty depends
/// only on the resulting block norm.
///
/// This is the native non-convex penalty that replaces the LLA-wrapped
/// `SparseGroupLasso::with_coord_weights` surrogate used by the LS and
/// GLM paths in skein v0.7–v0.8 (`solver::lla::surrogate_sparse_group_mcp`).
/// Mirrors the M13.4b / M13.4c switch from LLA-wrapped GroupLasso to
/// native GroupMcp on the non-sparse group families.
pub struct SparseGroupMcp {
    lambda: f64,
    alpha: f64,
    gamma: f64,
    weights: Array1<f64>,
    /// Per-group, per-position-in-group L1 weights. `None` ⇒ uniform 1.
    coord_weights: Option<Vec<Array1<f64>>>,
}

impl SparseGroupMcp {
    pub fn new(lambda: f64, alpha: f64, gamma: f64, n_groups: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert!(gamma > 1.0, "gamma must be > 1 for MCP (got {})", gamma);
        Self {
            lambda,
            alpha,
            gamma,
            weights: Array1::ones(n_groups),
            coord_weights: None,
        }
    }

    pub fn with_weights(lambda: f64, alpha: f64, gamma: f64, weights: Array1<f64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert!(gamma > 1.0, "gamma must be > 1 for MCP (got {})", gamma);
        Self {
            lambda,
            alpha,
            gamma,
            weights,
            coord_weights: None,
        }
    }

    /// Two-level construction: per-group L2 weights plus per-group,
    /// per-position-in-group L1 weights. `coord_weights[g]` must have
    /// length equal to the size of group `g`. Mirrors
    /// `SparseGroupLasso::with_coord_weights`.
    pub fn with_coord_weights(
        lambda: f64,
        alpha: f64,
        gamma: f64,
        group_weights: Array1<f64>,
        coord_weights: Vec<Array1<f64>>,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert!(gamma > 1.0, "gamma must be > 1 for MCP (got {})", gamma);
        assert_eq!(
            group_weights.len(),
            coord_weights.len(),
            "coord_weights must have one entry per group"
        );
        Self {
            lambda,
            alpha,
            gamma,
            weights: group_weights,
            coord_weights: Some(coord_weights),
        }
    }
}

/// Scalar MCP penalty value `λ · w · MCP(|z|; γ)` for one coordinate.
#[inline]
fn mcp_scalar_value(z: f64, lambda: f64, gamma: f64, weight: f64) -> f64 {
    let lam = lambda * weight;
    let abs_z = z.abs();
    if abs_z >= gamma * lam {
        // Saturated regime: penalty = γλ²/2 (constant).
        gamma * lam * lam / 2.0
    } else {
        lam * abs_z - abs_z * abs_z / (2.0 * gamma)
    }
}

impl GroupPenalty for SparseGroupMcp {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64 {
        let mut total = 0.0;
        for g in 0..groups.n_groups() {
            let cols = groups.group(g);
            let coord_w = self.coord_weights.as_ref().map(|cw| &cw[g]);
            // Per-coord MCP.
            let mut coord_term = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for (k, &j) in cols.iter().enumerate() {
                let b = beta[j];
                let v_k = coord_w.map(|cw| cw[k]).unwrap_or(1.0);
                coord_term += mcp_scalar_value(b, self.lambda * self.alpha, self.gamma, v_k);
                sum_sq += b * b;
            }
            // Per-group L2-MCP.
            let block_l2 = sum_sq.sqrt();
            let group_term = mcp_scalar_value(
                block_l2,
                self.lambda * (1.0 - self.alpha),
                self.gamma,
                self.weights[g],
            );
            total += coord_term + group_term;
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let coord_w = self.coord_weights.as_ref().map(|cw| &cw[g]);

        // Step 1: per-coordinate scalar MCP prox.
        for (k, x) in block.iter_mut().enumerate() {
            let v_k = coord_w.map(|cw| cw[k]).unwrap_or(1.0);
            *x = mcp_prox(*x, step, self.lambda * self.alpha, self.gamma, v_k);
        }

        // Step 2: group-MCP block prox. Mirrors `GroupMcp::prox_group`
        // (Breheny & Huang 2015 §3 closed form) applied to the L2 norm
        // of the post-step-1 block.
        let slice = block.as_slice_mut().expect("contiguous block expected");
        let norm: f64 = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
        let lam_g = self.lambda * (1.0 - self.alpha) * self.weights[g];

        if norm >= self.gamma * lam_g {
            return; // Saturated: no group shrinkage.
        }
        if self.gamma > step {
            let thr = step * lam_g;
            if norm <= thr {
                for x in slice.iter_mut() {
                    *x = 0.0;
                }
            } else {
                let scale = (1.0 - thr / norm) / (1.0 - step / self.gamma);
                for x in slice.iter_mut() {
                    *x *= scale;
                }
            }
        } else {
            // Degenerate γ ≤ step: hard-threshold convention (matches
            // GroupMcp::prox_group).
            let cutoff = (step * self.gamma * lam_g * lam_g).sqrt();
            if norm <= cutoff {
                for x in slice.iter_mut() {
                    *x = 0.0;
                }
            }
        }
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::{DenseMatrix, DesignMatrix};
    use crate::penalty::{GroupMcp, SparseGroupLasso};
    use crate::solver::{block_cd_solve, CdConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    fn problem(seed: u64) -> (DenseMatrix, Array1<f64>, Groups) {
        let n = 120;
        let p = 8;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 1.5;
        true_beta[1] = -1.0;
        true_beta[4] = 0.7;
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), y, groups)
    }

    // ---- penalty value ----------------------------------------------

    #[test]
    fn sparse_group_mcp_value_zero_when_blocks_are_zero() {
        let beta = Array1::<f64>::zeros(4);
        let groups = Groups::contiguous_blocks(4, 2);
        let pen = SparseGroupMcp::new(0.1, 0.5, 3.0, 2);
        assert_abs_diff_eq!(pen.value(beta.view(), &groups), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn sparse_group_mcp_value_saturates_above_gamma_lambda() {
        // Large β well above γλ: per-coord MCP returns γλ²/2 per coord;
        // group MCP saturates similarly.
        let beta = array![10.0, -10.0];
        let groups = Groups::contiguous_blocks(2, 2);
        let pen = SparseGroupMcp::new(0.1, 0.5, 3.0, 1);
        let v = pen.value(beta.view(), &groups);
        // Coord contributions: 2 × (γ · (λα)² / 2) = 3 · (0.05)² = 0.0075.
        // Group L2 = sqrt(200) ≈ 14.14, well above γ · λ(1-α) = 3·0.05 = 0.15.
        // Group term: γ · (λ(1-α))² / 2 = 3 · (0.05)² / 2 = 0.00375.
        let expected = 2.0 * 3.0 * 0.05 * 0.05 / 2.0 + 3.0 * 0.05 * 0.05 / 2.0;
        assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
    }

    // ---- prox -------------------------------------------------------

    #[test]
    fn sparse_group_mcp_prox_zeros_block_under_strong_penalty() {
        // Large λ small block ⇒ block zeroed by step 2 (group MCP).
        let pen = SparseGroupMcp::new(10.0, 0.5, 3.0, 1);
        let mut block = array![0.3, -0.4];
        pen.prox_group(0, block.view_mut(), 1.0);
        for &v in block.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn sparse_group_mcp_prox_identity_above_kink() {
        // Block large enough to saturate BOTH per-coord and group MCPs:
        // each |β_k| > γ·λ·α·v_k AND ‖β‖_2 > γ·λ·(1-α)·w_g. Then prox
        // returns the block unchanged.
        let pen = SparseGroupMcp::new(0.1, 0.5, 3.0, 1);
        let mut block = array![5.0, -4.0];
        let block_before = block.clone();
        pen.prox_group(0, block.view_mut(), 1.0);
        for (a, b) in block.iter().zip(block_before.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    // ---- reductions to known special cases --------------------------

    #[test]
    fn sparse_group_mcp_with_large_gamma_matches_sparse_group_lasso() {
        // As γ → ∞, MCP → L1, so SparseGroupMCP → SparseGroupLasso at
        // the same α. Use γ = 1e8 to approximate.
        let (design, y, groups) = problem(11);
        let datafit = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let lambda = 0.05;
        let alpha = 0.5;

        let pen_mcp = SparseGroupMcp::new(lambda, alpha, 1e8, groups.n_groups());
        let pen_sgl = SparseGroupLasso::new(lambda, alpha, groups.n_groups());

        let (b_mcp, _) = block_cd_solve(&design, &datafit, &pen_mcp, &groups, &cfg);
        let (b_sgl, _) = block_cd_solve(&design, &datafit, &pen_sgl, &groups, &cfg);

        for j in 0..design.n_features() {
            assert_abs_diff_eq!(b_mcp[j], b_sgl[j], epsilon = 1e-5);
        }
    }

    #[test]
    fn sparse_group_mcp_with_alpha_zero_matches_group_mcp() {
        // α = 0 ⇒ only the per-group L2-MCP term remains; should match
        // GroupMcp at the same λ, γ.
        let (design, y, groups) = problem(12);
        let datafit = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let lambda = 0.05;
        let gamma = 3.0;

        let pen_sgm = SparseGroupMcp::new(lambda, 0.0, gamma, groups.n_groups());
        let pen_gm = GroupMcp::new(lambda, gamma, groups.n_groups());

        let (b_sgm, _) = block_cd_solve(&design, &datafit, &pen_sgm, &groups, &cfg);
        let (b_gm, _) = block_cd_solve(&design, &datafit, &pen_gm, &groups, &cfg);

        for j in 0..design.n_features() {
            assert_abs_diff_eq!(b_sgm[j], b_gm[j], epsilon = 1e-6);
        }
    }

    // ---- within-group sparsity recovery -----------------------------

    #[test]
    fn sparse_group_mcp_recovers_within_group_sparsity() {
        // Plant: only feature 0 active inside group 0; group 1 inactive.
        let n = 100;
        let p = 4;
        let mut state: u64 = 99;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let true_beta = array![2.0, 0.0, 0.0, 0.0];
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let groups = Groups::contiguous_blocks(p, 2);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };

        let pen = SparseGroupMcp::new(0.05, 0.5, 3.0, groups.n_groups());
        let (b, _) = block_cd_solve(&design, &datafit, &pen, &groups, &cfg);

        // Feature 0 large; features 1, 2, 3 small (within-group + whole-
        // group sparsity both active).
        assert!(b[0].abs() > 1.5, "feature 0 too small: {}", b[0]);
        for j in 1..p {
            assert!(b[j].abs() < 0.3, "feature {} too large: {}", j, b[j]);
        }
    }

    // ---- two-level construction (per-coord + per-group weights) -----

    #[test]
    fn sparse_group_mcp_coord_weights_zeros_specific_coord() {
        // Coord weight 100 on β_0 amplifies its per-coord MCP threshold
        // enough to zero it; β_1 lands in the shrinkage zone.
        // α = 1 so step 2 (group MCP) is identity (λ_g = 0).
        let group_w = array![1.0];
        let coord_w = vec![array![100.0, 1.0]];
        let pen = SparseGroupMcp::with_coord_weights(0.2, 1.0, 3.0, group_w, coord_w);
        let mut block = array![0.5, 0.5];
        pen.prox_group(0, block.view_mut(), 1.0);
        // β_0, weight 100: γ·λ·v_0 = 60; |0.5| < 60 ⇒ shrinkage zone.
        // s = (0.5 − step·λ·v_0).max(0) = (0.5 − 20).max(0) = 0 ⇒ β_0 = 0.
        assert_abs_diff_eq!(block[0], 0.0, epsilon = 1e-12);
        // β_1, weight 1: γ·λ·v_1 = 0.6; |0.5| < 0.6 ⇒ shrinkage zone.
        // s = (0.5 − step·λ·v_1).max(0) = 0.3; MCP correction
        // 1/(1 − step/γ) = 1/(2/3) = 1.5 ⇒ β_1 = 0.3 · 1.5 = 0.45.
        assert_abs_diff_eq!(block[1], 0.45, epsilon = 1e-10);
    }
}
