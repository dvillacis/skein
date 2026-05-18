use super::GroupPenalty;
use crate::groups::Groups;
use crate::prox::scad_prox;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Sparse-group SCAD penalty.
///
/// `λ · (α · Σ_k v_{g,k} · SCAD(|β_{g,k}|; a) + (1−α) · w_g · SCAD(‖β_g‖_2; a))`
///
/// Native direct-CD analogue of the LLA-wrapped weighted-sparse-group-lasso
/// surrogate (`solver::lla::surrogate_sparse_group_scad`). Mirrors
/// `SparseGroupMcp` exactly except the per-coord and per-group MCP
/// formulas are replaced with their SCAD three-region equivalents.
///
/// The per-group prox is the same two-step composition Breheny & Huang
/// 2015 derive for the MCP case: apply the closed-form scalar SCAD prox
/// to each coordinate of the block with threshold `α · λ · v_{g,k}`,
/// then apply the closed-form group-SCAD prox to the result with
/// threshold `(1−α) · λ · w_g`. The composition is exact because the
/// per-coord penalty is sign-invariant + coordinatewise separable and
/// the L2 penalty depends only on the resulting block norm.
pub struct SparseGroupScad {
    lambda: f64,
    alpha: f64,
    a: f64,
    weights: Array1<f64>,
    /// Per-group, per-position-in-group L1 weights. `None` ⇒ uniform 1.
    coord_weights: Option<Vec<Array1<f64>>>,
}

impl SparseGroupScad {
    pub fn new(lambda: f64, alpha: f64, a: f64, n_groups: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert!(a > 2.0, "a must be > 2 for SCAD (got {})", a);
        Self {
            lambda,
            alpha,
            a,
            weights: Array1::ones(n_groups),
            coord_weights: None,
        }
    }

    pub fn with_weights(lambda: f64, alpha: f64, a: f64, weights: Array1<f64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert!(a > 2.0, "a must be > 2 for SCAD (got {})", a);
        Self {
            lambda,
            alpha,
            a,
            weights,
            coord_weights: None,
        }
    }

    /// Two-level construction: per-group L2 weights plus per-group,
    /// per-position-in-group L1 weights. Mirrors
    /// `SparseGroupMcp::with_coord_weights`.
    pub fn with_coord_weights(
        lambda: f64,
        alpha: f64,
        a: f64,
        group_weights: Array1<f64>,
        coord_weights: Vec<Array1<f64>>,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert!(a > 2.0, "a must be > 2 for SCAD (got {})", a);
        assert_eq!(
            group_weights.len(),
            coord_weights.len(),
            "coord_weights must have one entry per group"
        );
        Self {
            lambda,
            alpha,
            a,
            weights: group_weights,
            coord_weights: Some(coord_weights),
        }
    }
}

/// Scalar SCAD penalty value `λ · w · SCAD(|z|; a)` for one coordinate.
#[inline]
fn scad_scalar_value(z: f64, lambda: f64, a: f64, weight: f64) -> f64 {
    let lam = lambda * weight;
    let abs_z = z.abs();
    if abs_z <= lam {
        lam * abs_z
    } else if abs_z <= a * lam {
        let num = abs_z * abs_z - 2.0 * a * lam * abs_z + lam * lam;
        lam * abs_z - num / (2.0 * (a - 1.0))
    } else {
        (a + 1.0) * lam * lam / 2.0
    }
}

impl GroupPenalty for SparseGroupScad {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64 {
        let mut total = 0.0;
        for g in 0..groups.n_groups() {
            let cols = groups.group(g);
            let coord_w = self.coord_weights.as_ref().map(|cw| &cw[g]);
            // Per-coord SCAD.
            let mut coord_term = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for (k, &j) in cols.iter().enumerate() {
                let b = beta[j];
                let v_k = coord_w.map(|cw| cw[k]).unwrap_or(1.0);
                coord_term += scad_scalar_value(b, self.lambda * self.alpha, self.a, v_k);
                sum_sq += b * b;
            }
            // Per-group L2-SCAD.
            let block_l2 = sum_sq.sqrt();
            let group_term = scad_scalar_value(
                block_l2,
                self.lambda * (1.0 - self.alpha),
                self.a,
                self.weights[g],
            );
            total += coord_term + group_term;
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let coord_w = self.coord_weights.as_ref().map(|cw| &cw[g]);

        // Step 1: per-coordinate scalar SCAD prox.
        for (k, x) in block.iter_mut().enumerate() {
            let v_k = coord_w.map(|cw| cw[k]).unwrap_or(1.0);
            *x = scad_prox(*x, step, self.lambda * self.alpha, self.a, v_k);
        }

        // Step 2: group-SCAD block prox. Mirrors `GroupScad::prox_group`
        // applied to the L2 norm of the post-step-1 block.
        let slice = block.as_slice_mut().expect("contiguous block expected");
        let norm: f64 = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
        let lam_g = self.lambda * (1.0 - self.alpha) * self.weights[g];

        // Flat region.
        if norm > self.a * lam_g {
            return;
        }

        // Lasso region.
        if norm <= (1.0 + step) * lam_g {
            let thr = step * lam_g;
            if norm <= thr {
                for x in slice.iter_mut() {
                    *x = 0.0;
                }
            } else {
                let scale = 1.0 - thr / norm;
                for x in slice.iter_mut() {
                    *x *= scale;
                }
            }
            return;
        }

        // Middle region: SCAD shrinkage with degenerate-regime clamp on `a`.
        let a_eff = self.a.max(step + 1.0 + 1e-9);
        let denom = 1.0 - step / (a_eff - 1.0);
        let num = norm - step * a_eff * lam_g / (a_eff - 1.0);
        let new_norm = num / denom;
        let scale = new_norm / norm;
        for x in slice.iter_mut() {
            *x *= scale;
        }
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    fn min_step_for_unimodal(&self) -> f64 {
        self.a - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::{DenseMatrix, DesignMatrix};
    use crate::penalty::{GroupScad, SparseGroupLasso};
    use crate::solver::{block_cd_solve, CdConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    const A: f64 = 3.7;

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
    fn value_zero_when_blocks_are_zero() {
        let beta = Array1::<f64>::zeros(4);
        let groups = Groups::contiguous_blocks(4, 2);
        let pen = SparseGroupScad::new(0.1, 0.5, A, 2);
        assert_abs_diff_eq!(pen.value(beta.view(), &groups), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn value_saturates_in_flat_region() {
        // Large β well above a·λ: per-coord SCAD = (a+1)·λ²/2 per coord;
        // group SCAD also saturates.
        let beta = array![10.0, -10.0];
        let groups = Groups::contiguous_blocks(2, 2);
        let pen = SparseGroupScad::new(0.1, 0.5, A, 1);
        let v = pen.value(beta.view(), &groups);
        let lam_coord = 0.1 * 0.5;
        let lam_group = 0.1 * 0.5;
        let expected =
            2.0 * (A + 1.0) * lam_coord * lam_coord / 2.0 + (A + 1.0) * lam_group * lam_group / 2.0;
        assert_abs_diff_eq!(v, expected, epsilon = 1e-10);
    }

    // ---- prox -------------------------------------------------------

    #[test]
    fn prox_zeros_block_under_strong_penalty() {
        let pen = SparseGroupScad::new(10.0, 0.5, A, 1);
        let mut block = array![0.3, -0.4];
        pen.prox_group(0, block.view_mut(), 1.0);
        for &v in block.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn prox_identity_well_above_kink() {
        // Block large enough to saturate BOTH per-coord SCAD (each |β_k| >
        // a·λ·α·v_k) AND group SCAD (‖β‖_2 > a·λ·(1-α)·w_g). Returns input.
        let pen = SparseGroupScad::new(0.1, 0.5, A, 1);
        let mut block = array![5.0, -4.0];
        let before = block.clone();
        pen.prox_group(0, block.view_mut(), 1.0);
        for (a, b) in block.iter().zip(before.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
        }
    }

    // ---- reductions to known special cases --------------------------

    #[test]
    fn with_large_a_matches_sparse_group_lasso() {
        // As a → ∞, SCAD → L1, so SparseGroupSCAD → SparseGroupLasso at
        // the same α. Use a = 1e6 to approximate.
        let (design, y, groups) = problem(11);
        let datafit = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let lambda = 0.05;
        let alpha = 0.5;

        let pen_scad = SparseGroupScad::new(lambda, alpha, 1e6, groups.n_groups());
        let pen_sgl = SparseGroupLasso::new(lambda, alpha, groups.n_groups());

        let (b_scad, _) = block_cd_solve(&design, &datafit, &pen_scad, &groups, &cfg);
        let (b_sgl, _) = block_cd_solve(&design, &datafit, &pen_sgl, &groups, &cfg);

        for j in 0..design.n_features() {
            assert_abs_diff_eq!(b_scad[j], b_sgl[j], epsilon = 1e-5);
        }
    }

    #[test]
    fn with_alpha_zero_matches_group_scad() {
        // α = 0 ⇒ only the per-group L2-SCAD term remains; should match
        // GroupScad at the same λ, a.
        let (design, y, groups) = problem(12);
        let datafit = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let lambda = 0.05;

        let pen_sgs = SparseGroupScad::new(lambda, 0.0, A, groups.n_groups());
        let pen_gs = GroupScad::new(lambda, A, groups.n_groups());

        let (b_sgs, _) = block_cd_solve(&design, &datafit, &pen_sgs, &groups, &cfg);
        let (b_gs, _) = block_cd_solve(&design, &datafit, &pen_gs, &groups, &cfg);

        for j in 0..design.n_features() {
            assert_abs_diff_eq!(b_sgs[j], b_gs[j], epsilon = 1e-6);
        }
    }

    // ---- within-group sparsity recovery -----------------------------

    #[test]
    fn recovers_within_group_sparsity() {
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

        let pen = SparseGroupScad::new(0.05, 0.5, A, groups.n_groups());
        let (b, _) = block_cd_solve(&design, &datafit, &pen, &groups, &cfg);

        assert!(b[0].abs() > 1.5, "feature 0 too small: {}", b[0]);
        for j in 1..p {
            assert!(b[j].abs() < 0.3, "feature {} too large: {}", j, b[j]);
        }
    }

    // ---- two-level construction (per-coord + per-group weights) -----

    #[test]
    fn coord_weights_zeros_specific_coord() {
        // Coord weight 100 on β_0 amplifies its per-coord SCAD threshold
        // enough to zero it. α = 1 so step 2 (group SCAD) is identity.
        let group_w = array![1.0];
        let coord_w = vec![array![100.0, 1.0]];
        let pen = SparseGroupScad::with_coord_weights(0.2, 1.0, A, group_w, coord_w);
        let mut block = array![0.5, 0.5];
        pen.prox_group(0, block.view_mut(), 1.0);
        // β_0, weight 100: λ_eff = 20. |0.5| ≤ λ_eff ⇒ lasso branch,
        // soft-threshold by step·λ_eff = 20 ⇒ 0.
        assert_abs_diff_eq!(block[0], 0.0, epsilon = 1e-12);
        // β_1, weight 1: λ_eff = 0.2. |0.5| in (λ_eff, (1+step)·λ_eff) = (0.2, 0.4)?
        // 0.5 > 0.4 = (1+step)·λ_eff, but ≤ a·λ_eff = 0.74 ⇒ middle branch.
        // denom = 1 - 1/(A-1) ≈ 0.6296
        // num = 0.5 - 1·A·0.2/(A-1) ≈ 0.5 - 0.2741 = 0.2259
        // β_1' = 0.2259 / 0.6296 ≈ 0.3588
        let denom = 1.0 - 1.0 / (A - 1.0);
        let num = 0.5 - A * 0.2 / (A - 1.0);
        let expected = num / denom;
        assert_abs_diff_eq!(block[1], expected, epsilon = 1e-10);
    }
}
