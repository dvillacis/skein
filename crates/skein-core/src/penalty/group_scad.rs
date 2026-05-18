use super::GroupPenalty;
use crate::groups::Groups;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Group SCAD: `SCAD(‖β_g‖₂; λ, a)` per group. Reduces to scalar SCAD when
/// every group is a singleton. Native direct-CD analogue of the LLA-wrapped
/// weighted-group-lasso surrogate previously used for group SCAD; mirrors
/// `GroupMcp` exactly except for the three-region SCAD shrinkage formula.
pub struct GroupScad {
    lambda: f64,
    a: f64,
    weights: Array1<f64>,
}

impl GroupScad {
    pub fn new(lambda: f64, a: f64, n_groups: usize) -> Self {
        assert!(a > 2.0, "a must be > 2 for SCAD (got {})", a);
        Self {
            lambda,
            a,
            weights: Array1::ones(n_groups),
        }
    }

    pub fn with_weights(lambda: f64, a: f64, weights: Array1<f64>) -> Self {
        assert!(a > 2.0, "a must be > 2 for SCAD (got {})", a);
        Self { lambda, a, weights }
    }
}

impl GroupPenalty for GroupScad {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64 {
        let mut total = 0.0;
        for g in 0..groups.n_groups() {
            let norm: f64 = groups
                .group(g)
                .iter()
                .map(|&j| beta[j] * beta[j])
                .sum::<f64>()
                .sqrt();
            let lam = self.lambda * self.weights[g];
            total += if norm <= lam {
                lam * norm
            } else if norm <= self.a * lam {
                let num = norm * norm - 2.0 * self.a * lam * norm + lam * lam;
                lam * norm - num / (2.0 * (self.a - 1.0))
            } else {
                (self.a + 1.0) * lam * lam / 2.0
            };
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let slice = block.as_slice_mut().expect("contiguous block expected");
        let norm: f64 = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
        let lam = self.lambda * self.weights[g];

        // Flat region: ‖block‖ > a·λ ⇒ identity.
        if norm > self.a * lam {
            return;
        }

        // Lasso region: ‖block‖ ≤ (1+step)·λ ⇒ block soft-threshold.
        if norm <= (1.0 + step) * lam {
            let thr = step * lam;
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

        // Middle region: (1+step)·λ < ‖block‖ ≤ a·λ ⇒ SCAD shrinkage.
        // Clamp `a` upward in the degenerate regime (step ≥ a-1) so the
        // divisor stays in (0, 1). Mirrors `scad_prox` in prox.rs.
        let a_eff = self.a.max(step + 1.0 + 1e-9);
        let denom = 1.0 - step / (a_eff - 1.0);
        let num = norm - step * a_eff * lam / (a_eff - 1.0);
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
    use crate::penalty::{Penalty, Scad};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    const A: f64 = 3.7;

    fn two_blocks_of_two() -> Groups {
        Groups::contiguous_blocks(4, 2)
    }

    #[test]
    fn value_zero_when_blocks_are_zero() {
        let pen = GroupScad::new(0.5, A, 2);
        let beta = Array1::<f64>::zeros(4);
        assert_abs_diff_eq!(pen.value(beta.view(), &two_blocks_of_two()), 0.0);
    }

    #[test]
    fn value_matches_scalar_scad_on_block_norm() {
        // Group SCAD on a single group of size 2 with ‖β_g‖ = r must equal
        // scalar SCAD at β = r (same λ_eff, same a). Mirrors GroupMcp test.
        let lambda = 0.5;
        let weights = array![1.0, 2.0];
        let pen = GroupScad::with_weights(lambda, A, weights.clone());
        let beta = array![3.0_f64, 4.0, 0.3, 0.4]; // ‖[3,4]‖=5, ‖[0.3,0.4]‖=0.5
        let actual = pen.value(beta.view(), &two_blocks_of_two());

        let r0 = 5.0;
        let r1 = 0.5;
        let scalar0 = Scad::with_weights(lambda, A, array![weights[0]]);
        let scalar1 = Scad::with_weights(lambda, A, array![weights[1]]);
        let expected = scalar0.value(array![r0].view()) + scalar1.value(array![r1].view());
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    #[test]
    fn prox_group_identity_above_a_lambda() {
        // ‖block‖ > a·λ ⇒ identity.
        let pen = GroupScad::new(0.5, A, 1); // a·λ = 1.85
        let mut block = array![10.0_f64, 0.0]; // norm 10 > 1.85
        pen.prox_group(0, block.view_mut(), 1.0);
        assert_abs_diff_eq!(block[0], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(block[1], 0.0);
    }

    #[test]
    fn prox_group_zeroes_block_below_l1_threshold() {
        // step=1, λ_eff=1 ⇒ thr=1; ‖[0.3,0.4]‖ = 0.5 < 1 ⇒ zero.
        let pen = GroupScad::new(1.0, A, 1);
        let mut block = array![0.3_f64, 0.4];
        pen.prox_group(0, block.view_mut(), 1.0);
        assert_abs_diff_eq!(block[0], 0.0);
        assert_abs_diff_eq!(block[1], 0.0);
    }

    #[test]
    fn prox_group_lasso_region_matches_block_soft_threshold() {
        // ‖block‖ ≤ (1+step)·λ ⇒ block soft-threshold (no SCAD bump yet).
        // step=1, λ=0.5 ⇒ lasso boundary = 1.0; ‖[0.6,0.8]‖ = 1.0 (edge).
        let pen = GroupScad::new(0.5, A, 1);
        let mut block = array![0.6_f64, 0.8];
        pen.prox_group(0, block.view_mut(), 1.0);
        // Block soft-threshold: scale = 1 - 0.5/1.0 = 0.5.
        assert_abs_diff_eq!(block[0], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(block[1], 0.4, epsilon = 1e-12);
    }

    #[test]
    fn prox_group_middle_region_matches_scalar_scad_on_norm() {
        // (1+step)·λ < ‖block‖ ≤ a·λ ⇒ scalar SCAD shrinkage of the norm,
        // applied to the block along the same ray.
        // step=1, λ=0.5, a=3.7 ⇒ middle region: 1.0 < ‖block‖ ≤ 1.85.
        // Pick ‖block‖ = 1.2.
        let pen = GroupScad::new(0.5, A, 1);
        let norm_in = 1.2_f64;
        // Block [0.72, 0.96] has norm 1.2.
        let mut block = array![0.72_f64, 0.96];
        pen.prox_group(0, block.view_mut(), 1.0);

        // Scalar SCAD prox at z = 1.2 with same params:
        // denom = 1 - 1/(3.7-1) = 1 - 1/2.7 ≈ 0.62963
        // num = 1.2 - 1·3.7·0.5/2.7 = 1.2 - 0.68519 ≈ 0.51481
        // new_norm = 0.51481 / 0.62963 ≈ 0.81765
        let denom = 1.0 - 1.0 / (A - 1.0);
        let num = norm_in - 1.0 * A * 0.5 / (A - 1.0);
        let new_norm = num / denom;
        let scale = new_norm / norm_in;
        assert_abs_diff_eq!(block[0], 0.72 * scale, epsilon = 1e-10);
        assert_abs_diff_eq!(block[1], 0.96 * scale, epsilon = 1e-10);
    }

    #[test]
    fn prox_group_indexes_weights_by_g() {
        // Group 0: weight 0 ⇒ λ_eff = 0 ⇒ identity.
        // Group 1: weight 1 ⇒ standard SCAD behavior.
        let pen = GroupScad::with_weights(1.0, A, array![0.0, 1.0]);
        let mut beta = array![3.0_f64, 4.0, 0.3, 0.4];
        let b0 = beta.slice_mut(ndarray::s![0..2]);
        pen.prox_group(0, b0, 1.0);
        assert_abs_diff_eq!(beta[0], 3.0);
        assert_abs_diff_eq!(beta[1], 4.0);
        let b1 = beta.slice_mut(ndarray::s![2..4]);
        pen.prox_group(1, b1, 1.0);
        assert_abs_diff_eq!(beta[2], 0.0);
        assert_abs_diff_eq!(beta[3], 0.0);
    }

    #[test]
    fn weights_view_returns_user_supplied() {
        let pen = GroupScad::with_weights(0.5, A, array![0.5, 2.0]);
        let w = pen.weights();
        assert_eq!(w.len(), 2);
        assert_abs_diff_eq!(w[0], 0.5);
        assert_abs_diff_eq!(w[1], 2.0);
    }

    #[test]
    fn default_weights_are_ones() {
        let pen = GroupScad::new(0.5, A, 3);
        for v in pen.weights().iter() {
            assert_abs_diff_eq!(*v, 1.0);
        }
    }
}
