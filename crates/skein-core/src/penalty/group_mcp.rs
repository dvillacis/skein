use super::GroupPenalty;
use crate::groups::Groups;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Group MCP: `MCP(‖β_g‖₂; λ, γ)` per group. Reduces to scalar MCP when
/// every group is a singleton.
pub struct GroupMcp {
    lambda: f64,
    gamma: f64,
    weights: Array1<f64>,
}

impl GroupMcp {
    pub fn new(lambda: f64, gamma: f64, n_groups: usize) -> Self {
        Self {
            lambda,
            gamma,
            weights: Array1::ones(n_groups),
        }
    }

    pub fn with_weights(lambda: f64, gamma: f64, weights: Array1<f64>) -> Self {
        Self {
            lambda,
            gamma,
            weights,
        }
    }
}

impl GroupPenalty for GroupMcp {
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
            total += if norm <= self.gamma * lam {
                lam * norm - norm * norm / (2.0 * self.gamma)
            } else {
                self.gamma * lam * lam / 2.0
            };
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let slice = block.as_slice_mut().expect("contiguous block expected");
        let norm: f64 = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
        let lam = self.lambda * self.weights[g];

        if norm >= self.gamma * lam {
            return;
        }
        if self.gamma > step {
            let thr = step * lam;
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
            // Degenerate γ ≤ step: hard-threshold convention.
            let cutoff = (step * self.gamma * lam * lam).sqrt();
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
    use crate::penalty::Mcp;
    use crate::penalty::Penalty;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    fn two_blocks_of_two() -> Groups {
        Groups::contiguous_blocks(4, 2)
    }

    #[test]
    fn value_zero_when_blocks_are_zero() {
        let pen = GroupMcp::new(0.5, 3.0, 2);
        let beta = Array1::<f64>::zeros(4);
        assert_abs_diff_eq!(pen.value(beta.view(), &two_blocks_of_two()), 0.0);
    }

    #[test]
    fn value_in_quadratic_regime_matches_scalar_mcp_on_block_norm() {
        // Group MCP value on a single group of size 2 with ‖β_g‖ = r
        // must equal scalar MCP at β = r (same λ_eff, same γ).
        let lambda = 0.5;
        let gamma = 3.0;
        let weights = array![1.0, 2.0];
        let pen = GroupMcp::with_weights(lambda, gamma, weights.clone());
        // ‖[3, 4]‖ = 5, ‖[0.3, 0.4]‖ = 0.5
        let beta = array![3.0_f64, 4.0, 0.3, 0.4];
        let actual = pen.value(beta.view(), &two_blocks_of_two());

        // Reference via scalar MCP applied to the block norms.
        let r0 = 5.0;
        let r1 = 0.5;
        let scalar0 = Mcp::with_weights(lambda, gamma, array![weights[0]]);
        let scalar1 = Mcp::with_weights(lambda, gamma, array![weights[1]]);
        let expected = scalar0.value(array![r0].view()) + scalar1.value(array![r1].view());
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    #[test]
    fn prox_group_returns_input_above_kink() {
        // ‖block‖ ≥ γ·λ_eff ⇒ identity (no shrinkage).
        let pen = GroupMcp::new(0.5, 3.0, 1); // γλ = 1.5
        let mut beta = array![10.0_f64, 0.0]; // norm = 10 > 1.5
        let block = beta.view_mut();
        pen.prox_group(0, block, 1.0);
        assert_abs_diff_eq!(beta[0], 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(beta[1], 0.0);
    }

    #[test]
    fn prox_group_zeroes_block_below_l1_threshold() {
        // step=1, λ_eff=1 ⇒ threshold step·λ_eff = 1.
        // ‖[0.3, 0.4]‖ = 0.5 < 1 and γ=3 > step=1 ⇒ zero out.
        let pen = GroupMcp::new(1.0, 3.0, 1);
        let mut beta = array![0.3_f64, 0.4];
        pen.prox_group(0, beta.view_mut(), 1.0);
        assert_abs_diff_eq!(beta[0], 0.0);
        assert_abs_diff_eq!(beta[1], 0.0);
    }

    #[test]
    fn prox_group_intermediate_regime_matches_scalar_mcp() {
        // For a single group with ‖block‖ = r in (step·λ_eff, γ·λ_eff),
        // the prox scales the block by the scalar MCP shrinkage factor
        // applied to r. Here: step=1, λ=0.5, γ=3 ⇒ thr=0.5, kink=1.5.
        // Pick ‖block‖ = 1.0.
        let pen = GroupMcp::new(0.5, 3.0, 1);
        let mut block = array![0.6_f64, 0.8]; // norm = 1.0
        let norm_in = 1.0_f64;
        pen.prox_group(0, block.view_mut(), 1.0);

        // Scalar MCP prox at z=1.0 with same params:
        // s = (1 − 0.5).max(0) = 0.5; scale = sign·s/(1 − step/γ) = 0.5 / (2/3) = 0.75.
        // So new norm should be 0.75; per-entry shrink = 0.75 / 1.0 = 0.75.
        assert_abs_diff_eq!(block[0], 0.6 * 0.75, epsilon = 1e-12);
        assert_abs_diff_eq!(block[1], 0.8 * 0.75, epsilon = 1e-12);
        // Sanity: norm collapsed appropriately.
        let norm_out = (block[0] * block[0] + block[1] * block[1]).sqrt();
        assert_abs_diff_eq!(norm_out, 0.75 * norm_in, epsilon = 1e-12);
    }

    #[test]
    fn prox_group_indexes_weights_by_g() {
        // Group 0: weight 0 ⇒ λ_eff = 0 ⇒ identity for all input.
        // Group 1: weight 1 ⇒ standard MCP behavior.
        let pen = GroupMcp::with_weights(1.0, 3.0, array![0.0, 1.0]);
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
        let pen = GroupMcp::with_weights(0.5, 3.0, array![0.5, 2.0]);
        let w = pen.weights();
        assert_eq!(w.len(), 2);
        assert_abs_diff_eq!(w[0], 0.5);
        assert_abs_diff_eq!(w[1], 2.0);
    }

    #[test]
    fn default_weights_are_ones() {
        let pen = GroupMcp::new(0.5, 3.0, 3);
        for v in pen.weights().iter() {
            assert_abs_diff_eq!(*v, 1.0);
        }
    }
}
