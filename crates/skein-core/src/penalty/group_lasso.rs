use super::GroupPenalty;
use crate::groups::Groups;
use crate::prox::group_soft_threshold;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

pub struct GroupLasso {
    lambda: f64,
    weights: Array1<f64>,
}

impl GroupLasso {
    pub fn new(lambda: f64, n_groups: usize) -> Self {
        Self {
            lambda,
            weights: Array1::ones(n_groups),
        }
    }

    pub fn with_weights(lambda: f64, weights: Array1<f64>) -> Self {
        Self { lambda, weights }
    }
}

impl GroupPenalty for GroupLasso {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64 {
        let mut total = 0.0;
        for g in 0..groups.n_groups() {
            let block_norm: f64 = groups
                .group(g)
                .iter()
                .map(|&j| beta[j] * beta[j])
                .sum::<f64>()
                .sqrt();
            total += self.lambda * self.weights[g] * block_norm;
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let slice = block.as_slice_mut().expect("contiguous block expected");
        group_soft_threshold(slice, step, self.lambda, self.weights[g]);
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    fn two_blocks_of_two() -> Groups {
        // Two contiguous groups of size 2: features 0..2 and 2..4.
        Groups::contiguous_blocks(4, 2)
    }

    #[test]
    fn value_zero_when_blocks_are_zero() {
        let pen = GroupLasso::new(0.5, 2);
        let beta = Array1::<f64>::zeros(4);
        assert_abs_diff_eq!(pen.value(beta.view(), &two_blocks_of_two()), 0.0);
    }

    #[test]
    fn value_is_lambda_times_block_norm_sum() {
        // β = [3, 4, 0, 1] → ‖block_0‖ = 5, ‖block_1‖ = 1.
        // λ=0.5, weights=[1, 2] ⇒ 0.5·1·5 + 0.5·2·1 = 2.5 + 1.0 = 3.5
        let pen = GroupLasso::with_weights(0.5, array![1.0, 2.0]);
        let beta = array![3.0, 4.0, 0.0, 1.0];
        assert_abs_diff_eq!(
            pen.value(beta.view(), &two_blocks_of_two()),
            3.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn prox_group_zeroes_block_when_norm_below_threshold() {
        // step·λ·w = 1·1·1 = 1; ‖[0.3, 0.4]‖ = 0.5 < 1 ⇒ zero out.
        let pen = GroupLasso::new(1.0, 2);
        let mut beta = array![0.3_f64, 0.4, 99.0, 99.0];
        let block = beta.slice_mut(ndarray::s![0..2]);
        pen.prox_group(0, block, 1.0);
        assert_abs_diff_eq!(beta[0], 0.0);
        assert_abs_diff_eq!(beta[1], 0.0);
    }

    #[test]
    fn prox_group_shrinks_block_when_norm_above_threshold() {
        // step·λ·w = 1; ‖[3, 4]‖ = 5; scale = 1 − 1/5 = 0.8.
        let pen = GroupLasso::new(1.0, 2);
        let mut beta = array![3.0_f64, 4.0, 0.0, 0.0];
        let block = beta.slice_mut(ndarray::s![0..2]);
        pen.prox_group(0, block, 1.0);
        assert_abs_diff_eq!(beta[0], 2.4, epsilon = 1e-12);
        assert_abs_diff_eq!(beta[1], 3.2, epsilon = 1e-12);
    }

    #[test]
    fn prox_group_indexes_weights_by_g() {
        // Group 0: weight 0 ⇒ no shrinkage.
        // Group 1: weight 1 ⇒ standard threshold.
        let pen = GroupLasso::with_weights(1.0, array![0.0, 1.0]);
        let mut beta = array![3.0_f64, 4.0, 0.3, 0.4];
        // Apply group 0 prox to first block: should leave it untouched.
        let b0 = beta.slice_mut(ndarray::s![0..2]);
        pen.prox_group(0, b0, 1.0);
        assert_abs_diff_eq!(beta[0], 3.0);
        assert_abs_diff_eq!(beta[1], 4.0);
        // Apply group 1 prox to second block: ‖0.3, 0.4‖ = 0.5 < 1 ⇒ zero.
        let b1 = beta.slice_mut(ndarray::s![2..4]);
        pen.prox_group(1, b1, 1.0);
        assert_abs_diff_eq!(beta[2], 0.0);
        assert_abs_diff_eq!(beta[3], 0.0);
    }

    #[test]
    fn weights_view_returns_user_supplied() {
        let pen = GroupLasso::with_weights(0.5, array![0.5, 2.0]);
        let w = pen.weights();
        assert_eq!(w.len(), 2);
        assert_abs_diff_eq!(w[0], 0.5);
        assert_abs_diff_eq!(w[1], 2.0);
    }

    #[test]
    fn default_weights_are_ones() {
        let pen = GroupLasso::new(0.5, 3);
        for v in pen.weights().iter() {
            assert_abs_diff_eq!(*v, 1.0);
        }
    }
}
