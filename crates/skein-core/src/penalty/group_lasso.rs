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
