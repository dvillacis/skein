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
