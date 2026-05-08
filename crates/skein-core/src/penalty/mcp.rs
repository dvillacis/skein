use super::Penalty;
use crate::prox::mcp_prox;
use ndarray::{Array1, ArrayView1};

pub struct Mcp {
    lambda: f64,
    gamma: f64,
    weights: Array1<f64>,
}

impl Mcp {
    pub fn new(lambda: f64, gamma: f64, n_features: usize) -> Self {
        Self {
            lambda,
            gamma,
            weights: Array1::ones(n_features),
        }
    }

    pub fn with_weights(lambda: f64, gamma: f64, weights: Array1<f64>) -> Self {
        Self { lambda, gamma, weights }
    }
}

impl Penalty for Mcp {
    fn value(&self, beta: ArrayView1<f64>) -> f64 {
        let mut total = 0.0;
        for (j, &b) in beta.iter().enumerate() {
            let lam = self.lambda * self.weights[j];
            let abs_b = b.abs();
            total += if abs_b <= self.gamma * lam {
                lam * abs_b - b * b / (2.0 * self.gamma)
            } else {
                self.gamma * lam * lam / 2.0
            };
        }
        total
    }

    fn prox_coord(&self, j: usize, z: f64, step: f64) -> f64 {
        mcp_prox(z, step, self.lambda, self.gamma, self.weights[j])
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }
}
