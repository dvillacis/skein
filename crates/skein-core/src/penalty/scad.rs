use super::Penalty;
use crate::prox::scad_prox;
use ndarray::{Array1, ArrayView1};

pub struct Scad {
    lambda: f64,
    a: f64,
    weights: Array1<f64>,
}

impl Scad {
    pub fn new(lambda: f64, a: f64, n_features: usize) -> Self {
        Self {
            lambda,
            a,
            weights: Array1::ones(n_features),
        }
    }

    pub fn with_weights(lambda: f64, a: f64, weights: Array1<f64>) -> Self {
        Self { lambda, a, weights }
    }
}

impl Penalty for Scad {
    fn value(&self, beta: ArrayView1<f64>) -> f64 {
        let mut total = 0.0;
        for (j, &b) in beta.iter().enumerate() {
            let lam = self.lambda * self.weights[j];
            let abs_b = b.abs();
            total += if abs_b <= lam {
                lam * abs_b
            } else if abs_b <= self.a * lam {
                let num = abs_b * abs_b - 2.0 * self.a * lam * abs_b + lam * lam;
                lam * abs_b - num / (2.0 * (self.a - 1.0))
            } else {
                (self.a + 1.0) * lam * lam / 2.0
            };
        }
        total
    }

    fn prox_coord(&self, j: usize, z: f64, step: f64) -> f64 {
        scad_prox(z, step, self.lambda, self.a, self.weights[j])
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }
}
