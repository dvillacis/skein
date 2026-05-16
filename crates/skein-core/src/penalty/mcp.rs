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
        Self {
            lambda,
            gamma,
            weights,
        }
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

    fn min_step_for_unimodal(&self) -> f64 {
        // MCP prox is unimodal iff step < γ (the firm-threshold formula's
        // convex regime). Beyond γ the prox is multimodal.
        self.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox::mcp_prox;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn value_zero_at_origin() {
        let pen = Mcp::new(0.5, 3.0, 4);
        assert_abs_diff_eq!(pen.value(array![0.0, 0.0, 0.0, 0.0].view()), 0.0);
    }

    #[test]
    fn value_in_quadratic_regime_matches_formula() {
        // |β| ≤ γλ ⇒ value_j = λ·w·|β| − β²/(2γ).
        let pen = Mcp::with_weights(0.5, 3.0, array![1.0, 2.0]);
        let beta = array![0.4, -0.3];
        // j=0: 0.5·|0.4| − 0.16/6 = 0.2 − 0.0266667 = 0.173333…
        // j=1: λ·w = 1.0; |β|=0.3; γλ_eff = 3·1 = 3 ≥ 0.3 ⇒ in regime.
        //      1.0·0.3 − 0.09/6 = 0.3 − 0.015 = 0.285
        let expected = (0.5 * 0.4 - 0.16 / 6.0) + (1.0 * 0.3 - 0.09 / 6.0);
        assert_abs_diff_eq!(pen.value(beta.view()), expected, epsilon = 1e-12);
    }

    #[test]
    fn value_in_flat_regime_caps_at_gamma_lambda_squared_over_two() {
        // |β| > γλ ⇒ value_j = γ·λ_eff²/2, regardless of β.
        let pen = Mcp::with_weights(0.5, 3.0, array![1.0, 2.0]);
        // j=0: γλ = 1.5; β = 5 > 1.5 ⇒ flat at 3·0.25/2 = 0.375
        // j=1: γλ_eff = 3·1 = 3; β = 5 > 3 ⇒ flat at 3·1/2 = 1.5
        let beta = array![5.0, 5.0];
        assert_abs_diff_eq!(pen.value(beta.view()), 0.375 + 1.5, epsilon = 1e-12);
        // Different |β| (still both flat) yields the same total — the
        // hallmark of MCP's flat tail.
        let beta2 = array![10.0, -7.0];
        assert_abs_diff_eq!(pen.value(beta2.view()), 0.375 + 1.5, epsilon = 1e-12);
    }

    #[test]
    fn prox_coord_delegates_with_correct_weight() {
        let weights = array![0.5, 1.0, 2.0];
        let pen = Mcp::with_weights(0.4, 2.5, weights.clone());
        for j in 0..3 {
            for &z in &[-1.5_f64, -0.2, 0.2, 1.5] {
                for &step in &[0.5_f64, 1.0] {
                    assert_abs_diff_eq!(
                        pen.prox_coord(j, z, step),
                        mcp_prox(z, step, 0.4, 2.5, weights[j]),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn prox_coord_indexes_weights_by_j() {
        // Different j ⇒ different effective weight ⇒ different output.
        // Catches a "j vs constant" indexing bug.
        let pen = Mcp::with_weights(1.0, 3.0, array![0.0, 1.0]);
        // j=0: weight 0 ⇒ no penalty ⇒ identity prox.
        assert_abs_diff_eq!(pen.prox_coord(0, 0.5, 1.0), 0.5, epsilon = 1e-12);
        // j=1: weight 1, threshold = 1.0 ⇒ z=0.5 → 0.
        assert_abs_diff_eq!(pen.prox_coord(1, 0.5, 1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn weights_view_returns_user_supplied() {
        let pen = Mcp::with_weights(0.5, 3.0, array![0.25, 1.0, 4.0]);
        let w = pen.weights();
        assert_eq!(w.len(), 3);
        assert_abs_diff_eq!(w[0], 0.25);
        assert_abs_diff_eq!(w[1], 1.0);
        assert_abs_diff_eq!(w[2], 4.0);
    }

    #[test]
    fn default_weights_are_ones() {
        let pen = Mcp::new(0.5, 3.0, 5);
        let w = pen.weights();
        for v in w.iter() {
            assert_abs_diff_eq!(*v, 1.0);
        }
    }
}
