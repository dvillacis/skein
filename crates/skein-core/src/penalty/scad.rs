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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox::scad_prox;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    const A: f64 = 3.7;

    #[test]
    fn value_zero_at_origin() {
        let pen = Scad::new(0.5, A, 3);
        assert_abs_diff_eq!(pen.value(array![0.0, 0.0, 0.0].view()), 0.0);
    }

    #[test]
    fn value_in_lasso_regime_is_lambda_abs_beta() {
        // |β| ≤ λ_eff ⇒ value_j = λ_eff · |β|.
        let pen = Scad::with_weights(0.5, A, array![1.0, 2.0]);
        // j=0: λ_eff=0.5, |β|=0.4 ⇒ 0.20
        // j=1: λ_eff=1.0, |β|=0.6 ⇒ 0.60
        let beta = array![0.4, -0.6];
        assert_abs_diff_eq!(pen.value(beta.view()), 0.80, epsilon = 1e-12);
    }

    #[test]
    fn value_in_flat_regime_caps_at_a_plus_one_lambda_squared_over_two() {
        // |β| > a·λ_eff ⇒ constant cap (a+1)·λ_eff² / 2.
        let pen = Scad::with_weights(0.5, A, array![1.0]);
        // a·λ_eff = 1.85; β = 5 > 1.85 ⇒ flat at (4.7)·0.25/2 = 0.5875
        let beta = array![5.0];
        assert_abs_diff_eq!(
            pen.value(beta.view()),
            (A + 1.0) * 0.25 / 2.0,
            epsilon = 1e-12
        );
        // Different |β| (still flat) gives same total.
        let beta2 = array![-10.0];
        assert_abs_diff_eq!(
            pen.value(beta2.view()),
            (A + 1.0) * 0.25 / 2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn value_in_quadratic_regime_matches_formula() {
        // λ < |β| ≤ a·λ_eff ⇒ value_j = λ_eff·|β| − (β² − 2aλ_eff|β| + λ_eff²) / (2(a−1)).
        let pen = Scad::with_weights(0.5, A, array![1.0]);
        let abs_b = 1.0; // > λ_eff=0.5 and ≤ a·λ_eff = 1.85
        let beta = array![abs_b];
        let lam = 0.5;
        let num = abs_b * abs_b - 2.0 * A * lam * abs_b + lam * lam;
        let expected = lam * abs_b - num / (2.0 * (A - 1.0));
        assert_abs_diff_eq!(pen.value(beta.view()), expected, epsilon = 1e-12);
    }

    #[test]
    fn prox_coord_delegates_with_correct_weight() {
        let weights = array![0.5, 1.0, 2.0];
        let pen = Scad::with_weights(0.4, A, weights.clone());
        for j in 0..3 {
            for &z in &[-1.5_f64, -0.2, 0.2, 1.5] {
                for &step in &[0.5_f64, 1.0] {
                    assert_abs_diff_eq!(
                        pen.prox_coord(j, z, step),
                        scad_prox(z, step, 0.4, A, weights[j]),
                        epsilon = 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn prox_coord_indexes_weights_by_j() {
        let pen = Scad::with_weights(1.0, A, array![0.0, 1.0]);
        // j=0: weight 0 ⇒ identity.
        assert_abs_diff_eq!(pen.prox_coord(0, 0.5, 1.0), 0.5, epsilon = 1e-12);
        // j=1: weight 1, lasso-regime threshold = 1.0 ⇒ z=0.5 → 0.
        assert_abs_diff_eq!(pen.prox_coord(1, 0.5, 1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn weights_view_returns_user_supplied() {
        let pen = Scad::with_weights(0.5, A, array![0.25, 1.0, 4.0]);
        let w = pen.weights();
        assert_eq!(w.len(), 3);
        assert_abs_diff_eq!(w[0], 0.25);
        assert_abs_diff_eq!(w[1], 1.0);
        assert_abs_diff_eq!(w[2], 4.0);
    }

    #[test]
    fn default_weights_are_ones() {
        let pen = Scad::new(0.5, A, 4);
        for v in pen.weights().iter() {
            assert_abs_diff_eq!(*v, 1.0);
        }
    }
}
