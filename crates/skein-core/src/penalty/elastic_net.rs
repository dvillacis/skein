//! Elastic-net penalty (Zou & Hastie 2005): `α λ |β_j| + (1-α) λ β_j² / 2`
//! per feature, weighted by `w_j`.
//!
//! Convex (the ridge term is strictly convex), so the M1 CD solver
//! converges to the global optimum without LLA. The prox is closed-
//! form via [`crate::prox::elastic_net_prox`] — soft-threshold the L1
//! component, then divide by the ridge shrinkage factor.
//!
//! `α = 1` recovers pure lasso (this matches `crate::penalty::Mcp` at
//! `γ → ∞`); `α = 0` recovers pure ridge. The classical glmnet default
//! is `α ∈ (0, 1)`.

use super::Penalty;
use crate::prox::elastic_net_prox;
use ndarray::{Array1, ArrayView1};

pub struct ElasticNet {
    lambda: f64,
    alpha: f64,
    /// User-supplied per-feature weights (apply to both L1 and L2 parts).
    weights: Array1<f64>,
    /// L1-effective per-feature weights = `α · weights`. Returned by
    /// the `weights()` trait accessor because every solver-side caller
    /// (`lambda_max`, strong-rule screening, KKT verification) treats
    /// `weights()` as the per-feature L1 active-set-boundary
    /// multipliers — for elastic net those are `α·w_j` (the ridge
    /// term contributes 0 to the subdifferential at β = 0).
    weights_l1: Array1<f64>,
}

impl ElasticNet {
    /// Construct an elastic-net penalty with uniform per-feature weights.
    ///
    /// Panics if `alpha ∉ [0, 1]`.
    pub fn new(lambda: f64, alpha: f64, n_features: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "ElasticNet: alpha must be in [0, 1]; got {alpha}"
        );
        let weights = Array1::ones(n_features);
        let weights_l1 = &weights * alpha;
        Self {
            lambda,
            alpha,
            weights,
            weights_l1,
        }
    }

    /// Construct an elastic-net penalty with per-feature weights.
    pub fn with_weights(lambda: f64, alpha: f64, weights: Array1<f64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "ElasticNet: alpha must be in [0, 1]; got {alpha}"
        );
        let weights_l1 = &weights * alpha;
        Self {
            lambda,
            alpha,
            weights,
            weights_l1,
        }
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// User-supplied per-feature weights (`w_j`), distinct from the
    /// L1-effective view returned by [`weights()`](Self::weights).
    pub fn raw_weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }
}

impl Penalty for ElasticNet {
    fn value(&self, beta: ArrayView1<f64>) -> f64 {
        let mut total = 0.0;
        for (j, &b) in beta.iter().enumerate() {
            let w_lam = self.weights[j] * self.lambda;
            total += w_lam * (self.alpha * b.abs() + 0.5 * (1.0 - self.alpha) * b * b);
        }
        total
    }

    fn prox_coord(&self, j: usize, z: f64, step: f64) -> f64 {
        elastic_net_prox(z, step, self.lambda, self.alpha, self.weights[j])
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights_l1.view()
    }

    fn dual_correction(&self, beta: ArrayView1<'_, f64>) -> f64 {
        // celer's `dual_enet`: subtract `α(1−l1_ratio)/2 · Σ w_j · β_j²`
        // (their α / l1_ratio map to skein's λ / α). For α = 1 the
        // ridge term vanishes — pure lasso, no correction needed.
        if self.alpha >= 1.0 {
            return 0.0;
        }
        let factor = 0.5 * self.lambda * (1.0 - self.alpha);
        let mut s = 0.0_f64;
        for (j, &b) in beta.iter().enumerate() {
            let w = self.weights[j];
            if w > 0.0 && w.is_finite() {
                s += w * b * b;
            }
        }
        factor * s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::penalty::Mcp;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn value_zero_at_origin() {
        let pen = ElasticNet::new(0.5, 0.5, 4);
        assert_abs_diff_eq!(pen.value(array![0.0, 0.0, 0.0, 0.0].view()), 0.0);
    }

    #[test]
    fn value_matches_hand_computation() {
        let pen = ElasticNet::with_weights(0.4, 0.5, array![1.0, 2.0]);
        let beta = array![1.0, -0.5];
        // Per-feature: λ·w·(α·|β| + (1-α)·β²/2).
        // j=0: 0.4·1·(0.5·1 + 0.5·0.5) = 0.4·0.75 = 0.30
        // j=1: 0.4·2·(0.5·0.5 + 0.5·0.125) = 0.8·0.3125 = 0.25
        // Total: 0.55
        assert_abs_diff_eq!(pen.value(beta.view()), 0.55, epsilon = 1e-12);
    }

    #[test]
    fn alpha_one_prox_matches_pure_lasso_via_mcp_high_gamma() {
        // ElasticNet at α=1 is pure lasso; MCP at large γ also reduces
        // to lasso. Their per-coordinate prox should match closely.
        let p = 5;
        let lambda = 0.3;
        let en = ElasticNet::new(lambda, 1.0, p);
        let mcp = Mcp::new(lambda, 1e8, p);
        for z in [-1.5, -0.5, -0.1, 0.0, 0.1, 0.5, 1.5] {
            for step in [0.5, 1.0, 2.0] {
                let pe = en.prox_coord(0, z, step);
                let pm = mcp.prox_coord(0, z, step);
                assert_abs_diff_eq!(pe, pm, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn raw_weights_returns_user_supplied() {
        let pen = ElasticNet::with_weights(0.1, 0.5, array![0.5, 1.0, 2.0]);
        let raw = pen.raw_weights();
        assert_eq!(raw.len(), 3);
        assert_abs_diff_eq!(raw[0], 0.5);
        assert_abs_diff_eq!(raw[1], 1.0);
        assert_abs_diff_eq!(raw[2], 2.0);
    }

    #[test]
    fn weights_accessor_returns_l1_effective() {
        // The `Penalty::weights()` accessor must return the L1-effective
        // weights `α · w_j` so that lambda_max and strong-rule screening
        // see the right active-set-boundary multipliers.
        let pen = ElasticNet::with_weights(0.1, 0.5, array![0.5, 1.0, 2.0]);
        let l1 = pen.weights();
        // α · raw = 0.5 · [0.5, 1.0, 2.0] = [0.25, 0.5, 1.0].
        assert_abs_diff_eq!(l1[0], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(l1[1], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(l1[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn weights_accessor_at_alpha_one_matches_raw() {
        let pen = ElasticNet::with_weights(0.1, 1.0, array![0.5, 1.0, 2.0]);
        let l1 = pen.weights();
        let raw = pen.raw_weights();
        for j in 0..3 {
            assert_abs_diff_eq!(l1[j], raw[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn weights_accessor_at_alpha_zero_is_all_zeros() {
        // Pure ridge: no L1 active-set boundary.
        let pen = ElasticNet::with_weights(0.1, 0.0, array![0.5, 1.0, 2.0]);
        let l1 = pen.weights();
        for j in 0..3 {
            assert_abs_diff_eq!(l1[j], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "alpha must be in [0, 1]")]
    fn panics_on_alpha_above_one() {
        let _ = ElasticNet::new(0.1, 1.5, 3);
    }

    #[test]
    #[should_panic(expected = "alpha must be in [0, 1]")]
    fn panics_on_negative_alpha() {
        let _ = ElasticNet::new(0.1, -0.1, 3);
    }

    /// Solver-level equivalence: at α=1 (pure lasso), the elastic-net
    /// path must match an MCP-at-γ=∞ path on the same problem within
    /// machine precision. Validates that the ElasticNet penalty wires
    /// through the existing CD path solver correctly.
    #[test]
    fn elastic_net_alpha_one_solver_path_matches_mcp_high_gamma() {
        use crate::datafit::LeastSquares;
        use crate::design::DenseMatrix;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};
        use ndarray::Array2;

        let n = 40;
        let p = 6;
        let mut state = 11_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let y = Array1::<f64>::from_shape_fn(n, |_| 0.3 * sample());
        let design = DenseMatrix::new(x);

        let cfg = PathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-12,
                acceleration: Some(5),
            },
            screening: Screening::Off,
            p0: 10,
        };
        let datafit_a = LeastSquares::new(y.clone());
        let datafit_b = LeastSquares::new(y);

        let make_mcp = |lam: f64| -> Box<dyn crate::Penalty> { Box::new(Mcp::new(lam, 1e8, p)) };
        let make_en =
            |lam: f64| -> Box<dyn crate::Penalty> { Box::new(ElasticNet::new(lam, 1.0, p)) };

        let (betas_mcp, _) = solve_path(&design, &datafit_a, make_mcp, &cfg);
        let (betas_en, _) = solve_path(&design, &datafit_b, make_en, &cfg);

        assert_eq!(betas_mcp.shape(), betas_en.shape());
        for k in 0..betas_mcp.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_mcp[[k, j]], betas_en[[k, j]], epsilon = 1e-6);
            }
        }
    }

    /// At α=0 (pure ridge), the LS+ridge problem has a closed-form
    /// solution: `β = (XᵀX/n + λ I)⁻¹ Xᵀy/n`. Run skein at α=0 and
    /// compare against the closed form.
    #[test]
    fn elastic_net_alpha_zero_recovers_closed_form_ridge() {
        use crate::datafit::LeastSquares;
        use crate::design::DenseMatrix;
        use crate::solver::{cd_solve, CdConfig};
        use ndarray::Array2;

        let n = 50;
        let p = 4;
        let mut state = 7_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let y = Array1::<f64>::from_shape_fn(n, |_| 0.3 * sample());

        // Closed-form ridge: β = (XᵀX/n + λI)⁻¹ · Xᵀy/n.
        let lambda = 0.5_f64;
        let xtx = x.t().dot(&x) / (n as f64);
        let mut a = xtx.clone();
        for j in 0..p {
            a[[j, j]] += lambda;
        }
        let xty = x.t().dot(&y) / (n as f64);
        // Tiny p; solve via Gauss-Jordan on a 4×4. Build augmented matrix.
        let mut aug = Array2::<f64>::zeros((p, p + 1));
        for i in 0..p {
            for j in 0..p {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, p]] = xty[i];
        }
        for k in 0..p {
            // Pivot.
            let pivot = aug[[k, k]];
            for j in 0..=p {
                aug[[k, j]] /= pivot;
            }
            for i in 0..p {
                if i == k {
                    continue;
                }
                let factor = aug[[i, k]];
                for j in 0..=p {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }
        let beta_closed: Array1<f64> = (0..p).map(|i| aug[[i, p]]).collect();

        // skein α=0 ridge solve.
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let pen = ElasticNet::new(lambda, 0.0, p);
        let (beta_skein, _) = cd_solve(
            &design,
            &datafit,
            &pen,
            &CdConfig {
                max_iter: 10000,
                tol: 1e-12,
                acceleration: Some(5),
            },
        );

        for j in 0..p {
            assert_abs_diff_eq!(beta_skein[j], beta_closed[j], epsilon = 1e-7);
        }
    }
}
