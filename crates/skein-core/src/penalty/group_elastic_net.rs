//! Group elastic-net penalty: `α λ w_g ‖β_g‖₂ + (1-α) λ w_g ‖β_g‖₂² / 2`
//! per group, weighted by `w_g`.
//!
//! Convex (the per-block ridge term is strictly convex), so M2's block-CD
//! solver converges to the global optimum with no LLA. The block prox is
//! closed-form via [`crate::prox::group_elastic_net_prox`] — block soft-
//! threshold, then divide by the per-block ridge shrinkage factor.
//!
//! `α = 1` recovers plain group lasso (matches [`crate::penalty::GroupLasso`]
//! exactly); `α = 0` recovers per-block ridge with no thresholding.

use super::GroupPenalty;
use crate::groups::Groups;
use crate::prox::group_elastic_net_prox;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

pub struct GroupElasticNet {
    lambda: f64,
    alpha: f64,
    /// User-supplied per-group weights (apply to both group-L2 and ridge parts).
    weights: Array1<f64>,
    /// L1-effective per-group weights = `α · weights`. Returned by the
    /// `weights()` trait accessor because every solver-side caller
    /// (`block_lambda_max`, block strong-rule screening, block KKT
    /// verification) treats `weights()` as the per-group group-L2
    /// active-set-boundary multipliers — for group elastic net those
    /// are `α·w_g` (the ridge term contributes 0 to the subdifferential
    /// at β_g = 0).
    weights_l1: Array1<f64>,
}

impl GroupElasticNet {
    /// Construct with uniform per-group weights.
    ///
    /// Panics if `alpha ∉ [0, 1]`.
    pub fn new(lambda: f64, alpha: f64, n_groups: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "GroupElasticNet: alpha must be in [0, 1]; got {alpha}"
        );
        let weights = Array1::ones(n_groups);
        let weights_l1 = &weights * alpha;
        Self {
            lambda,
            alpha,
            weights,
            weights_l1,
        }
    }

    /// Construct with per-group weights.
    pub fn with_weights(lambda: f64, alpha: f64, weights: Array1<f64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "GroupElasticNet: alpha must be in [0, 1]; got {alpha}"
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

    /// User-supplied per-group weights (`w_g`), distinct from the
    /// L1-effective view returned by [`weights()`](Self::weights).
    pub fn raw_weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }
}

impl GroupPenalty for GroupElasticNet {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64 {
        let mut total = 0.0;
        for g in 0..groups.n_groups() {
            let block_norm: f64 = groups
                .group(g)
                .iter()
                .map(|&j| beta[j] * beta[j])
                .sum::<f64>()
                .sqrt();
            let w_lam = self.weights[g] * self.lambda;
            total += w_lam
                * (self.alpha * block_norm + 0.5 * (1.0 - self.alpha) * block_norm * block_norm);
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let slice = block.as_slice_mut().expect("contiguous block expected");
        group_elastic_net_prox(slice, step, self.lambda, self.alpha, self.weights[g]);
    }

    fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights_l1.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datafit::LeastSquares;
    use crate::design::{DenseMatrix, DesignMatrix};
    use crate::penalty::GroupLasso;
    use crate::solver::{block_cd_solve, solve_block_path, BlockPathConfig, CdConfig, Screening};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    // ---- penalty value --------------------------------------------------

    #[test]
    fn value_zero_at_origin() {
        let pen = GroupElasticNet::new(0.5, 0.5, 2);
        let groups = Groups::contiguous_blocks(4, 2);
        assert_abs_diff_eq!(pen.value(array![0.0, 0.0, 0.0, 0.0].view(), &groups), 0.0);
    }

    #[test]
    fn value_matches_hand_computation() {
        // p=4, two groups of 2; β = [1, -1, 0.5, 0.5]; λ = 0.1, α = 0.5.
        // Group L2 norms: ‖β_0‖ = √2, ‖β_1‖ = √0.5.
        // Per-group: λ·w·(α·‖_2 + (1-α)·‖_2²/2)
        // group 0 (w=1): 0.1·(0.5·√2 + 0.5·2/2) = 0.1·(0.7071 + 0.5) = 0.12071
        // group 1 (w=2): 0.1·2·(0.5·√0.5 + 0.5·0.5/2) = 0.2·(0.3536 + 0.125)
        //                = 0.2·0.4786 = 0.09571
        // Total ≈ 0.21642
        let pen = GroupElasticNet::with_weights(0.1, 0.5, array![1.0, 2.0]);
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = array![1.0, -1.0, 0.5, 0.5];
        let v = pen.value(beta.view(), &groups);
        let expected = 0.1 * (0.5 * (2.0_f64).sqrt() + 0.5 * 2.0 / 2.0)
            + 0.1 * 2.0 * (0.5 * (0.5_f64).sqrt() + 0.5 * 0.5 / 2.0);
        assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
    }

    // ---- weights accessor ----------------------------------------------

    #[test]
    fn raw_weights_returns_user_supplied() {
        let pen = GroupElasticNet::with_weights(0.1, 0.5, array![0.5, 1.0, 2.0]);
        let raw = pen.raw_weights();
        assert_eq!(raw.len(), 3);
        assert_abs_diff_eq!(raw[0], 0.5);
        assert_abs_diff_eq!(raw[1], 1.0);
        assert_abs_diff_eq!(raw[2], 2.0);
    }

    #[test]
    fn weights_accessor_returns_l1_effective() {
        let pen = GroupElasticNet::with_weights(0.1, 0.5, array![0.5, 1.0, 2.0]);
        let l1 = pen.weights();
        assert_abs_diff_eq!(l1[0], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(l1[1], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(l1[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn weights_accessor_at_alpha_one_matches_raw() {
        let pen = GroupElasticNet::with_weights(0.1, 1.0, array![0.5, 1.0, 2.0]);
        let l1 = pen.weights();
        let raw = pen.raw_weights();
        for g in 0..3 {
            assert_abs_diff_eq!(l1[g], raw[g], epsilon = 1e-12);
        }
    }

    #[test]
    fn weights_accessor_at_alpha_zero_is_all_zeros() {
        // Pure block ridge: no L2 active-set boundary.
        let pen = GroupElasticNet::with_weights(0.1, 0.0, array![0.5, 1.0, 2.0]);
        let l1 = pen.weights();
        for g in 0..3 {
            assert_abs_diff_eq!(l1[g], 0.0, epsilon = 1e-12);
        }
    }

    // ---- panics --------------------------------------------------------

    #[test]
    #[should_panic(expected = "alpha must be in [0, 1]")]
    fn panics_on_alpha_above_one() {
        let _ = GroupElasticNet::new(0.1, 1.5, 3);
    }

    #[test]
    #[should_panic(expected = "alpha must be in [0, 1]")]
    fn panics_on_negative_alpha() {
        let _ = GroupElasticNet::new(0.1, -0.1, 3);
    }

    // ---- prox ----------------------------------------------------------

    #[test]
    fn prox_zeros_small_block_under_strong_l1() {
        // λ=10, α=1, step=1, weight=1 → l1_thr = 10 ≫ ‖block‖.
        let pen = GroupElasticNet::new(10.0, 1.0, 1);
        let mut block = array![0.3, -0.4];
        pen.prox_group(0, block.view_mut(), 1.0);
        for &v in block.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn prox_at_alpha_one_matches_group_lasso_prox() {
        // α = 1 ⇒ pure group lasso. Match GroupLasso prox exactly on
        // the same block.
        let pen_en = GroupElasticNet::with_weights(0.5, 1.0, array![1.5]);
        let pen_gl = GroupLasso::with_weights(0.5, array![1.5]);
        for &block_init in &[[3.0_f64, 4.0], [-1.0, 0.5], [0.0, 0.0]] {
            for &step in &[0.5_f64, 1.0, 2.0] {
                let mut a = Array1::from(block_init.to_vec());
                let mut b = Array1::from(block_init.to_vec());
                pen_en.prox_group(0, a.view_mut(), step);
                pen_gl.prox_group(0, b.view_mut(), step);
                assert_abs_diff_eq!(a[0], b[0], epsilon = 1e-12);
                assert_abs_diff_eq!(a[1], b[1], epsilon = 1e-12);
            }
        }
    }

    // ---- problem builders ----------------------------------------------

    fn group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Groups) {
        let n = 60;
        let p = 8;
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let mut true_beta = Array1::<f64>::zeros(p);
        true_beta[0] = 1.5;
        true_beta[1] = -1.0;
        true_beta[4] = 0.7;
        true_beta[5] = 1.2;
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let groups = Groups::contiguous_blocks(p, 2);
        (DenseMatrix::new(x), y, groups)
    }

    // ---- solver-level reductions ---------------------------------------

    /// At α=1, the GroupElasticNet path must coincide with the GroupLasso
    /// path on the same problem within machine precision. Validates that
    /// `weights_l1` correctly drives `block_lambda_max` / strong-rule /
    /// KKT cycle to identify the same active set.
    #[test]
    fn group_elastic_net_alpha_one_path_matches_group_lasso() {
        let (design, y, groups) = group_problem(50);
        let cfg = BlockPathConfig {
            n_lambdas: 8,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 5000,
                tol: 1e-12,
                acceleration: Some(5),
            },
            screening: Screening::Off,
            parallel: false,
        };
        let datafit_a = LeastSquares::new(y.clone());
        let datafit_b = LeastSquares::new(y);

        let make_gl = |lam: f64| -> Box<dyn GroupPenalty> {
            Box::new(GroupLasso::new(lam, groups.n_groups()))
        };
        let make_gen = |lam: f64| -> Box<dyn GroupPenalty> {
            Box::new(GroupElasticNet::new(lam, 1.0, groups.n_groups()))
        };

        let (betas_gl, _) = solve_block_path(&design, &datafit_a, make_gl, &groups, &cfg);
        let (betas_gen, _) = solve_block_path(&design, &datafit_b, make_gen, &groups, &cfg);

        assert_eq!(betas_gl.shape(), betas_gen.shape());
        for k in 0..betas_gl.nrows() {
            for j in 0..design.n_features() {
                assert_abs_diff_eq!(betas_gl[[k, j]], betas_gen[[k, j]], epsilon = 1e-6);
            }
        }
    }

    /// At α=0, the LS + block-ridge problem has a closed-form per-group
    /// solution: each group block solves `(X_gᵀ X_g/n + λ I) β_g = X_gᵀ
    /// (y - X_{−g}β_{−g})/n`. Because every group is hit by the same λI
    /// shift, the global solution coincides with classical ridge
    /// `β = (XᵀX/n + λI)⁻¹ Xᵀy/n` (per-feature ridge with uniform
    /// per-group weight = 1). Validate against the closed form.
    #[test]
    fn group_elastic_net_alpha_zero_recovers_closed_form_ridge() {
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

        let lambda = 0.5_f64;

        // Closed-form: β = (XᵀX/n + λI)⁻¹ · Xᵀy/n.
        let xtx = x.t().dot(&x) / (n as f64);
        let mut a = xtx.clone();
        for j in 0..p {
            a[[j, j]] += lambda;
        }
        let xty = x.t().dot(&y) / (n as f64);
        let mut aug = Array2::<f64>::zeros((p, p + 1));
        for i in 0..p {
            for j in 0..p {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, p]] = xty[i];
        }
        for k in 0..p {
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

        // skein α=0 group ridge solve. Use singleton groups so each
        // "group" is one feature — keeps the CD/closed-form mapping
        // direct and side-steps the per-group L2 prox structure
        // (which is irrelevant when α=0 since prox is just per-coord
        // shrinkage by 1/(1+step·λ)).
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let groups = Groups::singletons(p);
        let pen = GroupElasticNet::new(lambda, 0.0, p);
        let (beta_skein, _) = block_cd_solve(
            &design,
            &datafit,
            &pen,
            &groups,
            &CdConfig {
                max_iter: 20000,
                tol: 1e-12,
                acceleration: Some(5),
            },
        );

        for j in 0..p {
            assert_abs_diff_eq!(beta_skein[j], beta_closed[j], epsilon = 1e-7);
        }
    }
}
