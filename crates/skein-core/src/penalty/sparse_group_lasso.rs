use super::GroupPenalty;
use crate::groups::Groups;
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Sparse-group lasso (Simon, Friedman, Hastie, Tibshirani 2013).
///
/// **Uniform-weight form** (default): `λ · w_g · (α‖β_g‖₁ + (1−α)‖β_g‖₂)`.
///
/// **Two-level form** (set via [`Self::with_coord_weights`]): allows the
/// L1 part to use *per-coordinate* weights `v_{g,k}` independent of the
/// per-group L2 weight `w_g`:
/// `λ · (α · Σ_{k} v_{g,k} |β_{g,k}| + (1−α) · w_g · ‖β_g‖₂)`.
/// The two-level form is what an LLA wrapper around `SparseGroupMCP`
/// produces — its surrogate has separate per-coord L1 weights and
/// per-group L2 weights at every outer iteration.
///
/// `α = 0` reduces to plain group lasso (with `w_g`); `α = 1` reduces to
/// weighted lasso with weights `v_{g,k}` (the group structure becomes
/// irrelevant). Two-step prox: first soft-threshold each coordinate by
/// `α·step·λ·v_{g,k}`, then group-soft-threshold by `(1−α)·step·λ·w_g`.
/// The composition is exact for SGL.
pub struct SparseGroupLasso {
    lambda: f64,
    alpha: f64,
    weights: Array1<f64>,
    /// Per-group, per-position-in-group L1 weights. `None` ⇒ uniform 1.
    coord_weights: Option<Vec<Array1<f64>>>,
}

impl SparseGroupLasso {
    pub fn new(lambda: f64, alpha: f64, n_groups: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        Self {
            lambda,
            alpha,
            weights: Array1::ones(n_groups),
            coord_weights: None,
        }
    }

    pub fn with_weights(lambda: f64, alpha: f64, weights: Array1<f64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        Self {
            lambda,
            alpha,
            weights,
            coord_weights: None,
        }
    }

    /// Two-level construction: per-group L2 weights plus per-group,
    /// per-position-in-group L1 weights. `coord_weights[g]` must have
    /// length equal to the size of group `g`.
    pub fn with_coord_weights(
        lambda: f64,
        alpha: f64,
        group_weights: Array1<f64>,
        coord_weights: Vec<Array1<f64>>,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1] (got {})",
            alpha
        );
        assert_eq!(
            group_weights.len(),
            coord_weights.len(),
            "coord_weights must have one entry per group"
        );
        Self {
            lambda,
            alpha,
            weights: group_weights,
            coord_weights: Some(coord_weights),
        }
    }
}

impl GroupPenalty for SparseGroupLasso {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64 {
        let mut total = 0.0;
        for g in 0..groups.n_groups() {
            let cols = groups.group(g);
            let coord_w = self.coord_weights.as_ref().map(|cw| &cw[g]);
            let mut block_weighted_l1 = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for (k, &j) in cols.iter().enumerate() {
                let b = beta[j];
                let v_k = coord_w.map(|cw| cw[k]).unwrap_or(1.0);
                block_weighted_l1 += v_k * b.abs();
                sum_sq += b * b;
            }
            let block_l2 = sum_sq.sqrt();
            total += self.lambda
                * (self.alpha * block_weighted_l1
                    + (1.0 - self.alpha) * self.weights[g] * block_l2);
        }
        total
    }

    fn prox_group(&self, g: usize, mut block: ArrayViewMut1<f64>, step: f64) {
        let l2_thr = (1.0 - self.alpha) * step * self.lambda * self.weights[g];
        let coord_w = self.coord_weights.as_ref().map(|cw| &cw[g]);

        // Step 1: per-coordinate soft-threshold (with per-coord weight if set).
        for (k, x) in block.iter_mut().enumerate() {
            let v_k = coord_w.map(|cw| cw[k]).unwrap_or(1.0);
            let l1_thr = self.alpha * step * self.lambda * v_k;
            if *x > l1_thr {
                *x -= l1_thr;
            } else if *x < -l1_thr {
                *x += l1_thr;
            } else {
                *x = 0.0;
            }
        }

        // Step 2: block soft-threshold.
        let slice = block.as_slice_mut().expect("contiguous block expected");
        let norm: f64 = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= l2_thr {
            for x in slice.iter_mut() {
                *x = 0.0;
            }
        } else if l2_thr > 0.0 {
            let scale = 1.0 - l2_thr / norm;
            for x in slice.iter_mut() {
                *x *= scale;
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
    use crate::datafit::LeastSquares;
    use crate::design::{DenseMatrix, DesignMatrix};
    use crate::penalty::{GroupLasso, Mcp};
    use crate::solver::{block_cd_solve, cd_solve, CdConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    fn sparse_group_problem(seed: u64) -> (DenseMatrix, Array1<f64>, Groups) {
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

    // ---- penalty value --------------------------------------------------

    #[test]
    fn sparse_group_lasso_value_hand_computed() {
        // p=4, two groups of 2; β = [1, -1, 0.5, 0.5]; uniform weights;
        // λ = 0.1, α = 0.5.
        // Block 0: ‖β‖_1 = 2, ‖β‖_2 = √2 ≈ 1.4142
        // Block 1: ‖β‖_1 = 1, ‖β‖_2 = √(0.5) ≈ 0.7071
        // Per-group penalty contribution: 0.1·(0.5·‖_1 + 0.5·‖_2)
        //   group 0: 0.1 · (0.5·2 + 0.5·√2) = 0.1·(1 + 0.7071) = 0.17071
        //   group 1: 0.1 · (0.5·1 + 0.5·√0.5) = 0.1·(0.5 + 0.3536) = 0.08536
        //   total ≈ 0.25607
        let beta = array![1.0, -1.0, 0.5, 0.5];
        let groups = Groups::contiguous_blocks(4, 2);
        let pen = SparseGroupLasso::new(0.1, 0.5, 2);
        let v = pen.value(beta.view(), &groups);
        let expected =
            0.1 * (0.5 * 2.0 + 0.5 * (2.0_f64).sqrt()) + 0.1 * (0.5 * 1.0 + 0.5 * (0.5_f64).sqrt());
        assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
    }

    // ---- prox -----------------------------------------------------------

    #[test]
    fn sparse_group_lasso_prox_zeros_block_under_strong_penalty() {
        // Large λ relative to block magnitude ⇒ block goes to zero.
        let pen = SparseGroupLasso::new(10.0, 0.5, 1);
        let mut block = array![0.3, -0.4];
        pen.prox_group(0, block.view_mut(), 1.0);
        for &v in block.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    // ---- reductions to known special cases ------------------------------

    #[test]
    fn sparse_group_lasso_with_alpha_zero_matches_group_lasso_solution() {
        let (design, y, groups) = sparse_group_problem(70);
        let datafit = LeastSquares::new(y);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let lambda = 0.01;

        let pen_group = GroupLasso::new(lambda, groups.n_groups());
        let pen_sgl = SparseGroupLasso::new(lambda, 0.0, groups.n_groups());

        let (b_group, _) = block_cd_solve(&design, &datafit, &pen_group, &groups, &cfg);
        let (b_sgl, _) = block_cd_solve(&design, &datafit, &pen_sgl, &groups, &cfg);

        for j in 0..design.n_features() {
            assert_abs_diff_eq!(b_group[j], b_sgl[j], epsilon = 1e-6);
        }
    }

    #[test]
    fn sparse_group_lasso_with_alpha_one_matches_lasso_solution() {
        // α = 1 ⇒ pure L1 per coord; group structure is irrelevant.
        // Compare against scalar `cd_solve` on the equivalent lasso
        // (Mcp at γ → ∞).
        let (design, y, groups) = sparse_group_problem(71);
        let datafit = LeastSquares::new(y);
        let p = design.n_features();
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: None,
        };
        let lambda = 0.01;

        let pen_sgl = SparseGroupLasso::new(lambda, 1.0, groups.n_groups());
        let pen_lasso = Mcp::new(lambda, 1e10, p);

        let (b_sgl, _) = block_cd_solve(&design, &datafit, &pen_sgl, &groups, &cfg);
        let (b_lasso, _) = cd_solve(&design, &datafit, &pen_lasso, &cfg);

        for j in 0..p {
            assert_abs_diff_eq!(b_sgl[j], b_lasso[j], epsilon = 1e-5);
        }
    }

    // ---- two-level SGL (per-coord + per-group weights) -----------------

    #[test]
    fn sparse_group_lasso_with_coord_weights_value_hand_computed() {
        // p=4, two groups of 2; β = [1, -1, 0.5, 0.5]; λ = 0.1, α = 0.5.
        // Group L2 weights = [1, 2]; coord L1 weights per group:
        //   group 0: [3, 1] (so |β_0|·3 + |β_1|·1 = 3 + 1 = 4)
        //   group 1: [2, 2] (so |β_2|·2 + |β_3|·2 = 1 + 1 = 2)
        // Per-group:
        //   group 0: 0.1 · (0.5 · 4 + 0.5 · 1 · √2)
        //          = 0.1 · (2 + 0.5·1.4142) = 0.1·2.7071 = 0.27071
        //   group 1: 0.1 · (0.5 · 2 + 0.5 · 2 · √0.5)
        //          = 0.1 · (1 + √0.5) = 0.1·1.7071 = 0.17071
        let beta = array![1.0, -1.0, 0.5, 0.5];
        let groups = Groups::contiguous_blocks(4, 2);
        let group_w = array![1.0, 2.0];
        let coord_w = vec![array![3.0, 1.0], array![2.0, 2.0]];
        let pen = SparseGroupLasso::with_coord_weights(0.1, 0.5, group_w, coord_w);
        let v = pen.value(beta.view(), &groups);
        let expected = 0.1 * (0.5 * 4.0 + 0.5 * 1.0 * (2.0_f64).sqrt())
            + 0.1 * (0.5 * 2.0 + 0.5 * 2.0 * (0.5_f64).sqrt());
        assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
    }

    #[test]
    fn sparse_group_lasso_with_coord_weights_zeros_specific_coord() {
        // Per-coord weights amplify the L1 threshold for one coordinate
        // enough to zero it while leaving the rest intact.
        // λ=0.1, α=1 (pure L1 per coord), step=1. Coord 0 has weight 100
        // ⇒ threshold 10 ⇒ |β_0|=0.5 zeroed. Coord 1 weight 1 ⇒ threshold
        // 0.1 ⇒ |β_1|=0.5 → 0.4.
        let group_w = array![1.0];
        let coord_w = vec![array![100.0, 1.0]];
        let pen = SparseGroupLasso::with_coord_weights(0.1, 1.0, group_w, coord_w);
        let mut block = array![0.5, 0.5];
        pen.prox_group(0, block.view_mut(), 1.0);
        assert_abs_diff_eq!(block[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(block[1], 0.4, epsilon = 1e-12);
    }

    // ---- within-group sparsity recovery ---------------------------------

    #[test]
    fn sparse_group_lasso_recovers_within_group_sparsity() {
        // Construct a problem where:
        // - group 0 is partly active: feature 0 large, feature 1 zero
        // - group 1 fully inactive
        // SGL with α > 0 should zero feature 1 (within-group sparsity).
        // GroupLasso with α = 0 would keep feature 1 nonzero (whole-group activation).
        let n = 80;
        let p = 4;
        let mut state: u64 = 99;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        // Truth: only feature 0 is active.
        let true_beta = array![2.0, 0.0, 0.0, 0.0];
        let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
        let y = x.dot(&true_beta) + &noise;
        let design = DenseMatrix::new(x);
        let datafit = LeastSquares::new(y);
        let groups = Groups::contiguous_blocks(p, 2);
        let cfg = CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        };
        let lambda = 0.05;

        let pen_group = GroupLasso::new(lambda, groups.n_groups());
        let pen_sgl = SparseGroupLasso::new(lambda, 0.5, groups.n_groups());

        let (b_group, _) = block_cd_solve(&design, &datafit, &pen_group, &groups, &cfg);
        let (b_sgl, _) = block_cd_solve(&design, &datafit, &pen_sgl, &groups, &cfg);

        // Group lasso keeps the whole group active even though feature 1's
        // contribution is noise.
        assert!(
            b_group[1].abs() > 0.01,
            "group lasso should keep whole group active, got |β_1|={}",
            b_group[1].abs()
        );
        // SGL zeros feature 1 (within-group L1 prefers sparsity inside an
        // active group).
        assert!(
            b_sgl[1].abs() < b_group[1].abs() / 2.0,
            "SGL should shrink feature 1 more than group lasso (sgl={}, group={})",
            b_sgl[1].abs(),
            b_group[1].abs()
        );
    }
}
