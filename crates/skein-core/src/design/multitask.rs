//! Virtual multi-task design wrapper.
//!
//! Multi-task LS with response matrix `Y ∈ ℝ^{n×K}` and coefficient matrix
//! `B ∈ ℝ^{p×K}` reduces algebraically to a single group-lasso problem if
//! we lay `B` out row-major (`bvec[jK+k] = B[j,k]`) and stack `Y` task-
//! outer (`yvec[k·n+i] = Y[i,k]`). The virtual design `X̃ ∈ ℝ^{nK×pK}`
//! has one column per `(feature, task)` pair: column `jK+k` holds the
//! base column `X[:, j]` lifted into row block `k` (rows `k·n..(k+1)·n`)
//! and zeros elsewhere.
//!
//! With this layout, the row-grouping of `B` (which is what multi-task
//! lasso penalizes — `λ Σ_j w_j ‖B[j, :]‖_2`) becomes the standard
//! contiguous-block grouping `{jK, jK+1, …, jK+K-1}` for each feature `j`,
//! which the existing M2 block-CD machinery handles unchanged.
//!
//! Per-virtual-column ops are O(n) — the same cost as a scalar column op
//! on the base — and the wrapper carries no state beyond the inner
//! design and the task count `K`.
//!
//! Composes with [`Augmented`](super::Augmented) and
//! [`Standardized`](super::Standardized) the same way every other backend
//! does (intercept handling and per-task scaling are layered on top).

use super::DesignMatrix;
use crate::groups::Groups;
use ndarray::{s, Array1, Array2, ArrayView1};

pub struct MultiTaskDesign<D: DesignMatrix> {
    base: D,
    n_tasks: usize,
}

impl<D: DesignMatrix> MultiTaskDesign<D> {
    /// Wrap `base` as the design for a `K`-task multi-response problem.
    /// Panics if `n_tasks == 0`.
    pub fn new(base: D, n_tasks: usize) -> Self {
        assert!(n_tasks > 0, "MultiTaskDesign: n_tasks must be ≥ 1");
        Self { base, n_tasks }
    }

    pub fn base(&self) -> &D {
        &self.base
    }

    pub fn n_tasks(&self) -> usize {
        self.n_tasks
    }

    /// Build the row-grouping that pairs each base feature with its `K`
    /// virtual columns: group `j` covers indices `{jK, jK+1, …, jK+K-1}`.
    /// Pass this to the M2 block-CD path solvers alongside a
    /// `MultiTaskDesign`.
    pub fn auto_groups(n_features: usize, n_tasks: usize) -> Groups {
        Groups::contiguous_blocks(n_features * n_tasks, n_tasks)
    }
}

impl<D: DesignMatrix> DesignMatrix for MultiTaskDesign<D> {
    fn n_samples(&self) -> usize {
        self.base.n_samples() * self.n_tasks
    }

    fn n_features(&self) -> usize {
        self.base.n_features() * self.n_tasks
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        let k = self.n_tasks;
        let n = self.base.n_samples();
        debug_assert_eq!(beta.len(), self.base.n_features() * k);
        let mut out = Array1::<f64>::zeros(n * k);
        // Task-by-task: r[k*n..(k+1)*n] = X · β[k::K] where the strided
        // view picks out the kth task's coefficient column from the
        // row-major bvec.
        for task in 0..k {
            let beta_task = beta.slice(s![task..; k as isize]);
            let r_task = self.base.matvec(beta_task);
            out.slice_mut(s![task * n..(task + 1) * n]).assign(&r_task);
        }
        out
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        let k = self.n_tasks;
        let n = self.base.n_samples();
        let p = self.base.n_features();
        debug_assert_eq!(r.len(), n * k);
        let mut out = Array1::<f64>::zeros(p * k);
        for task in 0..k {
            let r_task = r.slice(s![task * n..(task + 1) * n]);
            let g_task = self.base.rmatvec(r_task);
            for j in 0..p {
                out[j * k + task] = g_task[j];
            }
        }
        out
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        let k = self.n_tasks;
        let n = self.base.n_samples();
        let feature = j / k;
        let task = j % k;
        self.base
            .col_dot(feature, v.slice(s![task * n..(task + 1) * n]))
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        let feature = j / self.n_tasks;
        // The K virtual columns associated with feature j live in
        // disjoint row blocks, so each one's squared norm equals the
        // base column's squared norm — independent of the task index.
        self.base.col_sq_norm(feature)
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let k = self.n_tasks;
        let n = self.base.n_samples();
        let mut out = Array2::<f64>::zeros((n * k, cols.len()));
        // Pull each requested base feature once, then scatter its
        // contents into the right row block per virtual column.
        // Group requested cols by their base-feature index to avoid
        // calling `base.columns(...)` redundantly for the same feature.
        let mut feature_indices: Vec<usize> = cols.iter().map(|&c| c / k).collect();
        feature_indices.sort();
        feature_indices.dedup();
        let base_block = self.base.columns(&feature_indices);
        // Build a lookup so we can index `base_block` by feature.
        let mut feature_pos = std::collections::HashMap::with_capacity(feature_indices.len());
        for (slot, &feat) in feature_indices.iter().enumerate() {
            feature_pos.insert(feat, slot);
        }
        for (out_col, &c) in cols.iter().enumerate() {
            let feature = c / k;
            let task = c % k;
            let slot = feature_pos[&feature];
            let base_col = base_block.column(slot);
            // Place X[:, feature] in rows [task*n, (task+1)*n) of out's column.
            out.slice_mut(s![task * n..(task + 1) * n, out_col])
                .assign(&base_col);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    /// Hand-build the (nK × pK) reference X̃ that `MultiTaskDesign` is
    /// virtualizing, so we can assert agreement on every method.
    fn build_problem() -> (DenseMatrix, DenseMatrix, usize, usize, usize) {
        let n = 4;
        let p = 3;
        let k = 2;
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0],
        ];
        let mut x_tilde = Array2::<f64>::zeros((n * k, p * k));
        for j in 0..p {
            for task in 0..k {
                let col_idx = j * k + task;
                for i in 0..n {
                    x_tilde[[task * n + i, col_idx]] = x[[i, j]];
                }
            }
        }
        (DenseMatrix::new(x), DenseMatrix::new(x_tilde), n, p, k)
    }

    #[test]
    fn n_samples_and_n_features_scale_by_k() {
        let (base, _ref, n, p, k) = build_problem();
        let mt = MultiTaskDesign::new(base, k);
        assert_eq!(mt.n_samples(), n * k);
        assert_eq!(mt.n_features(), p * k);
        assert_eq!(mt.n_tasks(), k);
    }

    #[test]
    fn matvec_matches_reference() {
        let (base, ref_dense, n, p, k) = build_problem();
        let mt = MultiTaskDesign::new(base, k);
        // β corresponds to B with row-major layout.
        let beta = array![0.5, -1.0, 2.0, 0.3, -0.7, 1.5];
        let r_ref = ref_dense.matvec(beta.view());
        let r_mt = mt.matvec(beta.view());
        assert_eq!(r_ref.len(), n * k);
        assert_eq!(r_mt.len(), n * k);
        for i in 0..n * k {
            assert_abs_diff_eq!(r_mt[i], r_ref[i], epsilon = 1e-12);
        }
        let _ = p; // silence unused
    }

    #[test]
    fn rmatvec_matches_reference() {
        let (base, ref_dense, n, p, k) = build_problem();
        let mt = MultiTaskDesign::new(base, k);
        let r = array![1.0, -0.5, 2.0, 0.3, 0.1, -1.1, 0.7, 0.4];
        let g_ref = ref_dense.rmatvec(r.view());
        let g_mt = mt.rmatvec(r.view());
        assert_eq!(g_ref.len(), p * k);
        assert_eq!(g_mt.len(), p * k);
        for j in 0..p * k {
            assert_abs_diff_eq!(g_mt[j], g_ref[j], epsilon = 1e-12);
        }
        let _ = n;
    }

    #[test]
    fn col_dot_and_sq_norm_match_reference() {
        let (base, ref_dense, n, p, k) = build_problem();
        let mt = MultiTaskDesign::new(base, k);
        let v = array![0.7, -1.2, 3.0, 0.1, 1.5, -0.3, 2.1, 0.8];
        for j in 0..p * k {
            assert_abs_diff_eq!(
                mt.col_dot(j, v.view()),
                ref_dense.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(mt.col_sq_norm(j), ref_dense.col_sq_norm(j), epsilon = 1e-12);
        }
        let _ = n;
    }

    #[test]
    fn columns_block_matches_reference_for_full_feature_groups() {
        let (base, ref_dense, _n, _p, k) = build_problem();
        let mt = MultiTaskDesign::new(base, k);
        // The block-CD solver always asks for whole row-groups together
        // (group g = {gK, gK+1, ..., gK+K-1}), but the trait contract
        // allows arbitrary index lists — exercise a mixed pattern.
        let cols = [3_usize, 0, 5, 2];
        let blk_ref = ref_dense.columns(&cols);
        let blk_mt = mt.columns(&cols);
        assert_eq!(blk_ref.shape(), blk_mt.shape());
        for i in 0..blk_ref.nrows() {
            for c in 0..cols.len() {
                assert_abs_diff_eq!(blk_mt[[i, c]], blk_ref[[i, c]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn auto_groups_pairs_each_feature_with_k_virtual_columns() {
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(3, 2);
        assert_eq!(groups.n_groups(), 3);
        assert_eq!(groups.group(0), &[0, 1]);
        assert_eq!(groups.group(1), &[2, 3]);
        assert_eq!(groups.group(2), &[4, 5]);
    }

    #[test]
    fn k_equals_one_is_the_identity_wrapper() {
        // K = 1 is degenerate: virtual design = base design.
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let base = DenseMatrix::new(x);
        let mt = MultiTaskDesign::new(base, 1);
        assert_eq!(mt.n_samples(), 3);
        assert_eq!(mt.n_features(), 2);
        let beta = array![0.5, -1.0];
        let r_mt = mt.matvec(beta.view());
        // Same as base @ beta.
        let r_ref = mt.base().matvec(beta.view());
        for i in 0..3 {
            assert_abs_diff_eq!(r_mt[i], r_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "n_tasks must be ≥ 1")]
    fn panics_on_zero_tasks() {
        let x = array![[1.0]];
        let _ = MultiTaskDesign::new(DenseMatrix::new(x), 0);
    }

    /// `MultiTaskDesign` is generic over the inner backend, so wrapping
    /// `SparseCSC` should "just work". Validate by comparing the sparse-
    /// backed solver path to the dense-backed reference on the same
    /// problem.
    #[test]
    fn solver_path_matches_dense_reference_with_sparse_inner() {
        use crate::datafit::LeastSquares;
        use crate::design::SparseCSC;
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{solve_block_path, BlockPathConfig, CdConfig, Screening};

        let n = 30;
        let p = 5;
        let k = 3;

        let mut state: u64 = 211;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        // Build a sparse-ish X: ~50% nonzeros.
        let mut x_dense = Array2::<f64>::zeros((n, p));
        let mut data: Vec<f64> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0];
        for j in 0..p {
            for i in 0..n {
                let v = sample();
                if v.abs() > 0.5 {
                    x_dense[[i, j]] = v;
                    data.push(v);
                    indices.push(i);
                }
            }
            indptr.push(data.len());
        }
        let y = Array1::<f64>::from_shape_fn(n * k, |_| 0.4 * sample());

        let cfg = BlockPathConfig {
            n_lambdas: 6,
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
        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let n_groups = groups.n_groups();

        let dense = MultiTaskDesign::new(DenseMatrix::new(x_dense), k);
        let csc = SparseCSC::new(
            n,
            Array1::from(data),
            Array1::from(indices),
            Array1::from(indptr),
        );
        let sparse = MultiTaskDesign::new(csc, k);

        let datafit_d = LeastSquares::new(y.clone());
        let datafit_s = LeastSquares::new(y);
        let make_pen_d =
            |lam: f64| -> Box<dyn GroupPenalty> { Box::new(GroupLasso::new(lam, n_groups)) };
        let make_pen_s =
            |lam: f64| -> Box<dyn GroupPenalty> { Box::new(GroupLasso::new(lam, n_groups)) };

        let (betas_d, _) = solve_block_path(&dense, &datafit_d, make_pen_d, &groups, &cfg);
        let (betas_s, _) = solve_block_path(&sparse, &datafit_s, make_pen_s, &groups, &cfg);

        assert_eq!(betas_d.shape(), betas_s.shape());
        for k_lam in 0..betas_d.nrows() {
            for j in 0..p * k {
                assert_abs_diff_eq!(betas_d[[k_lam, j]], betas_s[[k_lam, j]], epsilon = 1e-9);
            }
        }
    }

    /// End-to-end solver-equivalence: a multi-task LS lasso path solved
    /// via `MultiTaskDesign + GroupLasso` must coincide with a
    /// hand-stacked group-lasso path on the equivalent dense `X̃` and
    /// `ỹ` within machine precision. This is the load-bearing test that
    /// validates the whole "stack-and-reuse" reduction.
    #[test]
    fn solver_path_matches_handstacked_reference() {
        use crate::datafit::LeastSquares;
        use crate::penalty::{GroupLasso, GroupPenalty};
        use crate::solver::{solve_block_path, BlockPathConfig, CdConfig, Screening};

        let n = 30;
        let p = 5;
        let k = 3;

        let mut state: u64 = 91;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let y = Array2::<f64>::from_shape_fn((n, k), |_| 0.4 * sample());

        // Hand-stack: row-major `bvec[jK+task] = B[j, task]`,
        // task-outer `yvec[task*n + i] = Y[i, task]`.
        let mut x_tilde = Array2::<f64>::zeros((n * k, p * k));
        for j in 0..p {
            for task in 0..k {
                for i in 0..n {
                    x_tilde[[task * n + i, j * k + task]] = x[[i, j]];
                }
            }
        }
        let mut y_tilde = Array1::<f64>::zeros(n * k);
        for task in 0..k {
            for i in 0..n {
                y_tilde[task * n + i] = y[[i, task]];
            }
        }

        let cfg = BlockPathConfig {
            n_lambdas: 6,
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

        let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
        let n_groups = groups.n_groups();

        // Wrapper-based fit.
        let design_mt = MultiTaskDesign::new(DenseMatrix::new(x.clone()), k);
        let datafit_mt = LeastSquares::new(y_tilde.clone());
        let make_pen_mt =
            |lam: f64| -> Box<dyn GroupPenalty> { Box::new(GroupLasso::new(lam, n_groups)) };
        let (betas_mt, _) = solve_block_path(&design_mt, &datafit_mt, make_pen_mt, &groups, &cfg);

        // Hand-stacked reference fit: same group-lasso path on X̃, ỹ.
        let design_ref = DenseMatrix::new(x_tilde);
        let datafit_ref = LeastSquares::new(y_tilde);
        let make_pen_ref =
            |lam: f64| -> Box<dyn GroupPenalty> { Box::new(GroupLasso::new(lam, n_groups)) };
        let (betas_ref, _) =
            solve_block_path(&design_ref, &datafit_ref, make_pen_ref, &groups, &cfg);

        assert_eq!(betas_mt.shape(), betas_ref.shape());
        for k_lam in 0..betas_mt.nrows() {
            for j in 0..p * k {
                assert_abs_diff_eq!(betas_mt[[k_lam, j]], betas_ref[[k_lam, j]], epsilon = 1e-10);
            }
        }
    }
}
