//! Lazy column-scaling wrapper around any [`DesignMatrix`].
//!
//! For dense data, the M1 standardization machinery (`standardize` /
//! `destandardize_path`) materializes a centered + scaled `Xs`. For
//! sparse data, materializing the centered matrix would destroy
//! sparsity (every column would gain an `−x̄_j` offset on every row).
//! The `glmnet` / sklearn-with-`with_mean=False` workaround for sparse
//! data is *scale-only*: leave the column means in place, just
//! multiply each column by `1/s_j`. The intercept (when fitted) is
//! absorbed by augmenting `X` with an unpenalized 1s column — the same
//! scheme the GLM paths already use — and the resulting β̃ is in the
//! scaled space, with original-scale `β_j = β̃_j / s_j`.
//!
//! The wrapper is generic over the base backend (`SparseCSC`,
//! `DenseMatrix`, …) so it composes with any future `DesignMatrix`
//! impl without rewriting the optimization code. It carries a
//! per-column scale vector and forwards every trait method through it:
//!
//! ```text
//!     X̃ = X · diag(1 / s)          (no centering)
//!     col_dot(j, v)    = base.col_dot(j, v) / s_j
//!     col_sq_norm(j)   = base.col_sq_norm(j) / s_j²
//!     matvec(β)        = base.matvec(β / s)
//!     rmatvec(r)       = base.rmatvec(r) / s
//!     columns(cols)    = base.columns(cols) with each column scaled
//! ```
//!
//! Set `s_j = 1` for columns that should not be scaled (e.g. an
//! augmented intercept column).

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

pub struct Standardized<D: DesignMatrix> {
    base: D,
    x_scale: Array1<f64>,
}

impl<D: DesignMatrix> Standardized<D> {
    /// Wrap `base` with per-column scales. Each `x_scale[j]` must be
    /// strictly positive and finite. Use `1.0` for unscaled columns.
    pub fn new(base: D, x_scale: Array1<f64>) -> Self {
        assert_eq!(
            x_scale.len(),
            base.n_features(),
            "x_scale length {} does not match n_features {}",
            x_scale.len(),
            base.n_features()
        );
        for (j, &s) in x_scale.iter().enumerate() {
            assert!(
                s > 0.0 && s.is_finite(),
                "x_scale[{}] = {} must be > 0 and finite",
                j,
                s
            );
        }
        Self { base, x_scale }
    }

    pub fn x_scale(&self) -> ArrayView1<'_, f64> {
        self.x_scale.view()
    }

    pub fn base(&self) -> &D {
        &self.base
    }
}

impl<D: DesignMatrix> DesignMatrix for Standardized<D> {
    fn n_samples(&self) -> usize {
        self.base.n_samples()
    }

    fn n_features(&self) -> usize {
        self.base.n_features()
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        // X̃ β = X · (β / s) — element-wise divide first, then forward.
        let beta_scaled: Array1<f64> = beta
            .iter()
            .zip(self.x_scale.iter())
            .map(|(&b, &s)| b / s)
            .collect();
        self.base.matvec(beta_scaled.view())
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        // X̃ᵀ r = (Xᵀ r) / s.
        let mut g = self.base.rmatvec(r);
        for (gj, &sj) in g.iter_mut().zip(self.x_scale.iter()) {
            *gj /= sj;
        }
        g
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        self.base.col_dot(j, v) / self.x_scale[j]
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        let s = self.x_scale[j];
        self.base.col_sq_norm(j) / (s * s)
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let mut out = self.base.columns(cols);
        let n = out.nrows();
        for (k, &j) in cols.iter().enumerate() {
            let s = self.x_scale[j];
            for i in 0..n {
                out[[i, k]] /= s;
            }
        }
        out
    }

    fn col_axpy(&self, j: usize, alpha: f64, r: ArrayViewMut1<f64>) {
        // X̃[:, j] = X[:, j] / s_j ⇒ r += (alpha / s_j) · X[:, j].
        self.base.col_axpy(j, alpha / self.x_scale[j], r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::{DenseMatrix, SparseCSC};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    /// 4×3 dense reference + matching sparse + matching scales. Asserts
    /// every trait method on `Standardized<base>` matches the dense
    /// reference of `X · diag(1/s)`.
    fn build_problem() -> (
        DenseMatrix, // raw dense
        SparseCSC,   // raw sparse (same X)
        DenseMatrix, // pre-scaled dense reference
        Array1<f64>, // x_scale
    ) {
        // Same hand-built sparse pattern as the SparseCSC tests.
        let x_dense = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 4.0, -2.0]
        ];
        let scales = array![2.0, 4.0, 1.0]; // arbitrary positive scales
        let mut x_scaled_ref = x_dense.clone();
        for j in 0..3 {
            for i in 0..4 {
                x_scaled_ref[[i, j]] /= scales[j];
            }
        }

        let data = array![1.0, -1.0, 3.0, 4.0, 2.0, -2.0];
        let indices = array![0_usize, 2, 1, 3, 0, 3];
        let indptr = array![0_usize, 2, 4, 6];
        let sparse = SparseCSC::new(4, data, indices, indptr);

        (
            DenseMatrix::new(x_dense),
            sparse,
            DenseMatrix::new(x_scaled_ref),
            scales,
        )
    }

    #[test]
    fn standardized_matvec_matches_pre_scaled_reference() {
        let (dense, sparse, ref_scaled, scales) = build_problem();
        let beta = array![0.5, -1.0, 2.0];

        let std_dense = Standardized::new(dense, scales.clone());
        let std_sparse = Standardized::new(sparse, scales);

        let r_ref = ref_scaled.matvec(beta.view());
        let r_std_dense = std_dense.matvec(beta.view());
        let r_std_sparse = std_sparse.matvec(beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(r_std_dense[i], r_ref[i], epsilon = 1e-12);
            assert_abs_diff_eq!(r_std_sparse[i], r_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn standardized_rmatvec_matches_reference() {
        let (dense, sparse, ref_scaled, scales) = build_problem();
        let r = array![1.0, -0.5, 2.0, 0.3];

        let std_dense = Standardized::new(dense, scales.clone());
        let std_sparse = Standardized::new(sparse, scales);

        let g_ref = ref_scaled.rmatvec(r.view());
        let g_std_dense = std_dense.rmatvec(r.view());
        let g_std_sparse = std_sparse.rmatvec(r.view());
        for j in 0..3 {
            assert_abs_diff_eq!(g_std_dense[j], g_ref[j], epsilon = 1e-12);
            assert_abs_diff_eq!(g_std_sparse[j], g_ref[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn standardized_col_dot_and_col_sq_norm_match_reference() {
        let (dense, sparse, ref_scaled, scales) = build_problem();
        let v = array![0.7, -1.2, 3.0, 0.1];

        let std_dense = Standardized::new(dense, scales.clone());
        let std_sparse = Standardized::new(sparse, scales);

        for j in 0..3 {
            assert_abs_diff_eq!(
                std_dense.col_dot(j, v.view()),
                ref_scaled.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                std_sparse.col_dot(j, v.view()),
                ref_scaled.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                std_dense.col_sq_norm(j),
                ref_scaled.col_sq_norm(j),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                std_sparse.col_sq_norm(j),
                ref_scaled.col_sq_norm(j),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn standardized_columns_block_matches_reference() {
        let (dense, sparse, ref_scaled, scales) = build_problem();
        let std_dense = Standardized::new(dense, scales.clone());
        let std_sparse = Standardized::new(sparse, scales);
        let cols = [0_usize, 2];

        let block_ref = ref_scaled.columns(&cols);
        let block_dense = std_dense.columns(&cols);
        let block_sparse = std_sparse.columns(&cols);
        assert_eq!(block_ref.shape(), block_dense.shape());
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(block_dense[[i, k]], block_ref[[i, k]], epsilon = 1e-12);
                assert_abs_diff_eq!(block_sparse[[i, k]], block_ref[[i, k]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn standardized_unit_scale_is_identity() {
        let (dense, sparse, _, _) = build_problem();
        let ones = Array1::<f64>::ones(3);
        let std_dense = Standardized::new(dense, ones.clone());
        let std_sparse = Standardized::new(sparse, ones);
        let v = array![1.0, -1.0, 0.5, 2.0];
        let beta = array![0.3, -0.7, 1.1];

        let dense_inner = std_dense.base();
        let sparse_inner = std_sparse.base();
        for j in 0..3 {
            assert_abs_diff_eq!(
                std_dense.col_dot(j, v.view()),
                dense_inner.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                std_sparse.col_dot(j, v.view()),
                sparse_inner.col_dot(j, v.view()),
                epsilon = 1e-12
            );
        }
        let mv = std_dense.matvec(beta.view());
        let mv_ref = dense_inner.matvec(beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(mv[i], mv_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "x_scale length")]
    fn standardized_panics_on_length_mismatch() {
        let (dense, _, _, _) = build_problem();
        let _ = Standardized::new(dense, Array1::ones(5));
    }

    #[test]
    #[should_panic(expected = "must be > 0")]
    fn standardized_panics_on_zero_scale() {
        let (dense, _, _, _) = build_problem();
        let _ = Standardized::new(dense, array![1.0, 0.0, 1.0]);
    }

    /// Solver equivalence: solve LS with MCP on a sparse design under
    /// `Standardized<SparseCSC>` and verify it matches the same solver
    /// run on a pre-scaled `DenseMatrix` reference.
    #[test]
    fn standardized_solver_path_matches_pre_scaled_dense() {
        use crate::datafit::LeastSquares;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};

        // Sparse problem with random pattern.
        let n = 30;
        let p = 5;
        let mut state = 17_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        // Dense X with ~50% sparsity.
        let x = Array2::<f64>::from_shape_fn((n, p), |_| {
            let v = sample();
            if v.abs() < 0.5 {
                0.0
            } else {
                v
            }
        });
        let y = Array1::<f64>::from_shape_fn(n, |_| 0.3 * sample());
        let scales = array![1.5, 2.0, 0.8, 3.0, 1.2];

        // Build pre-scaled dense reference.
        let mut x_scaled_ref = x.clone();
        for j in 0..p {
            for i in 0..n {
                x_scaled_ref[[i, j]] /= scales[j];
            }
        }
        let dense_ref = DenseMatrix::new(x_scaled_ref);

        // Build SparseCSC from x.
        let mut data: Vec<f64> = Vec::new();
        let mut indices: Vec<usize> = Vec::new();
        let mut indptr: Vec<usize> = vec![0];
        for j in 0..p {
            for i in 0..n {
                if x[[i, j]] != 0.0 {
                    data.push(x[[i, j]]);
                    indices.push(i);
                }
            }
            indptr.push(data.len());
        }
        let sparse = SparseCSC::new(
            n,
            Array1::from(data),
            Array1::from(indices),
            Array1::from(indptr),
        );
        let std_sparse = Standardized::new(sparse, scales);

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
        };
        let datafit_a = LeastSquares::new(y.clone());
        let datafit_b = LeastSquares::new(y);
        let make_pen = |lam: f64| -> Box<dyn crate::Penalty> { Box::new(Mcp::new(lam, 1e6, p)) };

        let (betas_ref, _) = solve_path(&dense_ref, &datafit_a, make_pen, &cfg);
        let (betas_lazy, _) = solve_path(&std_sparse, &datafit_b, make_pen, &cfg);
        assert_eq!(betas_ref.shape(), betas_lazy.shape());
        for k in 0..betas_ref.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_ref[[k, j]], betas_lazy[[k, j]], epsilon = 1e-7);
            }
        }
    }
}
