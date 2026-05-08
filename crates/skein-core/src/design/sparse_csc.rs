//! Compressed sparse column (CSC) `f64` design matrix.
//!
//! Layout matches scipy.sparse.csc_matrix:
//!
//! ```text
//!     data    : length nnz, the non-zero values
//!     indices : length nnz, row index of each non-zero
//!     indptr  : length n_features + 1, column pointers
//! ```
//!
//! For column `j`, `data[indptr[j]..indptr[j+1]]` are the non-zeros and
//! `indices[indptr[j]..indptr[j+1]]` are their row indices. Indices
//! within a column do not need to be sorted (the inner loops over
//! columns don't depend on order); duplicate row indices in the same
//! column are summed implicitly (`matvec`/`col_dot` add their products).
//!
//! `col_sq_norms` is precomputed once at construction so coordinate
//! descent's per-iteration Lipschitz lookup stays O(1).

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1};

pub struct SparseCSC {
    n_samples: usize,
    n_features: usize,
    data: Array1<f64>,
    indices: Array1<usize>,
    indptr: Array1<usize>,
    col_sq_norms: Array1<f64>,
}

impl SparseCSC {
    /// Construct from raw CSC arrays. Validates structural invariants:
    /// `indptr` length, monotonicity, terminal value matches `data` /
    /// `indices` length, and every row index `< n_samples`. Panics on
    /// violation — this is a typed-data invariant, not user input
    /// validation (PyO3 layer raises a clean Python error first).
    pub fn new(
        n_samples: usize,
        data: Array1<f64>,
        indices: Array1<usize>,
        indptr: Array1<usize>,
    ) -> Self {
        let nnz = data.len();
        assert_eq!(
            indices.len(),
            nnz,
            "indices length {} must equal data length {}",
            indices.len(),
            nnz
        );
        assert!(!indptr.is_empty(), "indptr must have length ≥ 1");
        let n_features = indptr.len() - 1;
        assert_eq!(indptr[0], 0, "indptr[0] must be 0 (got {})", indptr[0]);
        assert_eq!(
            indptr[n_features], nnz,
            "indptr[{}] must equal nnz={} (got {})",
            n_features, nnz, indptr[n_features]
        );
        for j in 0..n_features {
            assert!(
                indptr[j] <= indptr[j + 1],
                "indptr must be non-decreasing (column {} violates)",
                j
            );
        }
        for k in 0..nnz {
            assert!(
                indices[k] < n_samples,
                "row index {} at nnz={} exceeds n_samples={}",
                indices[k],
                k,
                n_samples
            );
        }

        // Precompute ‖X[:, j]‖² per column.
        let mut col_sq_norms = Array1::<f64>::zeros(n_features);
        for j in 0..n_features {
            let start = indptr[j];
            let end = indptr[j + 1];
            let mut s = 0.0_f64;
            for k in start..end {
                let v = data[k];
                s += v * v;
            }
            col_sq_norms[j] = s;
        }

        Self {
            n_samples,
            n_features,
            data,
            indices,
            indptr,
            col_sq_norms,
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> ArrayView1<'_, f64> {
        self.data.view()
    }

    pub fn indices(&self) -> ArrayView1<'_, usize> {
        self.indices.view()
    }

    pub fn indptr(&self) -> ArrayView1<'_, usize> {
        self.indptr.view()
    }
}

impl DesignMatrix for SparseCSC {
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.n_samples);
        for j in 0..self.n_features {
            let bj = beta[j];
            if bj == 0.0 {
                continue;
            }
            let start = self.indptr[j];
            let end = self.indptr[j + 1];
            for k in start..end {
                out[self.indices[k]] += self.data[k] * bj;
            }
        }
        out
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.n_features);
        for j in 0..self.n_features {
            let start = self.indptr[j];
            let end = self.indptr[j + 1];
            let mut s = 0.0_f64;
            for k in start..end {
                s += self.data[k] * r[self.indices[k]];
            }
            out[j] = s;
        }
        out
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        let start = self.indptr[j];
        let end = self.indptr[j + 1];
        let mut s = 0.0_f64;
        for k in start..end {
            s += self.data[k] * v[self.indices[k]];
        }
        s
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        self.col_sq_norms[j]
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        // Densify the requested columns into a (n, |cols|) array. Group
        // block-CD operates on dense column blocks; sparse-block CD is
        // a follow-up optimization.
        let n = self.n_samples;
        let mut out = Array2::<f64>::zeros((n, cols.len()));
        for (slot, &j) in cols.iter().enumerate() {
            let start = self.indptr[j];
            let end = self.indptr[j + 1];
            for k in start..end {
                out[[self.indices[k], slot]] = self.data[k];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    /// Build a (4, 3) example matrix with hand-known sparsity:
    ///
    /// ```text
    ///     [ 1.0   0    2.0]
    ///     [  0   3.0    0 ]
    ///     [-1.0   0     0 ]
    ///     [  0   4.0  -2.0]
    /// ```
    ///
    /// Column 0: rows {0, 2} → data [1.0, -1.0]
    /// Column 1: rows {1, 3} → data [3.0, 4.0]
    /// Column 2: rows {0, 3} → data [2.0, -2.0]
    fn known_sparse() -> (SparseCSC, Array2<f64>) {
        let data = array![1.0, -1.0, 3.0, 4.0, 2.0, -2.0];
        let indices = array![0_usize, 2, 1, 3, 0, 3];
        let indptr = array![0_usize, 2, 4, 6];
        let sparse = SparseCSC::new(4, data, indices, indptr);
        let dense = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 4.0, -2.0]
        ];
        (sparse, dense)
    }

    #[test]
    fn sparse_csc_shape_matches_inputs() {
        let (sparse, _) = known_sparse();
        assert_eq!(sparse.n_samples(), 4);
        assert_eq!(sparse.n_features(), 3);
        assert_eq!(sparse.nnz(), 6);
    }

    #[test]
    fn sparse_csc_matvec_matches_dense() {
        let (sparse, dense_arr) = known_sparse();
        let dense = DenseMatrix::new(dense_arr);
        let beta = array![0.5, -1.0, 2.0];
        let r_sparse = sparse.matvec(beta.view());
        let r_dense = dense.matvec(beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(r_sparse[i], r_dense[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn sparse_csc_rmatvec_matches_dense() {
        let (sparse, dense_arr) = known_sparse();
        let dense = DenseMatrix::new(dense_arr);
        let r = array![1.0, -0.5, 2.0, 0.3];
        let g_sparse = sparse.rmatvec(r.view());
        let g_dense = dense.rmatvec(r.view());
        for j in 0..3 {
            assert_abs_diff_eq!(g_sparse[j], g_dense[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn sparse_csc_col_dot_matches_dense() {
        let (sparse, dense_arr) = known_sparse();
        let dense = DenseMatrix::new(dense_arr);
        let v = array![0.7, -1.2, 3.0, 0.1];
        for j in 0..3 {
            assert_abs_diff_eq!(
                sparse.col_dot(j, v.view()),
                dense.col_dot(j, v.view()),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn sparse_csc_col_sq_norms_match_dense() {
        let (sparse, dense_arr) = known_sparse();
        let dense = DenseMatrix::new(dense_arr);
        for j in 0..3 {
            assert_abs_diff_eq!(sparse.col_sq_norm(j), dense.col_sq_norm(j), epsilon = 1e-12);
        }
    }

    #[test]
    fn sparse_csc_columns_block_matches_dense() {
        let (sparse, dense_arr) = known_sparse();
        let dense = DenseMatrix::new(dense_arr);
        let cols = [0_usize, 2];
        let block_sparse = sparse.columns(&cols);
        let block_dense = dense.columns(&cols);
        assert_eq!(block_sparse.shape(), &[4, 2]);
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(block_sparse[[i, k]], block_dense[[i, k]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn sparse_csc_matvec_skips_zero_beta_entries() {
        // Beta with only one nonzero entry — verify only the
        // corresponding column's nonzeros contribute.
        let (sparse, _) = known_sparse();
        let beta = array![0.0, 2.0, 0.0]; // only column 1 active
        let r = sparse.matvec(beta.view());
        // Column 1 has 3.0 at row 1 and 4.0 at row 3 → r = [0, 6, 0, 8].
        assert_abs_diff_eq!(r[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[3], 8.0, epsilon = 1e-12);
    }

    #[test]
    fn sparse_csc_empty_column_returns_zeros() {
        // Build a (3, 2) matrix where column 0 is all zeros.
        let data = array![5.0, -1.0];
        let indices = array![0_usize, 2];
        let indptr = array![0_usize, 0, 2]; // col 0: empty; col 1: rows 0, 2
        let sparse = SparseCSC::new(3, data, indices, indptr);
        assert_eq!(sparse.col_sq_norm(0), 0.0);
        let v = array![1.0, 1.0, 1.0];
        assert_eq!(sparse.col_dot(0, v.view()), 0.0);
    }

    #[test]
    #[should_panic(expected = "indptr[0] must be 0")]
    fn sparse_csc_panics_on_bad_indptr_first() {
        let _ = SparseCSC::new(3, array![1.0], array![0_usize], array![1_usize, 1]);
    }

    #[test]
    #[should_panic(expected = "must equal nnz")]
    fn sparse_csc_panics_on_indptr_terminal_mismatch() {
        let _ = SparseCSC::new(
            3,
            array![1.0, 2.0],
            array![0_usize, 1],
            array![0_usize, 1, 1], // claims 1 nnz total but data has 2
        );
    }

    #[test]
    #[should_panic(expected = "exceeds n_samples")]
    fn sparse_csc_panics_on_out_of_range_row_index() {
        let _ = SparseCSC::new(
            3,
            array![1.0],
            array![5_usize], // row 5 not valid for n_samples=3
            array![0_usize, 1],
        );
    }
}
