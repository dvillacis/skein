//! Virtual intercept-column wrapper around any [`DesignMatrix`].
//!
//! Adds a single all-ones column at index `n_features` (the augmented
//! intercept slot) without touching the underlying storage. The dense
//! and sparse paths handle the intercept by physically appending a 1s
//! column to `X`; for mmap that would mean rewriting the file. This
//! wrapper is the same trick at the trait level — generic over the
//! base backend, O(1) construction, O(n) per intercept-column op.
//!
//! Composes with [`Standardized`](super::Standardized) (intercept is
//! never scaled, so user code wraps the augmented design in a
//! `Standardized` with `s_p = 1.0`).

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1};

pub struct Augmented<D: DesignMatrix> {
    base: D,
}

impl<D: DesignMatrix> Augmented<D> {
    pub fn new(base: D) -> Self {
        Self { base }
    }

    pub fn base(&self) -> &D {
        &self.base
    }
}

impl<D: DesignMatrix> DesignMatrix for Augmented<D> {
    fn n_samples(&self) -> usize {
        self.base.n_samples()
    }

    fn n_features(&self) -> usize {
        self.base.n_features() + 1
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        let p = self.base.n_features();
        debug_assert_eq!(beta.len(), p + 1);
        let mut out = self.base.matvec(beta.slice(ndarray::s![..p]));
        let intercept = beta[p];
        if intercept != 0.0 {
            for v in out.iter_mut() {
                *v += intercept;
            }
        }
        out
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        let p = self.base.n_features();
        let g_base = self.base.rmatvec(r);
        let mut out = Array1::<f64>::zeros(p + 1);
        for j in 0..p {
            out[j] = g_base[j];
        }
        out[p] = r.iter().sum();
        out
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        let p = self.base.n_features();
        if j < p {
            self.base.col_dot(j, v)
        } else {
            // ⟨1, v⟩ = Σ v_i.
            v.iter().sum()
        }
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        let p = self.base.n_features();
        if j < p {
            self.base.col_sq_norm(j)
        } else {
            self.base.n_samples() as f64
        }
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let p = self.base.n_features();
        let n = self.base.n_samples();
        // Two-pass: pull non-intercept cols from the base, fill the
        // intercept slot with 1.0. We could also build a small
        // permutation, but a copy is cheap relative to the file I/O
        // saved by mmap.
        let base_cols: Vec<usize> = cols.iter().copied().filter(|&j| j < p).collect();
        let base_block = if base_cols.is_empty() {
            Array2::<f64>::zeros((n, 0))
        } else {
            self.base.columns(&base_cols)
        };
        let mut out = Array2::<f64>::zeros((n, cols.len()));
        let mut next_base = 0_usize;
        for (k, &j) in cols.iter().enumerate() {
            if j < p {
                out.column_mut(k).assign(&base_block.column(next_base));
                next_base += 1;
            } else {
                out.column_mut(k).fill(1.0);
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
    use ndarray::array;

    /// Reference: a `DenseMatrix` with the 1s column physically
    /// appended. `Augmented<DenseMatrix>` must agree on every method.
    fn build_problem() -> (DenseMatrix, DenseMatrix) {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let mut x_aug = ndarray::Array2::<f64>::zeros((4, 4));
        x_aug.slice_mut(ndarray::s![.., ..3]).assign(&x);
        x_aug.column_mut(3).fill(1.0);
        (DenseMatrix::new(x), DenseMatrix::new(x_aug))
    }

    #[test]
    fn augmented_matvec_matches_physically_augmented_reference() {
        let (base, ref_aug) = build_problem();
        let aug = Augmented::new(base);
        let beta = array![0.5, -1.0, 2.0, 0.7];
        let r_ref = ref_aug.matvec(beta.view());
        let r_aug = aug.matvec(beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(r_aug[i], r_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn augmented_rmatvec_matches_reference() {
        let (base, ref_aug) = build_problem();
        let aug = Augmented::new(base);
        let r = array![1.0, -0.5, 2.0, 0.3];
        let g_ref = ref_aug.rmatvec(r.view());
        let g_aug = aug.rmatvec(r.view());
        for j in 0..4 {
            assert_abs_diff_eq!(g_aug[j], g_ref[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn augmented_col_dot_and_sq_norm_match_reference() {
        let (base, ref_aug) = build_problem();
        let aug = Augmented::new(base);
        let v = array![0.7, -1.2, 3.0, 0.1];
        for j in 0..4 {
            assert_abs_diff_eq!(
                aug.col_dot(j, v.view()),
                ref_aug.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                aug.col_sq_norm(j),
                ref_aug.col_sq_norm(j),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn augmented_columns_block_handles_mixed_indices() {
        let (base, ref_aug) = build_problem();
        let aug = Augmented::new(base);
        // Mix of base indices and intercept index, in interleaved order.
        let cols = [3_usize, 0, 3, 2];
        let blk_ref = ref_aug.columns(&cols);
        let blk_aug = aug.columns(&cols);
        assert_eq!(blk_ref.shape(), blk_aug.shape());
        for i in 0..4 {
            for k in 0..cols.len() {
                assert_abs_diff_eq!(blk_aug[[i, k]], blk_ref[[i, k]], epsilon = 1e-12);
            }
        }
    }
}
