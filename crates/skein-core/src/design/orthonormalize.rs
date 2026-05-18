//! Per-block (group) orthonormalization for grouped covariates
//! (Breheny–Huang trick from `grpreg`).
//!
//! Given a design matrix `X` partitioned by `Groups`, this module
//! produces an orthonormalized version `X̃` such that
//! `X̃_g^T X̃_g / n = I_{|g|}` for every group `g`. Block-CD and LLA
//! solvers operating on `X̃` see a per-group Lipschitz of exactly 1.0
//! and a closed-form block soft-threshold prox — the same regime that
//! grpreg's `gdfit_*` C kernels target.
//!
//! The transformation is invertible: caller solves the path on `X̃`,
//! then applies [`BlockBackTransform::apply_to_coefs`] (or its `_path`
//! sibling) to map coefficients back to original-feature scale.
//!
//! Implementation uses Cholesky decomposition on each per-group Gram
//! matrix `G_g = X_g^T X_g`. For a positive-definite `G_g = L_g L_g^T`,
//! the orthonormalizing transform is `T_g = sqrt(n) · L_g^{-T}`, and
//! `X̃_g = X_g · T_g` satisfies the unit Gram condition exactly. Blocks
//! that fail Cholesky (rank-deficient — typically a duplicate or
//! perfectly collinear column within the group) yield a clear
//! [`SkeinError::InvalidParameter`]; the caller must drop the dependent
//! column first. grpreg handles this via SVD-with-pivot — that
//! refinement is left as future work, since dense in-practice groups
//! are full-rank.

use crate::groups::Groups;
use crate::{Result, SkeinError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Per-group back-transform `T_g`. Owned by the caller after
/// [`orthonormalize_groups_dense`]; supports applying to a single β
/// vector or to an entire path.
#[derive(Debug, Clone)]
pub struct BlockBackTransform {
    /// One entry per group: `(cols_in_original_x, T_g matrix of size |g|×|g|)`.
    blocks: Vec<(Vec<usize>, Array2<f64>)>,
    /// Total number of features in the original (and orthonormalized) design.
    /// Same in both spaces — we don't drop rank-deficient columns in this v0.
    n_features: usize,
}

impl BlockBackTransform {
    /// Number of features (columns) in the design — matches the original.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Map a coefficient vector from orthonormalized space back to
    /// original-feature space: `β_orig = block_diag(T_g) · β_orth`.
    pub fn apply_to_coefs(&self, beta_orth: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(
            beta_orth.len(),
            self.n_features,
            "beta_orth length {} does not match n_features {}",
            beta_orth.len(),
            self.n_features
        );
        let mut beta_orig = Array1::<f64>::zeros(self.n_features);
        for (cols, t) in &self.blocks {
            // Gather β_orth restricted to this group's columns.
            let g_size = cols.len();
            let mut b_block = Array1::<f64>::zeros(g_size);
            for (k, &j) in cols.iter().enumerate() {
                b_block[k] = beta_orth[j];
            }
            // β_orig[cols] = T_g · b_block.
            let b_out = t.dot(&b_block);
            for (k, &j) in cols.iter().enumerate() {
                beta_orig[j] = b_out[k];
            }
        }
        beta_orig
    }

    /// Vectorized back-transform across an entire λ-path. Input shape
    /// `(n_lambdas, n_features)`; output same shape, each row mapped
    /// independently.
    pub fn apply_to_coefs_path(&self, betas_orth: ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(
            betas_orth.ncols(),
            self.n_features,
            "betas_orth ncols {} does not match n_features {}",
            betas_orth.ncols(),
            self.n_features
        );
        let n_lambdas = betas_orth.nrows();
        let mut betas_orig = Array2::<f64>::zeros((n_lambdas, self.n_features));
        for k in 0..n_lambdas {
            let row = self.apply_to_coefs(betas_orth.row(k));
            betas_orig.row_mut(k).assign(&row);
        }
        betas_orig
    }

    /// Read-only access to a single block's transform — useful for tests
    /// and downstream tooling that needs to inspect the orthonormalization.
    pub fn block(&self, g: usize) -> (&[usize], ArrayView2<'_, f64>) {
        let (cols, t) = &self.blocks[g];
        (cols, t.view())
    }
}

/// In-place Cholesky factorization `A = L · L^T` for a small symmetric
/// positive-definite matrix. Overwrites the lower triangle of `A` with
/// `L`. Returns `Err` if `A` is not numerically PD.
fn cholesky_in_place(a: &mut Array2<f64>, tol: f64) -> Result<()> {
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), n, "matrix must be square");
    for j in 0..n {
        let mut sum = a[[j, j]];
        for k in 0..j {
            sum -= a[[j, k]] * a[[j, k]];
        }
        if sum <= tol {
            return Err(SkeinError::InvalidParameter(format!(
                "Cholesky failed: diagonal entry {} <= {} (group is rank-deficient or \
                 numerically singular). Drop dependent columns within the group before \
                 orthonormalizing.",
                sum, tol
            )));
        }
        let ljj = sum.sqrt();
        a[[j, j]] = ljj;
        for i in (j + 1)..n {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= a[[i, k]] * a[[j, k]];
            }
            a[[i, j]] = sum / ljj;
        }
    }
    Ok(())
}

/// Solve `L · X = B` for `X`, where `L` is lower-triangular (stored in
/// the lower triangle of `l`). Overwrites `b`.
fn trsm_lower_in_place(l: &Array2<f64>, b: &mut Array2<f64>) {
    let n = l.nrows();
    let nrhs = b.ncols();
    debug_assert_eq!(b.nrows(), n);
    for k in 0..nrhs {
        for i in 0..n {
            let mut sum = b[[i, k]];
            for j in 0..i {
                sum -= l[[i, j]] * b[[j, k]];
            }
            b[[i, k]] = sum / l[[i, i]];
        }
    }
}

/// Orthonormalize a dense design matrix block-by-block according to
/// the supplied [`Groups`] partition. Returns
/// `(X_orth, BlockBackTransform)` where:
///
/// * `X_orth` has the same shape as `x` and satisfies
///   `X_orth_g^T X_orth_g / n = I_{|g|}` for every group `g`.
/// * `BlockBackTransform` carries the per-group `T_g = sqrt(n) · L_g^{-T}`
///   matrices needed to map coefficients fitted in orthonormalized space
///   back to original-feature scale.
///
/// Errors if any group's Gram matrix `X_g^T X_g` is not numerically
/// positive-definite (i.e. has perfectly collinear columns).
pub fn orthonormalize_groups_dense(
    x: ArrayView2<f64>,
    groups: &Groups,
) -> Result<(Array2<f64>, BlockBackTransform)> {
    let n = x.nrows();
    let p = x.ncols();
    let n_groups = groups.n_groups();

    let mut x_orth = Array2::<f64>::zeros((n, p));
    let mut blocks: Vec<(Vec<usize>, Array2<f64>)> = Vec::with_capacity(n_groups);
    let sqrt_n = (n as f64).sqrt();
    let cholesky_tol = 1e-10;

    for g in 0..n_groups {
        let cols: Vec<usize> = groups.group(g).to_vec();
        let g_size = cols.len();

        // Per-group Gram: G = X_g^T X_g  (size g_size × g_size).
        let mut gram = Array2::<f64>::zeros((g_size, g_size));
        for a in 0..g_size {
            let ca = cols[a];
            for b in 0..=a {
                let cb = cols[b];
                let mut s = 0.0;
                for i in 0..n {
                    s += x[[i, ca]] * x[[i, cb]];
                }
                gram[[a, b]] = s;
                gram[[b, a]] = s;
            }
        }

        // Cholesky: G = L L^T (lower triangle in-place).
        cholesky_in_place(&mut gram, cholesky_tol).map_err(|e| match e {
            SkeinError::InvalidParameter(msg) => {
                SkeinError::InvalidParameter(format!("group {}: {}", g, msg))
            }
            other => other,
        })?;

        // T = sqrt(n) · L^{-T} = sqrt(n) · (L^T)^{-1}.
        // Solve L^T · T = sqrt(n) · I. Equivalently, solve L · M = sqrt(n) · I
        // for M = (L^T · T) = L^{-1} · sqrt(n) · I, then T = M' with rows/cols
        // appropriately transposed. Simpler: build T column by column via
        // back-substitution on the upper triangle L^T.
        let mut t = Array2::<f64>::eye(g_size) * sqrt_n;
        // Backward-solve L^T · t_col = rhs for each column. L^T is upper-triangular
        // with diagonal entries L[i,i] (= L^T[i,i]).
        for col in 0..g_size {
            for i in (0..g_size).rev() {
                let mut sum = t[[i, col]];
                for k in (i + 1)..g_size {
                    sum -= gram[[k, i]] * t[[k, col]]; // L^T[i,k] = L[k,i]
                }
                t[[i, col]] = sum / gram[[i, i]];
            }
        }

        // Materialize the orthonormalized block: X_orth[:, cols] = X[:, cols] · T.
        let mut x_block = Array2::<f64>::zeros((n, g_size));
        for (k, &j) in cols.iter().enumerate() {
            for i in 0..n {
                x_block[[i, k]] = x[[i, j]];
            }
        }
        let x_orth_block = x_block.dot(&t);
        for (k, &j) in cols.iter().enumerate() {
            for i in 0..n {
                x_orth[[i, j]] = x_orth_block[[i, k]];
            }
        }

        blocks.push((cols, t));
    }

    Ok((
        x_orth,
        BlockBackTransform {
            blocks,
            n_features: p,
        },
    ))
}

// Silence the dead-code warning on the triangular solver: it's a
// generally useful primitive but the current `orthonormalize_groups_dense`
// uses an inline back-substitution loop instead. Kept for future use
// (e.g. multi-RHS back-transforms over many columns).
#[allow(dead_code)]
fn _trsm_used_externally(l: &Array2<f64>, b: &mut Array2<f64>) {
    trsm_lower_in_place(l, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    /// Construct an `n × p` matrix with random columns; columns in group g
    /// are independent so each block is full-rank.
    fn random_x(n: usize, p: usize, seed: u64) -> Array2<f64> {
        let mut state = seed.max(1);
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        Array2::<f64>::from_shape_fn((n, p), |_| sample())
    }

    #[test]
    fn cholesky_factorizes_simple_pd_matrix() {
        // A = [[4, 12], [12, 37]] → L = [[2, 0], [6, 1]].
        let mut a = ndarray::array![[4.0, 12.0], [12.0, 37.0]];
        cholesky_in_place(&mut a, 1e-10).unwrap();
        assert_abs_diff_eq!(a[[0, 0]], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[[1, 0]], 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a[[1, 1]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cholesky_rejects_singular_matrix() {
        // A = [[1, 1], [1, 1]] is singular.
        let mut a = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
        let result = cholesky_in_place(&mut a, 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn orthonormalize_produces_identity_gram_per_block() {
        let n = 80;
        let p = 6;
        let x = random_x(n, p, 7);
        let groups = Groups::contiguous_blocks(p, 3); // two groups of 3
        let (x_orth, _bt) = orthonormalize_groups_dense(x.view(), &groups).unwrap();
        for g in 0..2 {
            let cols = groups.group(g);
            // Compute (X_orth_g)^T X_orth_g / n.
            for &a in cols {
                for &b in cols {
                    let mut s = 0.0;
                    for i in 0..n {
                        s += x_orth[[i, a]] * x_orth[[i, b]];
                    }
                    let expected = if a == b { 1.0 } else { 0.0 };
                    assert_abs_diff_eq!(s / (n as f64), expected, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn back_transform_recovers_predictions() {
        // Key invariant: X · β_orig ≡ X_orth · β_orth where
        // β_orig = T · β_orth. Tests the round-trip on a random vector.
        let n = 50;
        let p = 4;
        let x = random_x(n, p, 11);
        let groups = Groups::contiguous_blocks(p, 2);
        let (x_orth, bt) = orthonormalize_groups_dense(x.view(), &groups).unwrap();

        let beta_orth = ndarray::array![0.5, -1.0, 0.3, 1.2];
        let beta_orig = bt.apply_to_coefs(beta_orth.view());

        let pred_via_orth = x_orth.dot(&beta_orth);
        let pred_via_orig = x.dot(&beta_orig);
        for i in 0..n {
            assert_abs_diff_eq!(pred_via_orth[i], pred_via_orig[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn back_transform_path_matches_per_lambda() {
        let n = 40;
        let p = 4;
        let x = random_x(n, p, 13);
        let groups = Groups::contiguous_blocks(p, 2);
        let (_, bt) = orthonormalize_groups_dense(x.view(), &groups).unwrap();

        let betas_orth = ndarray::array![
            [0.1, -0.2, 0.3, 0.4],
            [0.0, 0.0, 0.5, -0.5],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let betas_orig = bt.apply_to_coefs_path(betas_orth.view());
        for k in 0..3 {
            let expected = bt.apply_to_coefs(betas_orth.row(k));
            for j in 0..p {
                assert_abs_diff_eq!(betas_orig[[k, j]], expected[j], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn singleton_groups_orthonormalize_to_unit_norm() {
        let n = 60;
        let p = 3;
        let x = random_x(n, p, 17);
        let groups = Groups::singletons(p);
        let (x_orth, _bt) = orthonormalize_groups_dense(x.view(), &groups).unwrap();
        // Each column should have squared norm exactly n (so /n = 1.0).
        for j in 0..p {
            let s: f64 = (0..n).map(|i| x_orth[[i, j]] * x_orth[[i, j]]).sum();
            assert_abs_diff_eq!(s / (n as f64), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn orthonormalize_errors_on_collinear_columns() {
        // Build a 4x2 matrix where col1 = col0 (perfect collinearity).
        let x = ndarray::array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [0.5, 0.5]];
        let groups = Groups::contiguous_blocks(2, 2);
        let result = orthonormalize_groups_dense(x.view(), &groups);
        assert!(result.is_err());
        match result.unwrap_err() {
            SkeinError::InvalidParameter(msg) => {
                assert!(msg.contains("group 0") && msg.contains("rank-deficient"));
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }
}
