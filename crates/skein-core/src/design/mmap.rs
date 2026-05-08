//! Memory-mapped dense `f64` design matrix.
//!
//! Backed by a column-major (Fortran-order) raw `f64` file on disk.
//! Each column is a contiguous `n_samples × 8`-byte slice, so the CD
//! hot path (`col_dot`) reads sequentially and rides the OS page cache
//! instead of materializing the matrix in RAM. This is the backend
//! that lets `skein` fit problems where `X` is too large to load —
//! genomics, large-scale text, on-disk one-hot, etc.
//!
//! The file is assumed to be a raw little-endian `f64` array of shape
//! `(n_samples, n_features)` written in column-major order (e.g. via
//! `np.asfortranarray(x).astype(np.float64).tofile(path)` from numpy
//! or any `dgemm`-friendly producer). No header is parsed; the caller
//! supplies `(n_samples, n_features)` at open time. A header-aware
//! `.npy` constructor is a follow-up.
//!
//! `f32` and mixed-precision (f32 mmap with f64 active set refinement)
//! are separate M4.x bullets — this v1 is f64-only so the solver-
//! equivalence story against [`DenseMatrix`] is exact.
//!
//! `col_sq_norms` is precomputed once at open time (one full pass over
//! the file, paid by the OS page cache) and stored in RAM. Hot path
//! cost matches [`DenseMatrix`] from then on.

use super::DesignMatrix;
use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayView1};
use std::fs::File;
use std::io;
use std::path::Path;

/// Memory-mapped column-major `f64` matrix. Auto-`Sync + Send` because
/// `Mmap` is.
pub struct MmapMatrix {
    mmap: Mmap,
    n_samples: usize,
    n_features: usize,
    col_sq_norms: Array1<f64>,
}

impl MmapMatrix {
    /// Open `path` as a memory-mapped column-major `f64` matrix of
    /// shape `(n_samples, n_features)`. Validates that the file size
    /// matches `n_samples * n_features * 8` bytes and that the
    /// underlying mapping is `f64`-aligned (page-aligned mappings on
    /// every platform we target satisfy this for free).
    pub fn open(
        path: impl AsRef<Path>,
        n_samples: usize,
        n_features: usize,
    ) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: mmap is unsafe because external mutation of the file
        // (e.g. another process truncating it) breaks Rust's borrow
        // model. We don't promise tolerance to that — the matrix is
        // read-only from the caller's perspective and the file is
        // expected to be stable for the lifetime of the matrix.
        let mmap = unsafe { Mmap::map(&file)? };
        let expected = n_samples
            .checked_mul(n_features)
            .and_then(|nf| nf.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "n_samples * n_features * 8 overflowed",
                )
            })?;
        if mmap.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file size {} bytes does not match expected {} bytes \
                     for n_samples={} n_features={} f64",
                    mmap.len(),
                    expected,
                    n_samples,
                    n_features,
                ),
            ));
        }
        // Validate alignment: f64 needs 8-byte alignment. memmap2 maps
        // at a page boundary on every platform, so this is in practice
        // always satisfied — assert defensively.
        let (head, _, _) = unsafe { mmap.align_to::<f64>() };
        if !head.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "memory mapping is not 8-byte aligned (cannot reinterpret as f64)",
            ));
        }

        let mut out = Self {
            mmap,
            n_samples,
            n_features,
            col_sq_norms: Array1::<f64>::zeros(n_features),
        };
        let mut sq = Array1::<f64>::zeros(n_features);
        for j in 0..n_features {
            let col = out.column_slice(j);
            sq[j] = col.iter().map(|&v| v * v).sum();
        }
        out.col_sq_norms = sq;
        Ok(out)
    }

    /// Return the `j`-th column as a `&[f64]` slice into the mapping.
    /// Column-major layout means this is contiguous in file order.
    fn column_slice(&self, j: usize) -> &[f64] {
        debug_assert!(j < self.n_features);
        let start_bytes = j * self.n_samples * std::mem::size_of::<f64>();
        let end_bytes = start_bytes + self.n_samples * std::mem::size_of::<f64>();
        let bytes = &self.mmap[start_bytes..end_bytes];
        // Safety: we validated 8-byte alignment in `open` and the file
        // size matches `n_samples * n_features * 8` exactly.
        let (head, body, tail) = unsafe { bytes.align_to::<f64>() };
        debug_assert!(head.is_empty() && tail.is_empty());
        debug_assert_eq!(body.len(), self.n_samples);
        body
    }
}

impl DesignMatrix for MmapMatrix {
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.n_features);
        let mut out = Array1::<f64>::zeros(self.n_samples);
        // Skip zero columns: warm-started β has many zeros.
        for j in 0..self.n_features {
            let bj = beta[j];
            if bj == 0.0 {
                continue;
            }
            let col = self.column_slice(j);
            for (oi, &cv) in out.iter_mut().zip(col.iter()) {
                *oi += bj * cv;
            }
        }
        out
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        debug_assert_eq!(r.len(), self.n_samples);
        let mut out = Array1::<f64>::zeros(self.n_features);
        let r_slice = r.as_slice().expect("r must be contiguous");
        for j in 0..self.n_features {
            let col = self.column_slice(j);
            let mut acc = 0.0_f64;
            for (cv, rv) in col.iter().zip(r_slice.iter()) {
                acc += cv * rv;
            }
            out[j] = acc;
        }
        out
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        debug_assert_eq!(v.len(), self.n_samples);
        let col = self.column_slice(j);
        let v_slice = v.as_slice().expect("v must be contiguous");
        let mut acc = 0.0_f64;
        for (cv, vv) in col.iter().zip(v_slice.iter()) {
            acc += cv * vv;
        }
        acc
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        self.col_sq_norms[j]
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let n = self.n_samples;
        let mut out = Array2::<f64>::zeros((n, cols.len()));
        for (k, &j) in cols.iter().enumerate() {
            let col = self.column_slice(j);
            for (i, &v) in col.iter().enumerate() {
                out[[i, k]] = v;
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
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a (n, p) f64 array to disk in column-major order and
    /// return the path-holding tempfile alongside the in-memory
    /// reference. Caller keeps the tempfile alive for the lifetime of
    /// any `MmapMatrix::open` against it.
    fn write_fortran_f64(x: &Array2<f64>) -> (NamedTempFile, DenseMatrix) {
        let n = x.nrows();
        let p = x.ncols();
        let mut bytes = Vec::with_capacity(n * p * std::mem::size_of::<f64>());
        for j in 0..p {
            for i in 0..n {
                bytes.extend_from_slice(&x[[i, j]].to_le_bytes());
            }
        }
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(&bytes).expect("write");
        f.flush().expect("flush");
        (f, DenseMatrix::new(x.clone()))
    }

    #[test]
    fn mmap_matvec_matches_dense_reference() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f64(&x);
        let mmap = MmapMatrix::open(f.path(), 4, 3).expect("open");
        let beta = array![0.5, -1.0, 2.0];
        let r_ref = dense.matvec(beta.view());
        let r_mmap = mmap.matvec(beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(r_mmap[i], r_ref[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn mmap_rmatvec_matches_dense_reference() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f64(&x);
        let mmap = MmapMatrix::open(f.path(), 4, 3).expect("open");
        let r = array![1.0, -0.5, 2.0, 0.3];
        let g_ref = dense.rmatvec(r.view());
        let g_mmap = mmap.rmatvec(r.view());
        for j in 0..3 {
            assert_abs_diff_eq!(g_mmap[j], g_ref[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn mmap_col_dot_and_col_sq_norm_match_dense() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f64(&x);
        let mmap = MmapMatrix::open(f.path(), 4, 3).expect("open");
        let v = array![0.7, -1.2, 3.0, 0.1];
        for j in 0..3 {
            assert_abs_diff_eq!(
                mmap.col_dot(j, v.view()),
                dense.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(
                mmap.col_sq_norm(j),
                dense.col_sq_norm(j),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn mmap_columns_block_matches_dense() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f64(&x);
        let mmap = MmapMatrix::open(f.path(), 4, 3).expect("open");
        let cols = [0_usize, 2];
        let blk_ref = dense.columns(&cols);
        let blk_mmap = mmap.columns(&cols);
        assert_eq!(blk_ref.shape(), blk_mmap.shape());
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(blk_mmap[[i, k]], blk_ref[[i, k]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn mmap_open_rejects_wrong_file_size() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let (f, _) = write_fortran_f64(&x);
        // File has 4 f64s = 32 bytes; we claim it's 2×3 = 48 bytes.
        let result = MmapMatrix::open(f.path(), 2, 3);
        assert!(result.is_err());
    }

    /// Solver-equivalence test: solve_path on `MmapMatrix` matches the
    /// in-memory `DenseMatrix` reference at every λ. This is the
    /// load-bearing assertion — the trait abstraction means the same
    /// solver works for both, and the two backends must agree to
    /// machine precision.
    #[test]
    fn mmap_solver_path_matches_dense_reference() {
        use crate::datafit::LeastSquares;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};
        use ndarray::Array1;

        // Random-ish problem with ~50% sparsity in X.
        let n = 40;
        let p = 6;
        let mut state = 17_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| {
            let v = sample();
            if v.abs() < 0.5 { 0.0 } else { v }
        });
        let y = Array1::<f64>::from_shape_fn(n, |_| 0.3 * sample());
        let (f, dense) = write_fortran_f64(&x);
        let mmap = MmapMatrix::open(f.path(), n, p).expect("open");

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
        let (betas_dense, _) = solve_path(&dense, &datafit_a, make_pen, &cfg);
        let (betas_mmap, _) = solve_path(&mmap, &datafit_b, make_pen, &cfg);
        assert_eq!(betas_dense.shape(), betas_mmap.shape());
        for k in 0..betas_dense.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(
                    betas_dense[[k, j]],
                    betas_mmap[[k, j]],
                    epsilon = 1e-7
                );
            }
        }
    }
}
