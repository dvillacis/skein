//! Memory-mapped column-major `f32` design matrix, exposed to the f64
//! solver via on-the-fly conversion.
//!
//! Same shape as [`super::MmapMatrix`] but the file holds `f32` values
//! — half the disk footprint and half the page-cache pressure for the
//! same `(n, p)`. Each `col_dot` / `matvec` / `rmatvec` / `columns`
//! call walks the f32 slice and casts to f64 elementwise; the solver
//! itself stays f64 throughout.
//!
//! This is "f32-on-disk", not "true mixed-precision" — the active set
//! refinement at f64 + bulk path work at f32 is a separate M4.x bullet
//! that requires parameterizing the solver core over `T: Float`. The
//! conversion cost is one f32→f64 cast per element loaded; modern
//! CPUs absorb this without measurably slowing the hot path because
//! disk I/O and the multiply-add already saturate.
//!
//! `col_sq_norms` is stored in f64 and computed once at open time off
//! the f32 file (same one-pass amortization as the f64 backend).

use super::DesignMatrix;
use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayView1};
use std::fs::File;
use std::io;
use std::path::Path;

pub struct MmapMatrixF32 {
    mmap: Mmap,
    n_samples: usize,
    n_features: usize,
    col_sq_norms: Array1<f64>,
}

impl MmapMatrixF32 {
    /// Open `path` as a memory-mapped column-major `f32` matrix of
    /// shape `(n_samples, n_features)`. Validates file size and
    /// 4-byte alignment.
    pub fn open(path: impl AsRef<Path>, n_samples: usize, n_features: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        // Safety: same caveat as MmapMatrix — file must be stable for
        // the lifetime of the matrix.
        let mmap = unsafe { Mmap::map(&file)? };
        let expected = n_samples
            .checked_mul(n_features)
            .and_then(|nf| nf.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "n_samples * n_features * 4 overflowed",
                )
            })?;
        if mmap.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file size {} bytes does not match expected {} bytes \
                     for n_samples={} n_features={} f32",
                    mmap.len(),
                    expected,
                    n_samples,
                    n_features,
                ),
            ));
        }
        let (head, _, _) = unsafe { mmap.align_to::<f32>() };
        if !head.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "memory mapping is not 4-byte aligned (cannot reinterpret as f32)",
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
            sq[j] = col.iter().map(|&v| (v as f64) * (v as f64)).sum();
        }
        out.col_sq_norms = sq;
        Ok(out)
    }

    fn column_slice(&self, j: usize) -> &[f32] {
        debug_assert!(j < self.n_features);
        let start_bytes = j * self.n_samples * std::mem::size_of::<f32>();
        let end_bytes = start_bytes + self.n_samples * std::mem::size_of::<f32>();
        let bytes = &self.mmap[start_bytes..end_bytes];
        // Safety: 4-byte alignment validated in `open`; column offsets
        // are multiples of `n_samples * 4` bytes, so each column slice
        // is also 4-byte aligned.
        let (head, body, tail) = unsafe { bytes.align_to::<f32>() };
        debug_assert!(head.is_empty() && tail.is_empty());
        debug_assert_eq!(body.len(), self.n_samples);
        body
    }
}

impl DesignMatrix for MmapMatrixF32 {
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.n_features);
        let mut out = Array1::<f64>::zeros(self.n_samples);
        for j in 0..self.n_features {
            let bj = beta[j];
            if bj == 0.0 {
                continue;
            }
            let col = self.column_slice(j);
            for (oi, &cv) in out.iter_mut().zip(col.iter()) {
                *oi += bj * (cv as f64);
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
                acc += (*cv as f64) * rv;
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
            acc += (*cv as f64) * vv;
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
                out[[i, k]] = v as f64;
            }
        }
        out
    }

    fn col_axpy(&self, j: usize, alpha: f64, mut r: ndarray::ArrayViewMut1<f64>) {
        // f32-on-disk → must promote per element; no BLAS path here.
        let col = self.column_slice(j);
        for (ri, &cv) in r.iter_mut().zip(col.iter()) {
            *ri += alpha * (cv as f64);
        }
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

    /// Write `x` to disk as a column-major `f32` array and return
    /// alongside the f32-rounded f64 reference. The reference is
    /// **f32-rounded** (i.e. cast to f32 and back) so equivalence
    /// tests don't fail on the truncation error introduced by the
    /// on-disk f32 storage.
    fn write_fortran_f32(x: &Array2<f64>) -> (NamedTempFile, DenseMatrix) {
        let n = x.nrows();
        let p = x.ncols();
        let mut bytes = Vec::with_capacity(n * p * std::mem::size_of::<f32>());
        let mut x_rounded = x.clone();
        for j in 0..p {
            for i in 0..n {
                let v32 = x[[i, j]] as f32;
                bytes.extend_from_slice(&v32.to_le_bytes());
                x_rounded[[i, j]] = v32 as f64;
            }
        }
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(&bytes).expect("write");
        f.flush().expect("flush");
        (f, DenseMatrix::new(x_rounded))
    }

    #[test]
    fn mmap_f32_matvec_matches_f32_rounded_reference() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f32(&x);
        let mmap = MmapMatrixF32::open(f.path(), 4, 3).expect("open");
        let beta = array![0.5, -1.0, 2.0];
        let r_ref = dense.matvec(beta.view());
        let r_mmap = mmap.matvec(beta.view());
        for i in 0..4 {
            // f32 storage with f64 arithmetic: integer-valued entries
            // round exactly; equality is fine.
            assert_abs_diff_eq!(r_mmap[i], r_ref[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn mmap_f32_rmatvec_matches_reference() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f32(&x);
        let mmap = MmapMatrixF32::open(f.path(), 4, 3).expect("open");
        let r = array![1.0, -0.5, 2.0, 0.3];
        let g_ref = dense.rmatvec(r.view());
        let g_mmap = mmap.rmatvec(r.view());
        for j in 0..3 {
            assert_abs_diff_eq!(g_mmap[j], g_ref[j], epsilon = 1e-6);
        }
    }

    #[test]
    fn mmap_f32_col_dot_and_sq_norm_match_reference() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f32(&x);
        let mmap = MmapMatrixF32::open(f.path(), 4, 3).expect("open");
        let v = array![0.7, -1.2, 3.0, 0.1];
        for j in 0..3 {
            assert_abs_diff_eq!(
                mmap.col_dot(j, v.view()),
                dense.col_dot(j, v.view()),
                epsilon = 1e-6
            );
            assert_abs_diff_eq!(mmap.col_sq_norm(j), dense.col_sq_norm(j), epsilon = 1e-6);
        }
    }

    #[test]
    fn mmap_f32_columns_block_matches_reference() {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0]
        ];
        let (f, dense) = write_fortran_f32(&x);
        let mmap = MmapMatrixF32::open(f.path(), 4, 3).expect("open");
        let cols = [0_usize, 2];
        let blk_ref = dense.columns(&cols);
        let blk_mmap = mmap.columns(&cols);
        assert_eq!(blk_ref.shape(), blk_mmap.shape());
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(blk_mmap[[i, k]], blk_ref[[i, k]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn mmap_f32_open_rejects_wrong_file_size() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let (f, _) = write_fortran_f32(&x);
        let result = MmapMatrixF32::open(f.path(), 2, 3);
        assert!(result.is_err());
    }

    /// Solver equivalence: solve_path on `MmapMatrixF32` matches the
    /// **f32-rounded** dense reference at every λ. The f64 dense path
    /// would diverge by ~1e-7 due to the f32 truncation; the f32-
    /// rounded reference makes the comparison about *correctness of
    /// the wrapper*, not about precision.
    #[test]
    fn mmap_f32_solver_path_matches_rounded_dense() {
        use crate::datafit::LeastSquares;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};
        use ndarray::Array1;

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
            if v.abs() < 0.5 {
                0.0
            } else {
                v
            }
        });
        let y = Array1::<f64>::from_shape_fn(n, |_| 0.3 * sample());
        let (f, dense) = write_fortran_f32(&x);
        let mmap = MmapMatrixF32::open(f.path(), n, p).expect("open");

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
        let make_pen = |lam: f64| -> Box<dyn crate::Penalty> { Box::new(Mcp::new(lam, 1e6, p)) };
        let (betas_dense, _) = solve_path(&dense, &datafit_a, make_pen, &cfg);
        let (betas_mmap, _) = solve_path(&mmap, &datafit_b, make_pen, &cfg);
        assert_eq!(betas_dense.shape(), betas_mmap.shape());
        for k in 0..betas_dense.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(betas_dense[[k, j]], betas_mmap[[k, j]], epsilon = 1e-6);
            }
        }
    }
}
