//! Row-block-chunked wrapper around a list of [`DesignMatrix`] chunks.
//!
//! For problems where `n` is too large to fit in a single mmap'd file
//! (or where chunking the data on disk is more convenient — say, one
//! file per shard from an upstream pipeline), `Chunked<C>` lets the
//! solver treat a list of equal-`n_features` chunks as a single
//! design matrix.
//!
//! Each chunk holds a contiguous row block; the wrapper records each
//! chunk's row offset so it can route hot-path calls:
//!
//! - `col_dot(j, v)` splits `v` into per-chunk segments and sums
//!   `chunk.col_dot(j, v_chunk)` across chunks.
//! - `matvec(β)` concatenates each chunk's `matvec`.
//! - `rmatvec(r)` slices `r` and sums each chunk's `rmatvec`.
//! - `col_sq_norm(j)` is precomputed once at construction by summing
//!   across chunks.
//! - `columns(cols)` stacks each chunk's column block vertically.
//!
//! Generic over the chunk backend `C: DesignMatrix`, so:
//! - `Chunked<MmapMatrix>` = chunked f64 mmap (the headline use case).
//! - `Chunked<MmapMatrixF32>` = chunked f32 mmap.
//! - `Chunked<DenseMatrix>` = chunked in-memory (mostly for tests).
//!
//! v1 is serial; chunks are an obvious axis for `rayon::par_iter`
//! parallelism, but the gain only shows up at scale where mmap I/O
//! is the bottleneck. Adding it is a one-liner once benches justify.
//! Composes with [`Augmented`](super::Augmented) and
//! [`Standardized`](super::Standardized) the same way every other
//! backend does.

use super::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1};

pub struct Chunked<C: DesignMatrix> {
    chunks: Vec<C>,
    /// Cumulative row offsets: `row_offsets[k]` is the first row of
    /// chunk `k` in the flat coordinate system. Length `chunks.len() + 1`,
    /// with `row_offsets[0] = 0` and `row_offsets[chunks.len()] =
    /// total n_samples`.
    row_offsets: Vec<usize>,
    n_features: usize,
    col_sq_norms: Array1<f64>,
}

impl<C: DesignMatrix> Chunked<C> {
    /// Build a chunked design from a non-empty list of chunks. All
    /// chunks must report the same `n_features`. `col_sq_norms` is
    /// computed once at construction by summing each chunk's cached
    /// column norms — the page-cache cost has already been paid by
    /// the chunk constructors (e.g. `MmapMatrix::open`).
    pub fn new(chunks: Vec<C>) -> Self {
        assert!(!chunks.is_empty(), "Chunked: at least one chunk required");
        let n_features = chunks[0].n_features();
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(
                c.n_features(),
                n_features,
                "Chunked: chunk {i} has n_features={}, expected {n_features}",
                c.n_features(),
            );
        }
        let mut row_offsets = Vec::with_capacity(chunks.len() + 1);
        let mut acc = 0_usize;
        row_offsets.push(0);
        for c in &chunks {
            acc += c.n_samples();
            row_offsets.push(acc);
        }
        let mut col_sq_norms = Array1::<f64>::zeros(n_features);
        for c in &chunks {
            for j in 0..n_features {
                col_sq_norms[j] += c.col_sq_norm(j);
            }
        }
        Self {
            chunks,
            row_offsets,
            n_features,
            col_sq_norms,
        }
    }

    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }
}

impl<C: DesignMatrix> DesignMatrix for Chunked<C> {
    fn n_samples(&self) -> usize {
        *self.row_offsets.last().unwrap()
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn matvec(&self, beta: ArrayView1<f64>) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.n_features);
        let n = self.n_samples();
        let mut out = Array1::<f64>::zeros(n);
        for (k, c) in self.chunks.iter().enumerate() {
            let lo = self.row_offsets[k];
            let hi = self.row_offsets[k + 1];
            let r = c.matvec(beta);
            // Copy chunk output into the corresponding flat slice.
            for (i, &v) in (lo..hi).zip(r.iter()) {
                out[i] = v;
            }
        }
        out
    }

    fn rmatvec(&self, r: ArrayView1<f64>) -> Array1<f64> {
        debug_assert_eq!(r.len(), self.n_samples());
        let mut out = Array1::<f64>::zeros(self.n_features);
        for (k, c) in self.chunks.iter().enumerate() {
            let lo = self.row_offsets[k];
            let hi = self.row_offsets[k + 1];
            let r_chunk = r.slice(ndarray::s![lo..hi]);
            let g = c.rmatvec(r_chunk);
            for j in 0..self.n_features {
                out[j] += g[j];
            }
        }
        out
    }

    fn col_dot(&self, j: usize, v: ArrayView1<f64>) -> f64 {
        debug_assert_eq!(v.len(), self.n_samples());
        let mut acc = 0.0_f64;
        for (k, c) in self.chunks.iter().enumerate() {
            let lo = self.row_offsets[k];
            let hi = self.row_offsets[k + 1];
            let v_chunk = v.slice(ndarray::s![lo..hi]);
            acc += c.col_dot(j, v_chunk);
        }
        acc
    }

    fn col_sq_norm(&self, j: usize) -> f64 {
        self.col_sq_norms[j]
    }

    fn columns(&self, cols: &[usize]) -> Array2<f64> {
        let n = self.n_samples();
        let mut out = Array2::<f64>::zeros((n, cols.len()));
        for (k, c) in self.chunks.iter().enumerate() {
            let lo = self.row_offsets[k];
            let block = c.columns(cols);
            for (di, src_i) in (0..block.nrows()).enumerate() {
                let dst_i = lo + di;
                for c_idx in 0..cols.len() {
                    out[[dst_i, c_idx]] = block[[src_i, c_idx]];
                }
            }
        }
        out
    }

    fn col_axpy(&self, j: usize, alpha: f64, mut r: ndarray::ArrayViewMut1<f64>) {
        for (k, c) in self.chunks.iter().enumerate() {
            let lo = self.row_offsets[k];
            let hi = self.row_offsets[k + 1];
            c.col_axpy(j, alpha, r.slice_mut(ndarray::s![lo..hi]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::DenseMatrix;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    /// Build a flat (n=4, p=3) DenseMatrix and the same matrix split
    /// into two row-blocks of shapes (2, 3) and (2, 3) — the chunked
    /// wrapper must produce identical trait-method outputs for both.
    fn build_problem() -> (DenseMatrix, Chunked<DenseMatrix>) {
        let x = array![
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [-1.0, 0.0, 0.5],
            [0.0, 4.0, -2.0],
        ];
        let chunk_a = x.slice(ndarray::s![..2, ..]).to_owned();
        let chunk_b = x.slice(ndarray::s![2.., ..]).to_owned();
        let chunked = Chunked::new(vec![DenseMatrix::new(chunk_a), DenseMatrix::new(chunk_b)]);
        (DenseMatrix::new(x), chunked)
    }

    #[test]
    fn chunked_dimensions_match_flat() {
        let (flat, chunked) = build_problem();
        assert_eq!(chunked.n_samples(), flat.n_samples());
        assert_eq!(chunked.n_features(), flat.n_features());
        assert_eq!(chunked.n_chunks(), 2);
    }

    #[test]
    fn chunked_matvec_matches_flat_reference() {
        let (flat, chunked) = build_problem();
        let beta = array![0.5, -1.0, 2.0];
        let r_flat = flat.matvec(beta.view());
        let r_chunked = chunked.matvec(beta.view());
        for i in 0..4 {
            assert_abs_diff_eq!(r_chunked[i], r_flat[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn chunked_rmatvec_matches_flat_reference() {
        let (flat, chunked) = build_problem();
        let r = array![1.0, -0.5, 2.0, 0.3];
        let g_flat = flat.rmatvec(r.view());
        let g_chunked = chunked.rmatvec(r.view());
        for j in 0..3 {
            assert_abs_diff_eq!(g_chunked[j], g_flat[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn chunked_col_dot_and_sq_norm_match_flat() {
        let (flat, chunked) = build_problem();
        let v = array![0.7, -1.2, 3.0, 0.1];
        for j in 0..3 {
            assert_abs_diff_eq!(
                chunked.col_dot(j, v.view()),
                flat.col_dot(j, v.view()),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(chunked.col_sq_norm(j), flat.col_sq_norm(j), epsilon = 1e-12);
        }
    }

    #[test]
    fn chunked_columns_block_matches_flat() {
        let (flat, chunked) = build_problem();
        let cols = [0_usize, 2];
        let blk_flat = flat.columns(&cols);
        let blk_chunked = chunked.columns(&cols);
        assert_eq!(blk_flat.shape(), blk_chunked.shape());
        for i in 0..4 {
            for k in 0..2 {
                assert_abs_diff_eq!(blk_chunked[[i, k]], blk_flat[[i, k]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn chunked_handles_uneven_chunk_sizes() {
        // 5 rows split as 2 + 1 + 2.
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];
        let chunks = vec![
            DenseMatrix::new(x.slice(ndarray::s![..2, ..]).to_owned()),
            DenseMatrix::new(x.slice(ndarray::s![2..3, ..]).to_owned()),
            DenseMatrix::new(x.slice(ndarray::s![3.., ..]).to_owned()),
        ];
        let chunked = Chunked::new(chunks);
        let flat = DenseMatrix::new(x);
        let beta = array![1.5, -0.3];
        let r_flat = flat.matvec(beta.view());
        let r_chunked = chunked.matvec(beta.view());
        for i in 0..5 {
            assert_abs_diff_eq!(r_chunked[i], r_flat[i], epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "n_features=2, expected 3")]
    fn chunked_panics_on_inconsistent_n_features() {
        let a = DenseMatrix::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b = DenseMatrix::new(array![[1.0, 2.0], [4.0, 5.0]]);
        let _ = Chunked::new(vec![a, b]);
    }

    /// Solver equivalence: solve_path on `Chunked<DenseMatrix>` with
    /// 3 row-blocks matches the same solver on the flat
    /// `DenseMatrix`. Validates the trait routing — same algorithm
    /// path through the same hot path callers, just with the
    /// per-chunk slicing interposed.
    #[test]
    fn chunked_solver_path_matches_flat() {
        use crate::datafit::LeastSquares;
        use crate::penalty::Mcp;
        use crate::solver::{solve_path, CdConfig, PathConfig, Screening};
        use ndarray::Array1;

        let n = 36;
        let p = 5;
        let mut state = 42_u64;
        let mut sample = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };
        let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
        let y = Array1::<f64>::from_shape_fn(n, |_| 0.3 * sample());

        let flat = DenseMatrix::new(x.clone());
        // Three uneven chunks: 12 + 8 + 16 = 36.
        let chunks = vec![
            DenseMatrix::new(x.slice(ndarray::s![..12, ..]).to_owned()),
            DenseMatrix::new(x.slice(ndarray::s![12..20, ..]).to_owned()),
            DenseMatrix::new(x.slice(ndarray::s![20.., ..]).to_owned()),
        ];
        let chunked = Chunked::new(chunks);

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
        let (b_flat, _) = solve_path(&flat, &datafit_a, make_pen, &cfg);
        let (b_chunk, _) = solve_path(&chunked, &datafit_b, make_pen, &cfg);
        assert_eq!(b_flat.shape(), b_chunk.shape());
        for k in 0..b_flat.nrows() {
            for j in 0..p {
                assert_abs_diff_eq!(b_flat[[k, j]], b_chunk[[k, j]], epsilon = 1e-7);
            }
        }
    }
}
