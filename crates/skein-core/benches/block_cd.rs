//! Microbenchmarks for the group block-CD inner solvers and the path
//! solver's screening modes. Run with `cargo bench -p skein-core`.
//!
//! Scenarios:
//!   - `serial_vs_parallel/blocks={n_groups}` — Jacobi parallel sweep
//!     speedup over serial Gauss-Seidel on uncorrelated random groups.
//!   - `screening/{off,strong,gap_safe}` — full path solve under each
//!     screening strategy; useful for both wall-clock and (eventually)
//!     working-set-size comparisons.
//!
//! These are *micro* benchmarks: small p, moderate n, contiguous groups
//! of fixed size. The scaling story (sparse X, n_groups in the thousands)
//! belongs in a separate suite once `SparseCSC` lands in M4.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    groups::Groups,
    penalty::GroupLasso,
    solver::{
        block_cd_solve_subset, block_cd_solve_subset_parallel, solve_block_path,
        BlockPathConfig, CdConfig, Screening,
    },
};

fn deterministic_problem(seed: u64, n: usize, p: usize, group_size: usize) -> (DenseMatrix, Array1<f64>, Groups) {
    let mut state = seed.max(1);
    let mut sample = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
    };
    let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
    let mut true_beta = Array1::<f64>::zeros(p);
    // First two groups active.
    for j in 0..(2 * group_size).min(p) {
        true_beta[j] = if j % 2 == 0 { 1.5 } else { -1.0 };
    }
    let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
    let y = x.dot(&true_beta) + &noise;
    let groups = Groups::contiguous_blocks(p, group_size);
    (DenseMatrix::new(x), y, groups)
}

fn bench_serial_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("serial_vs_parallel");
    for &n_groups in &[8usize, 32, 128] {
        let group_size = 4;
        let p = n_groups * group_size;
        let n = 200;
        let (design, y, groups) = deterministic_problem(42, n, p, group_size);
        let datafit = LeastSquares::new(y);
        let lambda = 0.01;
        let pen = GroupLasso::new(lambda, n_groups);
        let cfg = CdConfig {
            max_iter: 500,
            tol: 1e-8,
            acceleration: None,
        };
        let subset: Vec<usize> = (0..n_groups).collect();

        group.bench_with_input(
            BenchmarkId::new("serial", n_groups),
            &n_groups,
            |b, _| {
                b.iter(|| {
                    let init = Array1::<f64>::zeros(p);
                    block_cd_solve_subset(init, &subset, &design, &datafit, &pen, &groups, &cfg)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parallel", n_groups),
            &n_groups,
            |b, _| {
                b.iter(|| {
                    let init = Array1::<f64>::zeros(p);
                    block_cd_solve_subset_parallel(
                        init, &subset, &design, &datafit, &pen, &groups, &cfg,
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_screening_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("screening_modes");
    let n_groups = 64;
    let group_size = 4;
    let p = n_groups * group_size;
    let n = 200;
    let (design, y, groups_struct) = deterministic_problem(13, n, p, group_size);
    let datafit = LeastSquares::new(y);

    for &mode in &[Screening::Off, Screening::Strong, Screening::GapSafe] {
        let label = match mode {
            Screening::Off => "off",
            Screening::Strong => "strong",
            Screening::GapSafe => "gap_safe",
        };
        let cfg = BlockPathConfig {
            n_lambdas: 20,
            lambda_min_ratio: 1e-2,
            lambdas: None,
            cd: CdConfig {
                max_iter: 1000,
                tol: 1e-8,
                acceleration: None,
            },
            screening: mode,
            parallel: false,
        };
        let n_groups_owned = n_groups;
        group.bench_function(label, |b| {
            b.iter(|| {
                solve_block_path(
                    &design,
                    &datafit,
                    |lam| Box::new(GroupLasso::new(lam, n_groups_owned)),
                    &groups_struct,
                    &cfg,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_serial_vs_parallel, bench_screening_modes);
criterion_main!(benches);
