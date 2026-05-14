//! LLA outer-loop microbenchmarks for non-convex group penalties.
//!
//! The headline question this bench answers: how aggressively does the
//! γ knob (MCP shape parameter) push up the outer-iteration count?
//! γ → ∞ recovers convex group lasso (one outer iter is enough); small
//! γ ramps up the nonconvexity and forces more LLA passes. This file
//! captures both wall-clock and outer-iter count so a future
//! optimisation (M13.4 Phase 2.3 surrogate-fixed-point short-circuit
//! is one such win) shows up in the right column.
//!
//! Scenarios:
//!   - `lla_outer/gamma={1.5,3.0,10.0}` — single-λ group MCP solve at
//!     fixed (n, p, n_groups), varying γ. Reports the LLA outer-loop
//!     time *with* outer-count visible via `cargo bench -- --verbose`.
//!   - `lla_outer/n_groups={16,64,256}` — group MCP solve at fixed
//!     γ = 3.0, varying group count to expose how outer iters scale
//!     with problem size.
//!
//! Inner CD uses the same `tol = 1e-8` as `block_cd.rs` for
//! comparability. `outer_tol = 1e-6` follows what the path solvers
//! plumb in by default.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    groups::Groups,
    solver::{lla_solve, surrogate_weights_group_mcp, CdConfig},
};

fn deterministic_problem(
    seed: u64,
    n: usize,
    p: usize,
    group_size: usize,
) -> (DenseMatrix, Array1<f64>, Groups) {
    let mut state = seed.max(1);
    let mut sample = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
    };
    let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
    let mut true_beta = Array1::<f64>::zeros(p);
    for j in 0..(2 * group_size).min(p) {
        true_beta[j] = if j % 2 == 0 { 1.5 } else { -1.0 };
    }
    let noise = Array1::<f64>::from_shape_fn(n, |_| 0.05 * sample());
    let y = x.dot(&true_beta) + &noise;
    let groups = Groups::contiguous_blocks(p, group_size);
    (DenseMatrix::new(x), y, groups)
}

fn bench_lla_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("lla_outer_gamma");
    let n_groups = 64;
    let group_size = 4;
    let p = n_groups * group_size;
    let n = 200;
    let lambda = 0.05;
    let (design, y, groups) = deterministic_problem(11, n, p, group_size);
    let datafit = LeastSquares::new(y);
    let cd_cfg = CdConfig {
        max_iter: 1000,
        tol: 1e-8,
        acceleration: None,
    };
    let base_weights = Array1::<f64>::ones(n_groups);

    for &gamma in &[1.5_f64, 3.0, 10.0] {
        let label = format!("{gamma:.1}");
        group.bench_with_input(BenchmarkId::new("gamma", label), &gamma, |b, &gamma| {
            b.iter(|| {
                let init = Array1::<f64>::zeros(p);
                let bw = base_weights.clone();
                lla_solve(
                    &design,
                    &datafit,
                    &groups,
                    init,
                    lambda,
                    |beta, gs| surrogate_weights_group_mcp(beta, gs, lambda, gamma, bw.view()),
                    &cd_cfg,
                    20,
                    1e-6,
                )
            });
        });
    }
    group.finish();
}

fn bench_lla_n_groups(c: &mut Criterion) {
    let mut group = c.benchmark_group("lla_outer_n_groups");
    let group_size = 4;
    let n = 200;
    let lambda = 0.05;
    let gamma = 3.0;
    let cd_cfg = CdConfig {
        max_iter: 1000,
        tol: 1e-8,
        acceleration: None,
    };

    for &n_groups in &[16usize, 64, 256] {
        let p = n_groups * group_size;
        let (design, y, groups) = deterministic_problem(7, n, p, group_size);
        let datafit = LeastSquares::new(y);
        let base_weights = Array1::<f64>::ones(n_groups);
        group.bench_with_input(BenchmarkId::new("n_groups", n_groups), &n_groups, |b, _| {
            b.iter(|| {
                let init = Array1::<f64>::zeros(p);
                let bw = base_weights.clone();
                lla_solve(
                    &design,
                    &datafit,
                    &groups,
                    init,
                    lambda,
                    |beta, gs| surrogate_weights_group_mcp(beta, gs, lambda, gamma, bw.view()),
                    &cd_cfg,
                    20,
                    1e-6,
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lla_gamma, bench_lla_n_groups);
criterion_main!(benches);
