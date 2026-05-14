//! Graphical lasso microbenchmarks: single-population glasso (M11.1)
//! and joint glasso ADMM across populations (M11.2).
//!
//! The single-population glasso is `O(p²)` per outer sweep with a `p`
//! inner-CD solves, so wall-clock should grow ~`p³` with `p`. The joint
//! ADMM is `O(K · p²)` per ADMM iter (eigen-decomp dominates the Θ
//! step). This bench exposes both scalings so future changes to either
//! kernel show up in the right column.
//!
//! Scenarios:
//!   - `glasso/single/p={20,50,100}` — L1 glasso on a synthetic SPD
//!     covariance built from a random `n × p` matrix at moderate
//!     density.
//!   - `glasso/joint/K={2,3}_p=20` — joint glasso ADMM at fixed `p=20`,
//!     varying population count, using GroupLassoFactory.
//!
//! Tolerances are loosened relative to production defaults so the
//! microbench finishes in reasonable wall-clock; criterion gives us
//! relative comparisons within a scenario, not absolute numbers.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, ArrayView2};
use skein_core::{
    penalty::{GroupLassoFactory, LassoFactory},
    solver::{glasso_solve, joint_glasso_solve, CdConfig, GlassoConfig, JointGlassoConfig},
};

fn xorshift_sampler(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
    }
}

fn synthetic_cov(seed: u64, n: usize, p: usize) -> Array2<f64> {
    let mut sample = xorshift_sampler(seed);
    let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
    let mut s = x.t().dot(&x);
    s.mapv_inplace(|v| v / n as f64);
    // Friedman PSD-safety bump on the diagonal — λ-equivalent.
    for k in 0..p {
        s[[k, k]] += 0.05;
    }
    s
}

fn bench_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("glasso_single");
    let n = 200;
    let lambda = 0.1;
    let factory = LassoFactory { lambda };
    let cfg = GlassoConfig {
        max_outer_iter: 30,
        outer_tol: 1e-3,
        diag_offset: lambda,
        inner: CdConfig {
            max_iter: 100,
            tol: 1e-5,
            acceleration: None,
        },
        warm_start: None,
    };

    for &p in &[20usize, 50, 100] {
        let s = synthetic_cov(17, n, p);
        group.bench_with_input(BenchmarkId::new("p", p), &p, |b, _| {
            b.iter(|| glasso_solve(s.view(), None, &factory, &cfg));
        });
    }
    group.finish();
}

fn bench_joint(c: &mut Criterion) {
    let mut group = c.benchmark_group("glasso_joint");
    let n = 200;
    let p = 20;
    let lambda = 0.1;
    let factory = GroupLassoFactory { lambda };
    let cfg = JointGlassoConfig {
        max_iter: 40,
        primal_tol: 1e-3,
        dual_tol: 1e-3,
        rho: 1.0,
        diag_offset: 0.0,
    };

    for &k_pops in &[2usize, 3] {
        // Distinct seeds → distinct populations sharing the same `p`.
        let covs_owned: Vec<Array2<f64>> = (0..k_pops)
            .map(|k| synthetic_cov(101 + k as u64, n, p))
            .collect();
        let n_samples: Vec<f64> = vec![n as f64; k_pops];
        group.bench_with_input(BenchmarkId::new("K", k_pops), &k_pops, |bencher, _| {
            bencher.iter(|| {
                let views: Vec<ArrayView2<f64>> = covs_owned.iter().map(|s| s.view()).collect();
                joint_glasso_solve(&views, &n_samples, None, &factory, &cfg)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_single, bench_joint);
criterion_main!(benches);
