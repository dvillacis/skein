//! Proximal-Newton GLM inner-solver microbenchmarks.
//!
//! GLMs (logistic, Poisson, Cox, …) hit the M1 separable-penalty CD via
//! a prox-Newton outer loop that re-linearises the loss at every
//! iterate. The interesting question this bench answers: how much does
//! the outer IRLS layer add on top of a comparable LS lasso, and how
//! does that overhead scale with feature count?
//!
//! Scenarios:
//!   - `prox_newton_glm/logistic/p={64,256}` — single-λ logistic Lasso
//!     on synthetic Bernoulli data.
//!   - `prox_newton_glm/poisson/p={64,256}` — single-λ Poisson Lasso on
//!     synthetic count data with `μ_i = exp(x_iᵀ β_true)`.
//!
//! The dataset generators set up moderate signal-to-noise so the prox-
//! Newton outer loop converges in a handful of iterations (the typical
//! production regime); pathologically saturated regimes are out of
//! scope for a microbench.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use skein_core::{
    datafit::{BinomialLogit, PoissonLog},
    design::DenseMatrix,
    penalty::ElasticNet,
    solver::{prox_newton_solve, CdConfig},
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

fn logistic_problem(seed: u64, n: usize, p: usize) -> (DenseMatrix, Array1<f64>) {
    let mut sample = xorshift_sampler(seed);
    let x = Array2::<f64>::from_shape_fn((n, p), |_| sample());
    let mut true_beta = Array1::<f64>::zeros(p);
    // 4 active coords centred so |Σ x_ij β_j| stays ~O(1) — keeps the
    // dataset out of the saturating tail of the sigmoid.
    for (k, j) in [0usize, 1, 2, 3].iter().enumerate() {
        true_beta[*j] = if k % 2 == 0 { 0.8 } else { -0.6 };
    }
    let eta = x.dot(&true_beta);
    let y = Array1::from_iter(eta.iter().map(|&e| {
        let p = 1.0 / (1.0 + (-e).exp());
        // Deterministic Bernoulli draw via the same xorshift stream
        // (using sample() again would entangle x and y; this bench
        // wants a stable y per (seed, n, p) tuple instead).
        if p > 0.5 {
            1.0
        } else {
            0.0
        }
    }));
    (DenseMatrix::new(x), y)
}

fn poisson_problem(seed: u64, n: usize, p: usize) -> (DenseMatrix, Array1<f64>) {
    let mut sample = xorshift_sampler(seed);
    let x = Array2::<f64>::from_shape_fn((n, p), |_| 0.5 * sample());
    let mut true_beta = Array1::<f64>::zeros(p);
    for (k, j) in [0usize, 1, 2, 3].iter().enumerate() {
        true_beta[*j] = if k % 2 == 0 { 0.4 } else { -0.3 };
    }
    let eta = x.dot(&true_beta);
    let y = Array1::from_iter(eta.iter().map(|&e| e.exp().round()));
    (DenseMatrix::new(x), y)
}

fn bench_logistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("prox_newton_glm_logistic");
    let n = 200;
    let cd_cfg = CdConfig {
        max_iter: 500,
        tol: 1e-7,
        acceleration: None,
    };
    for &p in &[64usize, 256] {
        let (design, y) = logistic_problem(31, n, p);
        let glm = BinomialLogit::new(y);
        // λ small enough to keep ~4 active coords on this signal level.
        let pen = ElasticNet::new(0.02, 1.0, p);
        group.bench_with_input(BenchmarkId::new("p", p), &p, |b, _| {
            b.iter(|| {
                let init = Array1::<f64>::zeros(p);
                prox_newton_solve(&design, &glm, &pen, init, &cd_cfg, 25, 1e-6)
            });
        });
    }
    group.finish();
}

fn bench_poisson(c: &mut Criterion) {
    let mut group = c.benchmark_group("prox_newton_glm_poisson");
    let n = 200;
    let cd_cfg = CdConfig {
        max_iter: 500,
        tol: 1e-7,
        acceleration: None,
    };
    for &p in &[64usize, 256] {
        let (design, y) = poisson_problem(53, n, p);
        let glm = PoissonLog::new(y);
        let pen = ElasticNet::new(0.02, 1.0, p);
        group.bench_with_input(BenchmarkId::new("p", p), &p, |b, _| {
            b.iter(|| {
                let init = Array1::<f64>::zeros(p);
                prox_newton_solve(&design, &glm, &pen, init, &cd_cfg, 25, 1e-6)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_logistic, bench_poisson);
criterion_main!(benches);
