//! Profiling target for M10.3 perf work — runs the same medium lasso/LS
//! scenario the Python bench harness uses, but as a pure-Rust binary so
//! profilers (samply, cargo-flamegraph, Instruments) can resolve frames
//! without the PyO3 / interpreter layer in the way.
//!
//! Build: `cargo build --release --example lasso_ls_medium`
//! Profile: `samply record ./target/release/examples/lasso_ls_medium`
//!
//! The synthetic problem matches `benches/problems.py::gaussian_lasso`
//! at the medium size (n=10k, p=1k, k_active=10, snr=5, seed=1) — but
//! reproduces the data generator in Rust so the run is self-contained
//! and does not depend on numpy/scipy randomness.

use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    penalty::ElasticNet,
    solver::{lambda_max, solve_path, CdConfig, PathConfig, Screening},
    Penalty,
};
use std::time::Instant;

const N: usize = 10_000;
const P: usize = 1_000;
const N_LAMBDAS: usize = 100;
const TOL: f64 = 1e-6;

/// xorshift64 — the same deterministic PRNG used elsewhere in skein-core
/// tests, so the binary doesn't pull in `rand`.
struct Xorshift {
    state: u64,
}

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Box–Muller standard normal.
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn make_problem() -> (DenseMatrix, Array1<f64>) {
    let mut rng = Xorshift::new(1);

    let mut x = Array2::<f64>::zeros((N, P));
    for v in x.iter_mut() {
        *v = rng.normal();
    }

    // 10 active features in random positions with sign × U(0.5, 2.0).
    let mut beta = Array1::<f64>::zeros(P);
    let active: Vec<usize> = {
        let mut idx: Vec<usize> = (0..P).collect();
        // Fisher–Yates shuffle.
        for i in (1..P).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        idx[..10].to_vec()
    };
    for &j in &active {
        let sign = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        let mag = 0.5 + rng.next_f64() * 1.5;
        beta[j] = sign * mag;
    }

    let signal = x.dot(&beta);
    let signal_std = {
        let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        (signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / signal.len() as f64).sqrt()
    };
    let noise_scale = signal_std / 5.0; // SNR = 5

    let mut y = signal.clone();
    for v in y.iter_mut() {
        *v += noise_scale * rng.normal();
    }

    (DenseMatrix::new(x), y)
}

fn main() {
    println!("building problem (n={N}, p={P}, k_active=10, snr=5)…");
    let t0 = Instant::now();
    let (design, y) = make_problem();
    let t_build = t0.elapsed();
    println!("  built in {:?}", t_build);

    let datafit = LeastSquares::new(y);

    // Compute lambda_max from the cold-start KKT (same as the Python bench).
    let weights = Array1::<f64>::ones(P);
    let lam_max = lambda_max(&design, &datafit, weights.view());

    let config = PathConfig {
        n_lambdas: N_LAMBDAS,
        lambda_min_ratio: 1e-3,
        lambdas: None,
        cd: CdConfig {
            max_iter: 100,
            tol: TOL,
            acceleration: Some(5),
        },
        screening: Screening::Strong,
        p0: 10,
    };

    let make_pen = |lam: f64| -> Box<dyn Penalty> {
        Box::new(ElasticNet::with_weights(lam, 1.0, weights.clone()))
    };

    // Warm-up to get caches hot and JIT-equivalent codegen quirks settled.
    println!("warm-up fit…");
    let t0 = Instant::now();
    let (_, _) = solve_path(&design, &datafit, make_pen, &config);
    println!("  warm-up in {:?} (lam_max={:.6})", t0.elapsed(), lam_max);

    // Measured fit: this is what the profiler should focus on.
    println!("measured fit…");
    let t0 = Instant::now();
    let (betas, report) = solve_path(&design, &datafit, make_pen, &config);
    let elapsed = t0.elapsed();

    let final_active = betas
        .row(N_LAMBDAS - 1)
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    println!(
        "  fit in {:?}; final-λ active set = {} / {}",
        elapsed, final_active, P
    );
    println!(
        "  iter sum = {}, kkt_passes total = {}",
        report.iters.iter().sum::<usize>(),
        report.kkt_passes.iter().sum::<usize>()
    );
}
