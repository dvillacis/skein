//! M13.6 — scaling-exponent comparison for `solve_path` on Lasso.
//!
//! Runs the same problem generator at the canonical `small`, `medium`,
//! and (when `SKEIN_SCALING_LARGE=1`) `large` sizes from
//! `benches/v2/config.yaml`:
//!   small  : n=1000,   p=200,   k_active=10  (~3 MB design)
//!   medium : n=10000,  p=1000,  k_active=10  (~80 MB design)
//!   large  : n=50000,  p=5000,  k_active=10  (~2 GB design — opt-in)
//! Dense-tail regime (`λ_min/λ_max=1e-3`, 100 λs), `tol=1e-7` matching
//! the v2 default.
//!
//! Prints per-λ phase breakdown for each size and the size-to-size
//! ratio, so we can see how each phase scales independently. Run with
//! `SKEIN_PROFILE_PATH=1` for the breakdown. Warm-up is skipped on
//! large — one fit takes long enough that the cold-cache distortion is
//! amortized, and the second 2 GB allocation isn't worth it.
//!
//! Build / run:
//! ```
//! cargo build --release --example lasso_ls_scaling
//! SKEIN_PROFILE_PATH=1 ./target/release/examples/lasso_ls_scaling
//! SKEIN_PROFILE_PATH=1 SKEIN_SCALING_LARGE=1 ./target/release/examples/lasso_ls_scaling
//! ```

use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    penalty::ElasticNet,
    solver::{lambda_max, solve_path, CdConfig, PathConfig, Screening},
    Penalty,
};
use std::time::Instant;

const N_LAMBDAS: usize = 100;
const TOL: f64 = 1e-7;
const K_ACTIVE: usize = 10;

struct Xorshift {
    state: u64,
}

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_f64(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn make_problem(n: usize, p: usize, seed: u64) -> (DenseMatrix, Array1<f64>) {
    let mut rng = Xorshift::new(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for v in x.iter_mut() {
        *v = rng.normal();
    }
    let mut beta = Array1::<f64>::zeros(p);
    let active: Vec<usize> = {
        let mut idx: Vec<usize> = (0..p).collect();
        for i in (1..p).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        idx[..K_ACTIVE.min(p)].to_vec()
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

fn run(label: &str, n: usize, p: usize, warmup: bool) -> std::time::Duration {
    println!();
    println!("=== {label}: n={n}, p={p}, k_active={K_ACTIVE}, snr=5 ===");
    let (design, y) = make_problem(n, p, 1);
    let datafit = LeastSquares::new(y);
    let weights = Array1::<f64>::ones(p);
    let lam_max = lambda_max(&design, &datafit, weights.view());
    // SKEIN_REGIME=sparse selects the support-recovery regime
    // (λ_min/λ_max = 5e-2, active set stays at true support); default
    // matches benches/v2 `deep` (λ_min/λ_max = 1e-3, dense tail).
    // Added for the P5 saturation-threshold ablation — the threshold
    // affects deep and sparse cells differently, so both regimes need
    // to be measurable from the same example.
    let lambda_min_ratio = if std::env::var("SKEIN_REGIME").as_deref() == Ok("sparse") {
        5e-2
    } else {
        1e-3
    };
    let config = PathConfig {
        n_lambdas: N_LAMBDAS,
        lambda_min_ratio,
        lambdas: None,
        cd: CdConfig {
            max_iter: 200,
            tol: TOL,
            acceleration: Some(5),
        },
        screening: Screening::Strong,
        p0: 10,
    };
    // Warm-up (skip for sizes where the second fit would re-allocate
    // gigabytes — the measured run dominates wall-clock anyway). Build
    // a fresh `make_pen` per call: each takes the closure by value.
    if warmup {
        let weights_warm = weights.clone();
        let make_pen = move |lam: f64| -> Box<dyn Penalty> {
            Box::new(ElasticNet::with_weights(lam, 1.0, weights_warm.clone()))
        };
        let _ = solve_path(&design, &datafit, make_pen, &config);
    }

    // Measured
    let t0 = Instant::now();
    let make_pen = move |lam: f64| -> Box<dyn Penalty> {
        Box::new(ElasticNet::with_weights(lam, 1.0, weights.clone()))
    };
    let (betas, report) = solve_path(&design, &datafit, make_pen, &config);
    let elapsed = t0.elapsed();

    let final_active = betas
        .row(N_LAMBDAS - 1)
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    println!(
        "  fit in {:?} (lam_max={:.6}); final-λ active = {}/{}; iters={}, kkt_passes={}",
        elapsed,
        lam_max,
        final_active,
        p,
        report.iters.iter().sum::<usize>(),
        report.kkt_passes.iter().sum::<usize>()
    );
    elapsed
}

fn main() {
    let run_large = std::env::var("SKEIN_SCALING_LARGE").is_ok();

    let small = run("small", 1000, 200, true);
    let medium = run("medium", 10000, 1000, true);
    let large = if run_large {
        Some(run("large", 50000, 5000, false))
    } else {
        None
    };

    println!();
    println!("=== scaling ===");
    let report = |from_label: &str, to_label: &str, from: f64, to: f64, np_ratio: f64| {
        let ratio = to / from;
        println!("  {from_label} → {to_label}");
        println!("    n×p ratio   : {np_ratio:.1}×");
        println!("    wall ratio  : {ratio:.2}×");
        println!(
            "    factor      : {:.2}×  (wall ratio / np ratio — < 1 means sub-linear in n×p)",
            ratio / np_ratio
        );
    };
    report(
        "small",
        "medium",
        small.as_secs_f64(),
        medium.as_secs_f64(),
        (10000.0 * 1000.0) / (1000.0 * 200.0),
    );
    if let Some(large_dur) = large {
        report(
            "medium",
            "large",
            medium.as_secs_f64(),
            large_dur.as_secs_f64(),
            (50000.0 * 5000.0) / (10000.0 * 1000.0),
        );
        report(
            "small",
            "large",
            small.as_secs_f64(),
            large_dur.as_secs_f64(),
            (50000.0 * 5000.0) / (1000.0 * 200.0),
        );
    } else {
        println!();
        println!("  (set SKEIN_SCALING_LARGE=1 to also run large = n=50k, p=5k)");
    }
}
