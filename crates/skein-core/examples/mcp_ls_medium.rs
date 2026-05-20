//! Profiling target for P2 — same medium/LS scenario as
//! `lasso_ls_medium`, but with the MCP penalty so we can attribute
//! the MCP-vs-Lasso wall-clock gap to a specific phase via
//! `SKEIN_PROFILE_PATH=1`. P2's open question is whether the ~1.21×
//! MCP-specific excess (on top of the EN-vs-Lasso 1.24× generic
//! non-trivial-prox cost) lives in the prox call, the KKT-pass
//! re-evaluation, the per-λ weight construction, or somewhere else.
//!
//! Build: `cargo build --release --example mcp_ls_medium`
//! Profile (phase breakdown): `SKEIN_PROFILE_PATH=1 ./target/release/examples/mcp_ls_medium`
//! Profile (sampler):         `samply record ./target/release/examples/mcp_ls_medium`
//!
//! Two cells are run — `deep` (lambda_min_ratio = 1e-3, active set
//! saturates at the tail) and `sparse` (5e-2, stays at true support).
//! These match the v2 bench regime keys; the gap is on `deep`.

use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    penalty::{ElasticNet, Mcp},
    solver::{lambda_max, solve_path, CdConfig, PathConfig, Screening},
    Penalty,
};
use std::time::Instant;

const N: usize = 10_000;
const P: usize = 1_000;
const N_LAMBDAS: usize = 100;
const TOL: f64 = 1e-6;
const MCP_GAMMA: f64 = 3.0;

/// xorshift64 — same deterministic PRNG as `lasso_ls_medium`. Keeping
/// the data generator identical means the profiling deltas attribute
/// to the solver, not to a different design / response.
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

fn make_problem() -> (DenseMatrix, Array1<f64>) {
    let mut rng = Xorshift::new(1);

    let mut x = Array2::<f64>::zeros((N, P));
    for v in x.iter_mut() {
        *v = rng.normal();
    }

    let mut beta = Array1::<f64>::zeros(P);
    let active: Vec<usize> = {
        let mut idx: Vec<usize> = (0..P).collect();
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
    let noise_scale = signal_std / 5.0;

    let mut y = signal.clone();
    for v in y.iter_mut() {
        *v += noise_scale * rng.normal();
    }

    (DenseMatrix::new(x), y)
}

#[derive(Clone, Copy)]
enum Pen {
    Lasso,
    Mcp,
}

fn run_cell(
    name: &str,
    pen_kind: Pen,
    lambda_min_ratio: f64,
    design: &DenseMatrix,
    y: &Array1<f64>,
) {
    let datafit = LeastSquares::new(y.clone());
    let weights = Array1::<f64>::ones(P);
    let lam_max = lambda_max(design, &datafit, weights.view());

    let config = PathConfig {
        n_lambdas: N_LAMBDAS,
        lambda_min_ratio,
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
        match pen_kind {
            Pen::Lasso => Box::new(ElasticNet::with_weights(lam, 1.0, weights.clone())),
            Pen::Mcp => Box::new(Mcp::with_weights(lam, MCP_GAMMA, weights.clone())),
        }
    };

    eprintln!(
        "--- cell: {} (lambda_min_ratio={:.0e}) ---",
        name, lambda_min_ratio
    );
    // Warm-up — caches hot, codegen settled. Closure is non-`Copy`
    // (captures `weights: Array1`), so we pass by reference and reuse
    // for the measured fit.
    #[allow(clippy::needless_borrows_for_generic_args)]
    let _ = solve_path(design, &datafit, &make_pen, &config);

    let t0 = Instant::now();
    #[allow(clippy::needless_borrows_for_generic_args)]
    let (betas, report) = solve_path(design, &datafit, &make_pen, &config);
    let elapsed = t0.elapsed();

    let final_active = betas
        .row(N_LAMBDAS - 1)
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    let ws_sum: usize = report.working_set_sizes.iter().sum();
    let ws_max: usize = *report.working_set_sizes.iter().max().unwrap_or(&0);
    let ws_avg = ws_sum as f64 / report.working_set_sizes.len() as f64;
    // Avg WS × iters at that λ — proxy for "total coord visits" that the
    // inner-CD wall scales with. We don't have per-λ iter*ws, so settle
    // for sum_k(ws_k) × mean(iters_k); accurate enough to compare regimes.
    let iter_sum: usize = report.iters.iter().sum();
    eprintln!(
        "  fit in {:?}; final-λ active = {}/{}; iter sum = {}, kkt total = {}",
        elapsed,
        final_active,
        P,
        iter_sum,
        report.kkt_passes.iter().sum::<usize>(),
    );
    eprintln!(
        "  ws (per-λ): avg = {:.1}  max = {}  sum = {}  (lam_max={:.6})",
        ws_avg, ws_max, ws_sum, lam_max,
    );
    // Per-λ ws × iter product = better proxy for inner-CD coord work.
    let mut work_proxy: u64 = 0;
    for (ws, it) in report.working_set_sizes.iter().zip(report.iters.iter()) {
        work_proxy += (*ws as u64) * (*it as u64);
    }
    eprintln!("  inner-CD coord-work proxy (Σ ws × iter) = {}", work_proxy);
}

fn main() {
    eprintln!("building problem (n={N}, p={P}, k_active=10, snr=5)…");
    let t0 = Instant::now();
    let (design, y) = make_problem();
    eprintln!("  built in {:?}", t0.elapsed());

    // Order: lasso/deep, mcp/deep, lasso/sparse, mcp/sparse. The
    // phase log prints once per `solve_path` invocation, so each
    // cell's breakdown ends up in sequence on stderr.
    run_cell("lasso/deep", Pen::Lasso, 1e-3, &design, &y);
    run_cell("mcp/deep", Pen::Mcp, 1e-3, &design, &y);
    run_cell("lasso/sparse", Pen::Lasso, 5e-2, &design, &y);
    run_cell("mcp/sparse", Pen::Mcp, 5e-2, &design, &y);
}
