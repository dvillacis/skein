//! Microbench attribution for P2 — isolate `penalty.prox_coord` and
//! `penalty.value` for MCP vs Lasso (ElasticNet at α=1) so the per-
//! iter cost ratio in `cd_solve_subset` can be attributed to each
//! virtual call rather than to design-side BLAS work.
//!
//! Build + run: `cargo run --release --example mcp_vs_lasso_micro`
//!
//! Numbers below from the medium/deep phase-log run (host-dependent):
//!   inner_cd  Lasso = 697 ms (348 iters total across 100 λs)
//!   inner_cd  MCP   = 920 ms (324 iters total)
//!   per-iter ratio  ≈ 1.42×
//! and the candidates this microbench separates are `prox_coord` (per-
//! coord virtual + arithmetic) and `value` (per-iter O(p) scan).

use ndarray::Array1;
use skein_core::{
    penalty::{ElasticNet, Mcp},
    Penalty,
};
use std::hint::black_box;
use std::time::Instant;

const P: usize = 1_000;
const N_PROX_CALLS: usize = 800; // |ws| at the saturated tail
const N_VALUE_CALLS: usize = 4; // ~iters/λ avg
const REPS: usize = 100; // ~n_lambdas

fn main() {
    let weights = Array1::<f64>::ones(P);
    let lasso: Box<dyn Penalty> = Box::new(ElasticNet::with_weights(0.05, 1.0, weights.clone()));
    let mcp: Box<dyn Penalty> = Box::new(Mcp::with_weights(0.05, 3.0, weights.clone()));

    // A `β` that mirrors the deep-tail regime: ~80% saturated, mixed
    // magnitudes around the firm-threshold transition.
    let mut beta = Array1::<f64>::zeros(P);
    for j in 0..P {
        if j % 5 != 0 {
            beta[j] = ((j as f64 * 0.137).sin() * 0.4) + 0.05;
        }
    }

    // Synthesize (z, step) pairs of the same shape the inner CD loop
    // would produce. The exact distribution doesn't matter for cost
    // attribution — only that we hit the same branches.
    let zs: Vec<f64> = (0..N_PROX_CALLS)
        .map(|i| ((i as f64) * 0.027).sin() * 0.5)
        .collect();
    let step = 1.0; // standardized LS

    // --- prox_coord ---
    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..REPS {
        for (j, &z) in zs.iter().enumerate() {
            acc += lasso.prox_coord(j, z, step);
        }
    }
    let lasso_prox_ns = t0.elapsed().as_nanos();
    black_box(acc);

    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..REPS {
        for (j, &z) in zs.iter().enumerate() {
            acc += mcp.prox_coord(j, z, step);
        }
    }
    let mcp_prox_ns = t0.elapsed().as_nanos();
    black_box(acc);

    // --- value(beta) ---
    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..REPS * N_VALUE_CALLS {
        acc += lasso.value(beta.view());
    }
    let lasso_value_ns = t0.elapsed().as_nanos();
    black_box(acc);

    let t0 = Instant::now();
    let mut acc = 0.0_f64;
    for _ in 0..REPS * N_VALUE_CALLS {
        acc += mcp.value(beta.view());
    }
    let mcp_value_ns = t0.elapsed().as_nanos();
    black_box(acc);

    let n_prox = (REPS * N_PROX_CALLS) as f64;
    let n_value = (REPS * N_VALUE_CALLS) as f64;

    println!("prox_coord  ({} calls):", REPS * N_PROX_CALLS);
    println!(
        "  lasso : {:>8.2} ms  ({:>5.1} ns / call)",
        lasso_prox_ns as f64 / 1e6,
        lasso_prox_ns as f64 / n_prox
    );
    println!(
        "  mcp   : {:>8.2} ms  ({:>5.1} ns / call)  ratio = {:.2}×",
        mcp_prox_ns as f64 / 1e6,
        mcp_prox_ns as f64 / n_prox,
        mcp_prox_ns as f64 / lasso_prox_ns as f64,
    );
    println!();
    println!("value(beta) ({} calls × p={}):", REPS * N_VALUE_CALLS, P);
    println!(
        "  lasso : {:>8.2} ms  ({:>5.1} ns / call)",
        lasso_value_ns as f64 / 1e6,
        lasso_value_ns as f64 / n_value
    );
    println!(
        "  mcp   : {:>8.2} ms  ({:>5.1} ns / call)  ratio = {:.2}×",
        mcp_value_ns as f64 / 1e6,
        mcp_value_ns as f64 / n_value,
        mcp_value_ns as f64 / lasso_value_ns as f64,
    );

    // Project to the inner_cd budget at medium/deep. Per-λ averages
    // from the phase log are: Lasso ~3.5 inner iters × ~800 prox calls
    // + ~3.5 value calls.
    let lasso_prox_per_lam_us = (lasso_prox_ns as f64 / n_prox) * (N_PROX_CALLS as f64) * 3.5 / 1e3;
    let mcp_prox_per_lam_us = (mcp_prox_ns as f64 / n_prox) * (N_PROX_CALLS as f64) * 3.5 / 1e3;
    let lasso_value_per_lam_us = (lasso_value_ns as f64 / n_value) * 3.5 / 1e3;
    let mcp_value_per_lam_us = (mcp_value_ns as f64 / n_value) * 3.5 / 1e3;
    println!();
    println!("projected per-λ contribution (3.5 iters × 800 coords):");
    println!(
        "  lasso : prox {:>5.1} µs   value {:>5.1} µs   total {:>5.1} µs",
        lasso_prox_per_lam_us,
        lasso_value_per_lam_us,
        lasso_prox_per_lam_us + lasso_value_per_lam_us
    );
    println!(
        "  mcp   : prox {:>5.1} µs   value {:>5.1} µs   total {:>5.1} µs",
        mcp_prox_per_lam_us,
        mcp_value_per_lam_us,
        mcp_prox_per_lam_us + mcp_value_per_lam_us
    );
    println!(
        "  Δ (mcp − lasso) per λ ≈ {:.1} µs prox + {:.1} µs value",
        mcp_prox_per_lam_us - lasso_prox_per_lam_us,
        mcp_value_per_lam_us - lasso_value_per_lam_us
    );
    println!(
        "  observed inner_cd gap @ medium/deep ≈ {} µs / λ",
        (920_000 - 697_000) / 100
    );
}
