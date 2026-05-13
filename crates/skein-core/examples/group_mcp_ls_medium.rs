//! Profiling target for M13.4 perf work — pure-Rust ls_group_mcp at the
//! medium/dense bench cell (n=10k, p=1k, group_size=5, n_groups=200,
//! n_lambdas=100, lambda_min_ratio=1e-3, tol=1e-7, γ=3.0). Mirrors
//! `benches/problems.py::gaussian_group` so samply / cargo-flamegraph
//! can resolve frames without the PyO3 + interpreter layer.
//!
//! Build:   `cargo build --release --example group_mcp_ls_medium`
//! Profile: `samply record ./target/release/examples/group_mcp_ls_medium`
//!
//! Output: a markdown-flavored per-λ table on stdout — outer_iters,
//! inner_iters (summed across outer), kkt_passes, ws size, wall-ms.
//! Pipe into `docs/perf/m13_4_profile.md`'s data section.

use ndarray::{Array1, Array2, ArrayView1};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    penalty::GroupLasso,
    solver::{
        block_lambda_max, solve_block_path_lla, surrogate_weights_group_mcp, BlockPathConfig,
        CdConfig, Screening,
    },
    GroupPenalty, Groups,
};
use std::time::Instant;

const N: usize = 10_000;
const P: usize = 1_000;
const GROUP_SIZE: usize = 5;
const K_ACTIVE_GROUPS: usize = 5;
const SNR: f64 = 5.0;
const N_LAMBDAS: usize = 100;
const LAMBDA_MIN_RATIO: f64 = 1e-3;
const TOL: f64 = 1e-7;
const GAMMA: f64 = 3.0;
const MAX_OUTER: usize = 10;
const OUTER_TOL: f64 = 1e-6;

/// xorshift64 — matches the other examples; not seeded to match numpy's
/// Generator (the bench problem uses np.random.default_rng(seed)), so the
/// realized active groups and noise differ. The dimensions, density, and
/// SNR match — sufficient for profiling.
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

fn make_problem() -> (DenseMatrix, Array1<f64>, Groups) {
    let mut rng = Xorshift::new(1);

    let mut x = Array2::<f64>::zeros((N, P));
    for v in x.iter_mut() {
        *v = rng.normal();
    }

    let n_groups = P / GROUP_SIZE;
    let groups = Groups::contiguous_blocks(P, GROUP_SIZE);

    let mut beta = Array1::<f64>::zeros(P);
    let active_groups: Vec<usize> = {
        let mut idx: Vec<usize> = (0..n_groups).collect();
        for i in (1..n_groups).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        idx[..K_ACTIVE_GROUPS].to_vec()
    };
    for &g in &active_groups {
        let start = g * GROUP_SIZE;
        for j in 0..GROUP_SIZE {
            beta[start + j] = rng.normal();
        }
    }

    let signal = x.dot(&beta);
    let signal_std = {
        let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        (signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / signal.len() as f64).sqrt()
    };
    let noise_scale = signal_std / SNR;

    let mut y = signal.clone();
    for v in y.iter_mut() {
        *v += noise_scale * rng.normal();
    }

    (DenseMatrix::new(x), y, groups)
}

fn main() {
    println!("# M13.4 profile — ls_group_mcp medium/dense");
    println!();
    println!(
        "Config: n={N}, p={P}, group_size={GROUP_SIZE}, n_groups={}, \
         k_active_groups={K_ACTIVE_GROUPS}, snr={SNR}, n_lambdas={N_LAMBDAS}, \
         lambda_min_ratio={LAMBDA_MIN_RATIO}, tol={TOL}, γ={GAMMA}, \
         max_outer={MAX_OUTER}, outer_tol={OUTER_TOL}",
        P / GROUP_SIZE
    );
    println!();

    let t0 = Instant::now();
    let (design, y, groups) = make_problem();
    let t_build = t0.elapsed();
    println!("problem built in {:?}", t_build);

    let n_groups = groups.n_groups();
    let datafit = LeastSquares::new(y);
    let base = Array1::<f64>::ones(n_groups);

    let lam_max = block_lambda_max(&design, &datafit, base.view(), &groups);
    println!("block_lambda_max = {:.6e}", lam_max);

    let cfg = BlockPathConfig {
        n_lambdas: N_LAMBDAS,
        lambda_min_ratio: LAMBDA_MIN_RATIO,
        lambdas: None,
        cd: CdConfig {
            max_iter: 100,
            tol: TOL,
            acceleration: Some(5),
        },
        screening: Screening::Strong,
        parallel: false,
    };
    let base_ref = base.clone();
    let make_inner = |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_mcp(beta, g, lam, GAMMA, base_ref.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };

    // Warm-up to settle caches / first-touch.
    println!("warm-up fit…");
    let t0 = Instant::now();
    let _ = solve_block_path_lla(
        &design,
        &datafit,
        base.clone(),
        make_inner,
        &groups,
        &cfg,
        MAX_OUTER,
        OUTER_TOL,
    );
    println!("  warm-up in {:?}", t0.elapsed());

    println!("measured fit…");
    let t0 = Instant::now();
    let (betas, report) = solve_block_path_lla(
        &design,
        &datafit,
        base.clone(),
        make_inner,
        &groups,
        &cfg,
        MAX_OUTER,
        OUTER_TOL,
    );
    let elapsed = t0.elapsed();
    let elapsed_s = elapsed.as_secs_f64();

    let final_active = betas
        .row(N_LAMBDAS - 1)
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    let total_inner: usize = report.inner_iters.iter().sum();
    let total_kkt: usize = report.kkt_passes.iter().sum();
    let total_outer: usize = report.outer_iters.iter().sum();
    let total_wall_ns: u64 = report.per_lambda_wall_ns.iter().sum();

    println!();
    println!("## Headline");
    println!();
    println!("| metric | value |");
    println!("|---|---|");
    println!(
        "| total fit (incl. setup) | {:?} ({:.3} s) |",
        elapsed, elapsed_s
    );
    println!(
        "| sum(per_lambda_wall_ns) | {:.3} s |",
        total_wall_ns as f64 / 1e9
    );
    println!("| final-λ active features | {final_active} / {P} |");
    println!("| total outer iters across λ | {total_outer} |");
    println!("| total inner CD iters | {total_inner} |");
    println!("| total KKT passes | {total_kkt} |");
    println!();

    // Per-λ table — show every 10th λ + the last to keep it readable.
    println!("## Per-λ breakdown (every 10th + last)");
    println!();
    println!("| k | λ | outer | inner_sum | kkt | ws | wall_ms |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    let print_idx: Vec<usize> = (0..N_LAMBDAS)
        .step_by(10)
        .chain(std::iter::once(N_LAMBDAS - 1))
        .collect();
    for k in print_idx {
        println!(
            "| {k} | {:.4e} | {} | {} | {} | {} | {:.3} |",
            report.lambdas[k],
            report.outer_iters[k],
            report.inner_iters[k],
            report.kkt_passes[k],
            report.working_set_sizes[k],
            report.per_lambda_wall_ns[k] as f64 / 1e6,
        );
    }

    println!();
    println!("## Outer-iter histogram");
    println!();
    let max_outer_seen = *report.outer_iters.iter().max().unwrap_or(&0);
    let mut hist = vec![0usize; max_outer_seen + 1];
    for &o in &report.outer_iters {
        hist[o] += 1;
    }
    println!("| outer_iters | n_lambdas |");
    println!("|---:|---:|");
    for (k, c) in hist.iter().enumerate() {
        if *c > 0 {
            println!("| {k} | {c} |");
        }
    }

    println!();
    println!("## Wall-time concentration");
    println!();
    let mut wall_with_idx: Vec<(usize, u64)> = report
        .per_lambda_wall_ns
        .iter()
        .copied()
        .enumerate()
        .collect();
    wall_with_idx.sort_by_key(|(_, ns)| std::cmp::Reverse(*ns));
    let top10_ns: u64 = wall_with_idx.iter().take(10).map(|(_, ns)| *ns).sum();
    println!(
        "Top 10 λ account for {:.1}% of solve wall-time ({:.3}s / {:.3}s).",
        100.0 * (top10_ns as f64) / (total_wall_ns as f64),
        top10_ns as f64 / 1e9,
        total_wall_ns as f64 / 1e9
    );
    println!();
    println!("| rank | k | λ | outer | inner_sum | wall_ms |");
    println!("|---:|---:|---:|---:|---:|---:|");
    for (rank, (k, ns)) in wall_with_idx.iter().take(10).enumerate() {
        println!(
            "| {} | {k} | {:.4e} | {} | {} | {:.3} |",
            rank + 1,
            report.lambdas[*k],
            report.outer_iters[*k],
            report.inner_iters[*k],
            *ns as f64 / 1e6,
        );
    }
}
