//! P6 step 1 — per-phase scaling attribution for `cd_solve_subset` on
//! Lasso/LS. Re-implements the inner CD loop with per-call `Instant`
//! timers (same pattern as `mcp_cd_attribution.rs`) and runs the
//! canonical small / medium / large sizes from `benches/v2/config.yaml`:
//!   small  : n=1000,   p=200   (column ~  8 KB — fits L1)
//!   medium : n=10000,  p=1000  (column ~ 80 KB — fits L2)
//!   large  : n=50000,  p=5000  (column ~400 KB — column ≳ per-core L2)
//!
//! Each cell runs a single inner-CD sweep at λ_max × 1e-3 (deep tail,
//! cold start, full feature set). The reported number that decides P6's
//! bandwidth premise is the per-call cost of the two column-streaming
//! operations:
//!
//!   grad_ns_per_elem = t_grad_ns / (visits × n)
//!   axpy_ns_per_elem = t_axpy_ns / (nz_updates × n)
//!
//! Each is O(1) work per matrix element, so these stay flat across
//! sizes when the column lives in cache and grow when the column has
//! to be streamed from main memory each visit. If both spike on large
//! and not on medium, lever (A) — fused col_dot + col_axpy in a single
//! column pass — is the right attack. If they stay flat and the wall
//! growth lives elsewhere, P6 needs a different lever.
//!
//! Build + run:
//!   cargo build --release --example lasso_cd_attribution_scaling
//!   ./target/release/examples/lasso_cd_attribution_scaling
//!   SKEIN_SCALING_LARGE=1 ./target/release/examples/lasso_cd_attribution_scaling

use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares, design::DenseMatrix, penalty::ElasticNet, solver::lambda_max, Datafit,
    DesignMatrix, Penalty,
};
use std::time::Instant;

const K_ACTIVE: usize = 10;
const TOL: f64 = 1e-7;
const MAX_ITER: usize = 50;
const LAM_RATIO: f64 = 1e-3;

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
        let m: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        (signal.iter().map(|v| (v - m).powi(2)).sum::<f64>() / signal.len() as f64).sqrt()
    };
    let noise_scale = signal_std / 5.0;
    let mut y = signal.clone();
    for v in y.iter_mut() {
        *v += noise_scale * rng.normal();
    }
    (DenseMatrix::new(x), y)
}

#[derive(Default, Debug, Clone, Copy)]
struct CdPhase {
    t_grad_ns: u128,
    t_prox_ns: u128,
    t_axpy_ns: u128,
    t_value_ns: u128,
    n_coord_visits: u64,
    n_nonzero_updates: u64,
    iters: u64,
}

fn cd_inner(
    beta_init: Array1<f64>,
    features: &[usize],
    design: &dyn DesignMatrix,
    datafit: &dyn Datafit,
    penalty: &dyn Penalty,
    tol: f64,
    max_iter: usize,
) -> (Array1<f64>, CdPhase) {
    let mut beta = beta_init;
    let mut r = datafit.init_residual(design, beta.view());
    let mut ph = CdPhase::default();

    for it in 0..max_iter {
        let mut max_delta = 0.0_f64;
        for &j in features {
            let lj = datafit.coord_lipschitz(design, j);
            if lj == 0.0 {
                continue;
            }
            ph.n_coord_visits += 1;

            let t0 = Instant::now();
            let grad_j = datafit.coord_grad(design, j, r.view());
            ph.t_grad_ns += t0.elapsed().as_nanos();

            let z = beta[j] - grad_j / lj;
            let step = 1.0 / lj;

            let t0 = Instant::now();
            let new_bj = penalty.prox_coord(j, z, step);
            ph.t_prox_ns += t0.elapsed().as_nanos();

            let delta = new_bj - beta[j];
            if delta != 0.0 {
                let t0 = Instant::now();
                design.col_axpy(j, delta, r.view_mut());
                ph.t_axpy_ns += t0.elapsed().as_nanos();

                beta[j] = new_bj;
                ph.n_nonzero_updates += 1;
                let abs_delta = delta.abs();
                if abs_delta > max_delta {
                    max_delta = abs_delta;
                }
            }
        }

        let t0 = Instant::now();
        let _obj = datafit.value(r.view()) + penalty.value(beta.view());
        ph.t_value_ns += t0.elapsed().as_nanos();

        ph.iters = (it + 1) as u64;
        if max_delta < tol {
            break;
        }
    }
    (beta, ph)
}

#[derive(Debug, Clone, Copy)]
struct CellResult {
    label: &'static str,
    n: usize,
    p: usize,
    wall_ms: f64,
    ph: CdPhase,
}

fn run_cell(label: &'static str, n: usize, p: usize, warmup: bool) -> CellResult {
    println!();
    println!("=== {label}: n={n}, p={p} ===");
    let (design, y) = make_problem(n, p, 1);
    let datafit = LeastSquares::new(y);
    let weights = Array1::<f64>::ones(p);
    let lam_max = lambda_max(&design, &datafit, weights.view());
    let lam = lam_max * LAM_RATIO;
    let penalty = ElasticNet::with_weights(lam, 1.0, weights);
    let features: Vec<usize> = (0..p).collect();
    let beta0 = Array1::<f64>::zeros(p);

    if warmup {
        let _ = cd_inner(
            beta0.clone(),
            &features,
            &design,
            &datafit,
            &penalty,
            TOL,
            MAX_ITER,
        );
    }

    let t0 = Instant::now();
    let (beta, ph) = cd_inner(beta0, &features, &design, &datafit, &penalty, TOL, MAX_ITER);
    let wall = t0.elapsed();
    let wall_ms = wall.as_secs_f64() * 1e3;

    let active = beta.iter().filter(|&&v| v != 0.0).count();
    let total_ns = ph.t_grad_ns + ph.t_prox_ns + ph.t_axpy_ns + ph.t_value_ns;
    let share = |ns: u128| 100.0 * ns as f64 / total_ns as f64;
    let to_ms = |ns: u128| ns as f64 / 1e6;
    let grad_per_elem = ph.t_grad_ns as f64 / (ph.n_coord_visits.max(1) as f64 * n as f64);
    let axpy_per_elem = ph.t_axpy_ns as f64 / (ph.n_nonzero_updates.max(1) as f64 * n as f64);

    println!(
        "  wall={wall_ms:>7.1}ms  iters={iters}  visits={visits}  nz_updates={nz} ({nz_pct:.1}% of visits)  active={active}/{p}",
        iters = ph.iters,
        visits = ph.n_coord_visits,
        nz = ph.n_nonzero_updates,
        nz_pct = 100.0 * ph.n_nonzero_updates as f64 / ph.n_coord_visits.max(1) as f64,
    );
    println!(
        "  phase wall  grad={:>7.1}ms ({:>4.1}%)  prox={:>5.2}ms ({:>4.1}%)  axpy={:>7.1}ms ({:>4.1}%)  value={:>5.2}ms ({:>4.1}%)",
        to_ms(ph.t_grad_ns),
        share(ph.t_grad_ns),
        to_ms(ph.t_prox_ns),
        share(ph.t_prox_ns),
        to_ms(ph.t_axpy_ns),
        share(ph.t_axpy_ns),
        to_ms(ph.t_value_ns),
        share(ph.t_value_ns),
    );
    println!("  per-element  grad={grad_per_elem:.3} ns/elem  axpy={axpy_per_elem:.3} ns/elem");

    CellResult {
        label,
        n,
        p,
        wall_ms,
        ph,
    }
}

fn report_scaling(from: &CellResult, to: &CellResult) {
    let n_ratio = to.n as f64 / from.n as f64;
    let p_ratio = to.p as f64 / from.p as f64;
    let np_ratio = n_ratio * p_ratio;

    let wall_ratio = to.wall_ms / from.wall_ms;
    let grad_ratio = to.ph.t_grad_ns as f64 / from.ph.t_grad_ns.max(1) as f64;
    let axpy_ratio = to.ph.t_axpy_ns as f64 / from.ph.t_axpy_ns.max(1) as f64;

    let from_grad_per_elem =
        from.ph.t_grad_ns as f64 / (from.ph.n_coord_visits.max(1) as f64 * from.n as f64);
    let to_grad_per_elem =
        to.ph.t_grad_ns as f64 / (to.ph.n_coord_visits.max(1) as f64 * to.n as f64);
    let from_axpy_per_elem =
        from.ph.t_axpy_ns as f64 / (from.ph.n_nonzero_updates.max(1) as f64 * from.n as f64);
    let to_axpy_per_elem =
        to.ph.t_axpy_ns as f64 / (to.ph.n_nonzero_updates.max(1) as f64 * to.n as f64);

    println!();
    println!("  {} → {}", from.label, to.label);
    println!("    n×p ratio       : {np_ratio:.1}×  (n {n_ratio:.1}× × p {p_ratio:.1}×)");
    println!("    total wall      : {wall_ratio:.2}×");
    println!(
        "    grad wall       : {grad_ratio:.2}×    per-elem: {from_grad_per_elem:.3} → {to_grad_per_elem:.3} ns/elem  ({:.2}×)",
        to_grad_per_elem / from_grad_per_elem
    );
    println!(
        "    axpy wall       : {axpy_ratio:.2}×    per-elem: {from_axpy_per_elem:.3} → {to_axpy_per_elem:.3} ns/elem  ({:.2}×)",
        to_axpy_per_elem / from_axpy_per_elem
    );
    println!("    (per-elem ratios > 1× confirm column-streaming bandwidth wall;");
    println!("     ratios near 1× would mean wall growth lives elsewhere)");
}

fn main() {
    println!("P6 scaling attribution — Lasso/LS inner CD, λ_max × 1e-3, cold-start, full sweep");
    println!("  tol={TOL:.0e}  max_iter={MAX_ITER}  snr=5  k_active={K_ACTIVE}");

    let small = run_cell("small", 1_000, 200, true);
    let medium = run_cell("medium", 10_000, 1_000, true);
    let large = if std::env::var("SKEIN_SCALING_LARGE").is_ok() {
        Some(run_cell("large", 50_000, 5_000, false))
    } else {
        None
    };

    println!();
    println!("=== scaling ===");
    report_scaling(&small, &medium);
    if let Some(large) = large {
        report_scaling(&medium, &large);
        report_scaling(&small, &large);
    } else {
        println!();
        println!("  (set SKEIN_SCALING_LARGE=1 to also run large = n=50k, p=5k)");
    }
}
