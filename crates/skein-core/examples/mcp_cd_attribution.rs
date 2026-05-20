//! Inner-CD attribution for P2 — re-implements the `cd_solve_subset`
//! loop in this example so we can count nonzero updates and time the
//! per-phase BLAS calls (`coord_grad`, `col_axpy`, `value(r)`) for
//! Lasso vs MCP on the same medium/deep problem. The phase log from
//! `mcp_ls_medium` localised the gap to `inner_cd`; the microbench in
//! `mcp_vs_lasso_micro` ruled out `prox_coord` and `value(beta)` as the
//! cause. This example tests the next hypothesis: MCP's firm-threshold
//! shrinks fewer coords to exactly zero than Lasso's soft-threshold,
//! triggering more `col_axpy` calls per sweep.
//!
//! Build + run: `cargo run --release --example mcp_cd_attribution`

use ndarray::{Array1, Array2};
use skein_core::{
    datafit::LeastSquares,
    design::DenseMatrix,
    penalty::{ElasticNet, Mcp},
    solver::lambda_max,
    Datafit, DesignMatrix, Penalty,
};
use std::time::Instant;

const N: usize = 10_000;
const P: usize = 1_000;
const MCP_GAMMA: f64 = 3.0;

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
) -> (Array1<f64>, Array1<f64>, CdPhase) {
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
    (beta, r, ph)
}

fn print_phase(name: &str, ph: &CdPhase) {
    let to_ms = |ns: u128| ns as f64 / 1e6;
    let total = ph.t_grad_ns + ph.t_prox_ns + ph.t_axpy_ns + ph.t_value_ns;
    println!(
        "  {:>10}  iters={:>3}  visits={:>6}  nz_updates={:>6} ({:>5.1}% of visits)",
        name,
        ph.iters,
        ph.n_coord_visits,
        ph.n_nonzero_updates,
        100.0 * ph.n_nonzero_updates as f64 / ph.n_coord_visits as f64,
    );
    println!(
        "              grad={:>6.1}ms  prox={:>5.2}ms  axpy={:>6.1}ms  value={:>5.2}ms  total={:>6.1}ms",
        to_ms(ph.t_grad_ns),
        to_ms(ph.t_prox_ns),
        to_ms(ph.t_axpy_ns),
        to_ms(ph.t_value_ns),
        to_ms(total),
    );
}

fn main() {
    println!("building problem (n={N}, p={P}, k_active=10, snr=5)…");
    let t0 = Instant::now();
    let (design, y) = make_problem();
    println!("  built in {:?}", t0.elapsed());
    let datafit = LeastSquares::new(y);
    let weights = Array1::<f64>::ones(P);
    let lam_max = lambda_max(&design, &datafit, weights.view());

    // Pick a "deep tail" λ that saturates the active set. λ_max ratio
    // ≈ 1e-3 reproduces the regime where the path-level phase log
    // sees its biggest MCP/Lasso gap.
    let lam_tail = lam_max * 1e-3;
    let features: Vec<usize> = (0..P).collect();

    let lasso = ElasticNet::with_weights(lam_tail, 1.0, weights.clone());
    let mcp = Mcp::with_weights(lam_tail, MCP_GAMMA, weights.clone());

    // Cold start from zero — same starting point for both penalties so
    // the comparison is apples-to-apples. (Path-level runs warm-start;
    // we're isolating the inner CD cost, not the convergence dynamics.)
    let beta0 = Array1::<f64>::zeros(P);

    println!("\n--- inner CD @ λ_max × 1e-3 (deep tail), full feature sweep ---");

    // Warm-up.
    let _ = cd_inner(
        beta0.clone(),
        &features,
        &design,
        &datafit,
        &lasso,
        1e-6,
        100,
    );
    let _ = cd_inner(beta0.clone(), &features, &design, &datafit, &mcp, 1e-6, 100);

    let t0 = Instant::now();
    let (b_lasso, _, ph_lasso) = cd_inner(
        beta0.clone(),
        &features,
        &design,
        &datafit,
        &lasso,
        1e-6,
        100,
    );
    let wall_lasso = t0.elapsed();

    let t0 = Instant::now();
    let (b_mcp, _, ph_mcp) = cd_inner(beta0.clone(), &features, &design, &datafit, &mcp, 1e-6, 100);
    let wall_mcp = t0.elapsed();

    let active_lasso = b_lasso.iter().filter(|&&v| v != 0.0).count();
    let active_mcp = b_mcp.iter().filter(|&&v| v != 0.0).count();
    println!(
        "  Lasso wall = {:?}; final active = {}/{}",
        wall_lasso, active_lasso, P
    );
    print_phase("lasso", &ph_lasso);
    println!(
        "  MCP   wall = {:?}; final active = {}/{}",
        wall_mcp, active_mcp, P
    );
    print_phase("mcp", &ph_mcp);

    println!("\nratios (mcp / lasso):");
    println!(
        "  iters       : {:.2}×",
        ph_mcp.iters as f64 / ph_lasso.iters as f64
    );
    println!(
        "  visits      : {:.2}×",
        ph_mcp.n_coord_visits as f64 / ph_lasso.n_coord_visits as f64
    );
    println!(
        "  nz_updates  : {:.2}×",
        ph_mcp.n_nonzero_updates as f64 / ph_lasso.n_nonzero_updates as f64
    );
    println!(
        "  axpy wall   : {:.2}×",
        ph_mcp.t_axpy_ns as f64 / ph_lasso.t_axpy_ns as f64
    );
    println!(
        "  grad wall   : {:.2}×",
        ph_mcp.t_grad_ns as f64 / ph_lasso.t_grad_ns as f64
    );
}
