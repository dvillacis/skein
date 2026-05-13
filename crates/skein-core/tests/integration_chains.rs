//! End-to-end integration chains.
//!
//! Inline `#[cfg(test)]` modules cover each module in isolation. This
//! file exercises *compositions across module boundaries* — the kind
//! of bug an inline test for any one of (datafit, design wrapper,
//! penalty, solver) would never catch:
//!
//!   1. `Standardized<DenseMatrix>` + scalar MCP via the LLA path
//!      solver — composes the lazy-standardize wrapper with the
//!      nonconvex outer loop.
//!
//!   2. `SparseCSC` + lasso via `solve_path` — sparse design driving
//!      the strong-rule + screening path solver.
//!
//!   3. `Standardized<SparseCSC>` + `BinomialLogit` + group MCP via
//!      LLA → prox-Newton → block-CD — the headline chain spanning
//!      every trait. If a refactor breaks any of the four
//!      interfaces, this fires.

use ndarray::{array, Array1, Array2};
use skein_core::datafit::{BinomialLogit, LeastSquares};
use skein_core::design::{DenseMatrix, SparseCSC, Standardized};
use skein_core::penalty::{ElasticNet, GroupLasso, GroupPenalty, Mcp, Penalty};
use skein_core::solver::{
    prox_newton_block_solve_path, solve_path, solve_path_lla, surrogate_weights_group_mcp,
    CdConfig, PathConfig,
};
use skein_core::{DesignMatrix, Groups};

/// Synthetic LS problem: planted sparse signal in a Gaussian-ish
/// design. Returns `(X, y, beta_true)`.
fn ls_problem(
    n: usize,
    p: usize,
    active: &[(usize, f64)],
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    // Simple deterministic pseudo-random — Lehmer LCG, plenty for
    // recovery tests. Keeps the tree dependency-free.
    let mut state: u64 = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to a [-1, 1] f64 with reasonable spread.
        let bits = (state >> 11) as u32;
        let u = (bits as f64) / (u32::MAX as f64); // [0, 1)
        2.0 * u - 1.0
    };

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = next();
        }
    }
    let mut beta = Array1::<f64>::zeros(p);
    for &(j, v) in active {
        beta[j] = v;
    }
    let mut y = x.dot(&beta);
    // Light noise.
    for i in 0..n {
        y[i] += 0.05 * next();
    }
    (x, y, beta)
}

#[test]
fn standardized_dense_lasso_path_recovers_support() {
    // LS + lasso path through `Standardized<DenseMatrix>`. Confirms
    // that lazy column scaling composes cleanly with the strong-rule
    // path solver and that the small-λ end recovers the planted
    // support.
    let (x, y, _) = ls_problem(150, 12, &[(0, 1.5), (3, -2.0), (7, 0.8)], 1);
    let scales = Array1::from(vec![
        1.0, 1.5, 0.8, 1.2, 0.7, 2.1, 1.4, 0.9, 1.1, 1.6, 0.6, 1.3,
    ]);
    let design = Standardized::new(DenseMatrix::new(x), scales);
    let datafit = LeastSquares::new(y);

    let cfg = PathConfig {
        n_lambdas: 40,
        lambda_min_ratio: 5e-3,
        cd: CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: Some(5),
        },
        ..PathConfig::default()
    };

    let (betas, report) = solve_path(
        &design,
        &datafit,
        |lam| Box::new(ElasticNet::new(lam, 1.0, design.n_features())) as Box<dyn Penalty>,
        &cfg,
    );

    assert_eq!(betas.nrows(), report.lambdas.len());
    assert_eq!(betas.ncols(), design.n_features());
    // λ_max gives the all-zero solution.
    let first = betas.row(0);
    for v in first.iter() {
        assert!(v.abs() < 1e-8, "λ_max row should be zero, got {v}");
    }
    // Smallest λ should activate the planted features.
    let last = betas.row(report.lambdas.len() - 1);
    let active = |j: usize| last[j].abs() > 1e-3;
    assert!(
        active(0),
        "feature 0 should be active at smallest λ; got {}",
        last[0]
    );
    assert!(
        active(3),
        "feature 3 should be active at smallest λ; got {}",
        last[3]
    );
    assert!(
        active(7),
        "feature 7 should be active at smallest λ; got {}",
        last[7]
    );
    // Sign agreement with the planted signal.
    assert!(last[0] > 0.0);
    assert!(last[3] < 0.0);
    assert!(last[7] > 0.0);
}

#[test]
fn standardized_dense_mcp_via_lla_recovers_support() {
    // Same problem as above but with MCP (nonconvex) via the LLA path
    // solver. This is the chain: Standardized + LeastSquares + LLA
    // outer + lasso inner. Catches any breakage in surrogate-weight
    // wiring through the standardize-scaled gradient.
    let (x, y, _) = ls_problem(200, 15, &[(1, 2.0), (5, -1.5), (10, 1.0)], 7);
    let scales = Array1::from_iter((0..15).map(|j| 0.5 + 0.1 * j as f64));
    let design = Standardized::new(DenseMatrix::new(x), scales);
    let datafit = LeastSquares::new(y);
    let p = design.n_features();
    let base = Array1::<f64>::ones(p);
    let gamma = 3.0;

    // LLA inner: lasso with surrogate weights for MCP at current β.
    use ndarray::ArrayView1;

    let make_inner =
        move |beta: ArrayView1<'_, f64>, lam: f64, base: ArrayView1<'_, f64>| -> Box<dyn Penalty> {
            // Per-coord MCP surrogate: w_j = base_j · max(1 − |β_j|/(γ·λ·base_j), 0).
            let mut w = Array1::<f64>::zeros(p);
            for j in 0..p {
                let lam_eff = lam * base[j];
                let abs_b = beta[j].abs();
                w[j] = if abs_b >= gamma * lam_eff {
                    0.0
                } else {
                    base[j] * (1.0 - abs_b / (gamma * lam_eff)).max(0.0)
                };
            }
            Box::new(ElasticNet::with_weights(lam, 1.0, w)) as Box<dyn Penalty>
        };

    let (betas, report) = solve_path_lla(
        &design,
        &datafit,
        base,
        make_inner,
        25,
        5e-3,
        None,
        &CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: Some(5),
        },
        20,
        1e-8,
    );

    assert_eq!(betas.nrows(), report.lambdas.len());
    let last = betas.row(report.lambdas.len() - 1);
    assert!(
        last[1].abs() > 0.5,
        "feature 1 should be active; got {}",
        last[1]
    );
    assert!(
        last[5].abs() > 0.5,
        "feature 5 should be active; got {}",
        last[5]
    );
    assert!(
        last[10].abs() > 0.3,
        "feature 10 should be active; got {}",
        last[10]
    );
    // MCP shouldn't activate the planted-zero features.
    let inactive_count = (0..p)
        .filter(|&j| j != 1 && j != 5 && j != 10 && last[j].abs() < 0.1)
        .count();
    assert!(
        inactive_count >= 9,
        "expected most planted-zero features to remain ≈0; only {inactive_count} did"
    );
}

#[test]
fn sparse_csc_lasso_path_matches_dense_path() {
    // SparseCSC + lasso vs DenseMatrix + lasso on the same numerical
    // matrix. Catches divergence between the dense and sparse fast
    // paths inside the path solver / CD inner.
    let (x, y, _) = ls_problem(120, 10, &[(2, 1.5), (6, -1.0)], 11);
    // Build a SparseCSC view of the same matrix (no zeros — purely a
    // representation switch).
    let n = x.nrows();
    let p = x.ncols();
    let mut data = Vec::with_capacity(n * p);
    let mut indices = Vec::with_capacity(n * p);
    let mut indptr = Vec::with_capacity(p + 1);
    indptr.push(0_usize);
    for j in 0..p {
        for i in 0..n {
            data.push(x[[i, j]]);
            indices.push(i);
        }
        indptr.push(data.len());
    }
    let sparse = SparseCSC::new(
        n,
        Array1::from(data),
        Array1::from(indices),
        Array1::from(indptr),
    );
    let dense = DenseMatrix::new(x);

    let datafit_a = LeastSquares::new(y.clone());
    let datafit_b = LeastSquares::new(y);

    let cfg = PathConfig {
        n_lambdas: 30,
        lambda_min_ratio: 1e-2,
        cd: CdConfig {
            max_iter: 5000,
            tol: 1e-12,
            acceleration: Some(5),
        },
        ..PathConfig::default()
    };

    let (betas_dense, _) = solve_path(
        &dense,
        &datafit_a,
        |lam| Box::new(ElasticNet::new(lam, 1.0, p)) as Box<dyn Penalty>,
        &cfg,
    );
    let (betas_sparse, _) = solve_path(
        &sparse,
        &datafit_b,
        |lam| Box::new(ElasticNet::new(lam, 1.0, p)) as Box<dyn Penalty>,
        &cfg,
    );
    assert_eq!(betas_dense.dim(), betas_sparse.dim());
    // Match to ~1e-5 (CD tolerance + screening differences).
    for k in 0..betas_dense.nrows() {
        for j in 0..p {
            let d = (betas_dense[[k, j]] - betas_sparse[[k, j]]).abs();
            assert!(d < 1e-5, "λ-row {k}, j={j}: |dense - sparse| = {d:.2e}");
        }
    }
}

#[test]
fn standardized_sparse_csc_logistic_group_mcp_via_lla_recovers_active_groups() {
    // Headline chain: `Standardized<SparseCSC>` + `BinomialLogit` +
    // `GroupMcp` via prox-Newton block CD with LLA surrogate.
    // Spans: design wrapper composition (Standardized over SparseCSC),
    // GLM weighted-LS surrogate, block-CD inner, LLA outer.
    let n = 250;
    let p = 8;
    // Build a deterministic dense matrix, then convert to SparseCSC
    // (no zeros, but the sparse representation exercises the sparse
    // CD code path).
    let (xdense, _yreal, _) = ls_problem(n, p, &[], 17);
    let mut data = Vec::with_capacity(n * p);
    let mut indices = Vec::with_capacity(n * p);
    let mut indptr = Vec::with_capacity(p + 1);
    indptr.push(0_usize);
    for j in 0..p {
        for i in 0..n {
            data.push(xdense[[i, j]]);
            indices.push(i);
        }
        indptr.push(data.len());
    }
    let sparse = SparseCSC::new(
        n,
        Array1::from(data),
        Array1::from(indices),
        Array1::from(indptr),
    );
    let scales = Array1::from(vec![1.0, 1.4, 0.7, 1.2, 0.6, 1.8, 0.9, 1.1]);
    let design = Standardized::new(sparse, scales);

    // Plant signal in groups 0 and 2 (out of 4 groups of size 2).
    let mut beta_true = Array1::<f64>::zeros(p);
    beta_true[0] = 1.5; // group 0
    beta_true[1] = -1.0; // group 0
    beta_true[4] = 1.2; // group 2
    beta_true[5] = -0.8; // group 2
    let eta = design.matvec(beta_true.view());
    // Sample binary labels deterministically: y_i = 1 if σ(η_i) > 0.5
    // shifted by a tiny per-sample dither so it's not all-or-nothing.
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let p1 = 1.0 / (1.0 + (-eta[i]).exp());
            // Deterministic dither in [0, 1).
            let mut s = (i as u64).wrapping_mul(2654435761);
            s ^= s >> 16;
            let u = (s as u32 as f64) / (u32::MAX as f64);
            if u < p1 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let glm = BinomialLogit::new(y);
    let groups = Groups::contiguous_blocks(p, 2);
    let base = Array1::<f64>::ones(groups.n_groups());
    let gamma = 3.0;

    let base_for_closure = base.clone();
    let make_inner =
        move |beta: ndarray::ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_for_closure.view());
            Box::new(GroupLasso::with_weights(lam, w)) as Box<dyn GroupPenalty>
        };

    let (betas, report) = prox_newton_block_solve_path(
        &design,
        &glm,
        base,
        make_inner,
        &groups,
        25,
        5e-3,
        None,
        &CdConfig {
            max_iter: 5000,
            tol: 1e-10,
            acceleration: None,
        },
        30,
        1e-7,
    );

    let last = report.lambdas.len() - 1;
    let last_beta = betas.row(last).to_owned();
    let group_norm = |g: usize| -> f64 {
        groups
            .group(g)
            .iter()
            .map(|&j| last_beta[j] * last_beta[j])
            .sum::<f64>()
            .sqrt()
    };
    assert!(
        group_norm(0) > 0.2,
        "group 0 should be active; norm = {}",
        group_norm(0)
    );
    assert!(
        group_norm(2) > 0.2,
        "group 2 should be active; norm = {}",
        group_norm(2)
    );
}

// --- regression spot-check: verify the trait-object dispatch works
// --- through the public API (every concrete type is `Sync + Send`,
// --- and a `&dyn Datafit + Sync` is what the parallel solvers
// --- actually receive). A compile-time guard rather than a runtime
// --- assertion.
#[test]
fn trait_objects_satisfy_sync_send_at_public_surface() {
    fn assert_sync_send<T: Sync + Send>() {}
    assert_sync_send::<DenseMatrix>();
    assert_sync_send::<SparseCSC>();
    assert_sync_send::<Standardized<DenseMatrix>>();
    assert_sync_send::<Standardized<SparseCSC>>();
    assert_sync_send::<LeastSquares>();
    assert_sync_send::<BinomialLogit>();
    assert_sync_send::<Mcp>();
    assert_sync_send::<ElasticNet>();
    assert_sync_send::<GroupLasso>();

    // And the boxed-trait shape the LLA inner closures produce.
    fn box_penalty() -> Box<dyn Penalty> {
        Box::new(ElasticNet::new(0.1, 1.0, 4))
    }
    fn box_group_penalty() -> Box<dyn GroupPenalty> {
        Box::new(GroupLasso::new(0.1, 2))
    }
    let _ = box_penalty();
    let _ = box_group_penalty();

    // `array!` is included to keep the prelude lean; touch it once
    // so unused-import warnings don't fire on this minimal test.
    let _ = array![1.0_f64, 2.0];
}
