//! Multi-task LS (M7.x) PyO3 bindings — dense + sparse, lasso / MCP /
//! SCAD / elastic net.
//!
//! Extracted from `lib.rs` in the M12 P4 refactor. Multi-task LS with
//! response Y ∈ ℝ^(n×K) and coefficient matrix B ∈ ℝ^(p×K) reduces to a
//! single group-lasso problem on a virtual (nK × pK) design via
//! `MultiTaskDesign<DenseMatrix>` (or `MultiTaskDesign<SparseCSC>` /
//! `MultiTaskDesign<Augmented<SparseCSC>>` / `MultiTaskDesign<Standardized<...>>`)
//! with row-major bvec layout `bvec[jK+k] = B[j,k]` and groups
//! `{jK, …, jK+K-1}` per feature. Centering is per-task on Y plus
//! shared on X. Per-task intercepts via `α_k = ȳ_k − Σ_j x̄_j B[j,k]`.
//!
//! Sparse variant uses column-augmentation for the intercept (one 1s
//! column on the inner CSC; the `MultiTaskDesign` wrapper replicates it
//! K times into K virtual intercept columns living in disjoint row
//! blocks). The intercept "feature" gets its own row-group with weight
//! 0 — `block_lambda_max`/strong-rule/KKT all see weight=0 on that
//! group and therefore leave it unpenalized. Each per-task intercept
//! then ends up at `bvec[p*K + k]` after the solve. Centering would
//! densify X, which is exactly the wall the scalar sparse paths hit.
//!
//! Cross-cutting helpers (`parse_screening`, CSC readers, glmnet scales)
//! live in `lib.rs` as `pub(crate)`; called here as `crate::name`.

use ndarray::{Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skein_core::{
    datafit::LeastSquares,
    design::{Augmented, DenseMatrix, MultiTaskDesign, Standardized},
    groups::Groups,
    penalty::{GroupElasticNet, GroupLasso, GroupPenalty},
    solver::{
        solve_block_path_lla, solve_block_path_timed, surrogate_weights_group_mcp,
        surrogate_weights_group_scad, BlockPathConfig, CdConfig,
    },
};

type MultiTaskPathOutput<'py> = (
    Bound<'py, PyArray2<f64>>, // coefs: (n_lambdas, p*K), row-major bvec layout
    Bound<'py, PyArray2<f64>>, // intercepts: (n_lambdas, K)
    Bound<'py, PyArray1<f64>>, // lambdas
    Bound<'py, PyDict>,
);

/// glmnet-style per-column std for a dense `X`:
/// `s_j = sqrt((‖X[:,j]‖² − n · x̄_j²) / n)`. Constant columns clamp to 1.0.
fn compute_dense_glmnet_scales_2d(x: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut s = Array1::<f64>::ones(p);
    for j in 0..p {
        let col = x.column(j);
        let mean = col.sum() / (n as f64);
        let mut sq = 0.0;
        for &v in col.iter() {
            sq += v * v;
        }
        let var = (sq / (n as f64)) - mean * mean;
        let scale = var.max(0.0).sqrt();
        s[j] = if scale > 1e-12 { scale } else { 1.0 };
    }
    s
}

/// Center `X` by column means (shared across tasks) and `Y` by per-task
/// column means; optionally scale `X` by per-column glmnet std. Returns
/// `(x_processed, y_stacked, x_means, y_means, x_scales)`.
/// `y_stacked[task*n + i] = Y_centered[i, task]` (task-outer).
#[allow(clippy::type_complexity)]
fn multitask_center_and_scale(
    x: &Array2<f64>,
    y: &Array2<f64>,
    fit_intercept: bool,
    standardize_x: bool,
) -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let n = x.nrows();
    let p = x.ncols();
    let k = y.ncols();
    debug_assert_eq!(y.nrows(), n);

    let mut x_means = Array1::<f64>::zeros(p);
    let mut y_means = Array1::<f64>::zeros(k);
    if fit_intercept {
        for j in 0..p {
            x_means[j] = x.column(j).sum() / (n as f64);
        }
        for task in 0..k {
            y_means[task] = y.column(task).sum() / (n as f64);
        }
    }

    let x_scales = if standardize_x {
        compute_dense_glmnet_scales_2d(x)
    } else {
        Array1::<f64>::ones(p)
    };

    let mut x_proc = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mu = if fit_intercept { x_means[j] } else { 0.0 };
        let inv_s = 1.0 / x_scales[j];
        for i in 0..n {
            x_proc[[i, j]] = (x[[i, j]] - mu) * inv_s;
        }
    }

    let mut y_stacked = Array1::<f64>::zeros(n * k);
    if fit_intercept {
        for task in 0..k {
            let mu = y_means[task];
            for i in 0..n {
                y_stacked[task * n + i] = y[[i, task]] - mu;
            }
        }
    } else {
        for task in 0..k {
            for i in 0..n {
                y_stacked[task * n + i] = y[[i, task]];
            }
        }
    }
    (x_proc, y_stacked, x_means, y_means, x_scales)
}

/// Descale a row-major bvec coefficients matrix by per-feature scales:
/// `B[j, k] /= s_j` for each lambda. Modifies `betas` in place.
fn multitask_descale_inplace(
    betas: &mut Array2<f64>,
    x_scales: &Array1<f64>,
    n_features: usize,
    n_tasks: usize,
) {
    let n_lambdas = betas.nrows();
    for lam_idx in 0..n_lambdas {
        for j in 0..n_features {
            let inv_s = 1.0 / x_scales[j];
            for task in 0..n_tasks {
                betas[[lam_idx, j * n_tasks + task]] *= inv_s;
            }
        }
    }
}

/// Recover per-task intercept from the (centered+descaled) coefficients:
/// `α_k = ȳ_k − Σ_j x̄_j B[j,k]`, where `B[j,k] = bvec[jK+k]`.
fn multitask_recover_intercepts(
    betas: &Array2<f64>, // (n_lambdas, p*K) row-major bvec, **descaled**
    x_means: &Array1<f64>,
    y_means: &Array1<f64>,
    n_features: usize,
    n_tasks: usize,
    fit_intercept: bool,
) -> Array2<f64> {
    let n_lambdas = betas.nrows();
    let mut out = Array2::<f64>::zeros((n_lambdas, n_tasks));
    if !fit_intercept {
        return out;
    }
    for lam_idx in 0..n_lambdas {
        for task in 0..n_tasks {
            let mut shift = 0.0;
            for j in 0..n_features {
                shift += x_means[j] * betas[[lam_idx, j * n_tasks + task]];
            }
            out[[lam_idx, task]] = y_means[task] - shift;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_multitask_path_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    screening: &str,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    make_inner: F,
) -> PyResult<MultiTaskPathOutput<'py>>
where
    F: Fn(f64, Array1<f64>) -> Box<dyn GroupPenalty> + Send,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let n = x_arr.nrows();
    let p = x_arr.ncols();
    if y_arr.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "Y must have {} rows (matching X), got {}",
            n,
            y_arr.nrows()
        )));
    }
    let k = y_arr.ncols();
    if k < 1 {
        return Err(PyValueError::new_err("Y must have at least one task"));
    }

    let weights_orig = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    p
                )));
            }
            arr
        }
        None => Array1::ones(p),
    };

    let (x_proc, y_stacked, x_means, y_means, x_scales) =
        multitask_center_and_scale(&x_arr, &y_arr, fit_intercept, standardize_x);

    let mut weights_eff = weights_orig.clone();
    if standardize_x {
        for j in 0..p {
            weights_eff[j] /= x_scales[j];
        }
    }

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: crate::ls::parse_screening(screening)?,
        parallel,
    };

    let design = MultiTaskDesign::new(DenseMatrix::new(x_proc), k);
    let datafit = LeastSquares::new(y_stacked);
    let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
    let make_pen =
        move |lam: f64| -> Box<dyn GroupPenalty> { make_inner(lam, weights_eff.clone()) };
    let (mut betas, report, times_ns) = py
        .allow_threads(|| solve_block_path_timed(&design, &datafit, make_pen, &groups, &block_cfg));
    if standardize_x {
        multitask_descale_inplace(&mut betas, &x_scales, p, k);
    }
    let intercepts = multitask_recover_intercepts(&betas, &x_means, &y_means, p, k, fit_intercept);

    let info = PyDict::new_bound(py);
    info.set_item("iters", report.iters)?;
    info.set_item("converged", report.converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;
    info.set_item("times_ns", times_ns)?;
    info.set_item("n_tasks", k)?;
    info.set_item("n_features", p)?;

    Ok((
        betas.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_multitask_path_lla_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    screening: &str,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    make_inner: F,
) -> PyResult<MultiTaskPathOutput<'py>>
where
    F: Fn(ArrayView1<f64>, &Groups, f64) -> Box<dyn GroupPenalty> + Send,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let n = x_arr.nrows();
    let p = x_arr.ncols();
    if y_arr.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "Y must have {} rows (matching X), got {}",
            n,
            y_arr.nrows()
        )));
    }
    let k = y_arr.ncols();
    if k < 1 {
        return Err(PyValueError::new_err("Y must have at least one task"));
    }

    let weights_orig = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    p
                )));
            }
            arr
        }
        None => Array1::ones(p),
    };

    let (x_proc, y_stacked, x_means, y_means, x_scales) =
        multitask_center_and_scale(&x_arr, &y_arr, fit_intercept, standardize_x);

    let mut weights_eff = weights_orig.clone();
    if standardize_x {
        for j in 0..p {
            weights_eff[j] /= x_scales[j];
        }
    }

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: crate::ls::parse_screening(screening)?,
        parallel,
    };

    let design = MultiTaskDesign::new(DenseMatrix::new(x_proc), k);
    let datafit = LeastSquares::new(y_stacked);
    let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p, k);
    let (mut betas, report) = py.allow_threads(|| {
        solve_block_path_lla(
            &design,
            &datafit,
            weights_eff,
            make_inner,
            &groups,
            &block_cfg,
            max_outer,
            outer_tol,
        )
    });
    if standardize_x {
        multitask_descale_inplace(&mut betas, &x_scales, p, k);
    }
    let intercepts = multitask_recover_intercepts(&betas, &x_means, &y_means, p, k, fit_intercept);

    let info = PyDict::new_bound(py);
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;
    info.set_item("n_tasks", k)?;
    info.set_item("n_features", p)?;

    Ok((
        betas.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_lasso_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<MultiTaskPathOutput<'py>> {
    build_multitask_path_outputs(
        py,
        x,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(GroupLasso::with_weights(lam, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_mcp_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<MultiTaskPathOutput<'py>> {
    let x_view = x.as_array();
    let p = x_view.ncols();
    let mut base_weights = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => Array1::ones(p),
    };
    if standardize_x {
        let dense = x_view.to_owned();
        let scales = compute_dense_glmnet_scales_2d(&dense);
        for j in 0..p {
            base_weights[j] /= scales[j];
        }
    }
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_weights.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };
    build_multitask_path_lla_outputs(
        py,
        x,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        make_inner,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_scad_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    a: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<MultiTaskPathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    let x_view = x.as_array();
    let p = x_view.ncols();
    let mut base_weights = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => Array1::ones(p),
    };
    if standardize_x {
        let dense = x_view.to_owned();
        let scales = compute_dense_glmnet_scales_2d(&dense);
        for j in 0..p {
            base_weights[j] /= scales[j];
        }
    }
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_scad(beta, g, lam, a, base_weights.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };
    build_multitask_path_lla_outputs(
        py,
        x,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        make_inner,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_elastic_net_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<MultiTaskPathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_multitask_path_outputs(
        py,
        x,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(GroupElasticNet::with_weights(lam, alpha, w)),
    )
}

// ---- Sparse multi-task LS helpers (M7.2) --------------------------------

#[allow(clippy::too_many_arguments)]
fn build_multitask_path_outputs_sparse<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    screening: &str,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    make_inner: F,
) -> PyResult<MultiTaskPathOutput<'py>>
where
    F: Fn(f64, Array1<f64>) -> Box<dyn GroupPenalty> + Send,
{
    let y_arr = y.as_array().to_owned();
    if y_arr.nrows() != n_rows {
        return Err(PyValueError::new_err(format!(
            "Y must have {} rows (matching X), got {}",
            n_rows,
            y_arr.nrows()
        )));
    }
    let k_tasks = y_arr.ncols();
    if k_tasks < 1 {
        return Err(PyValueError::new_err("Y must have at least one task"));
    }
    let p = n_cols;

    let user_weights = match &weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    p
                )));
            }
            arr
        }
        None => Array1::ones(p),
    };

    let mut y_stacked = Array1::<f64>::zeros(n_rows * k_tasks);
    for task in 0..k_tasks {
        for i in 0..n_rows {
            y_stacked[task * n_rows + i] = y_arr[[i, task]];
        }
    }

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, x_data, x_indices, x_indptr)?;
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let n_features_aug = if fit_intercept { p + 1 } else { p };
    let mut weights_eff = Array1::<f64>::zeros(n_features_aug);
    for j in 0..p {
        weights_eff[j] = user_weights[j];
        if let Some(scales) = &scales_user {
            weights_eff[j] /= scales[j];
        }
    }
    if fit_intercept {
        weights_eff[p] = 0.0;
    }

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: crate::ls::parse_screening(screening)?,
        parallel,
    };

    let datafit = LeastSquares::new(y_stacked);
    let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(n_features_aug, k_tasks);

    let make_pen =
        move |lam: f64| -> Box<dyn GroupPenalty> { make_inner(lam, weights_eff.clone()) };

    let scale_vec = scales_user.as_ref().map(|scales| {
        let mut v = Array1::<f64>::ones(n_features_aug);
        for j in 0..p {
            v[j] = scales[j];
        }
        v
    });

    let (betas, report, times_ns) = py.allow_threads(|| match (fit_intercept, scale_vec) {
        (true, Some(scales)) => {
            let std_design = Standardized::new(Augmented::new(csc), scales);
            let design = MultiTaskDesign::new(std_design, k_tasks);
            solve_block_path_timed(&design, &datafit, make_pen, &groups, &block_cfg)
        }
        (true, None) => {
            let augmented = Augmented::new(csc);
            let design = MultiTaskDesign::new(augmented, k_tasks);
            solve_block_path_timed(&design, &datafit, make_pen, &groups, &block_cfg)
        }
        (false, Some(scales)) => {
            let std_design = Standardized::new(csc, scales);
            let design = MultiTaskDesign::new(std_design, k_tasks);
            solve_block_path_timed(&design, &datafit, make_pen, &groups, &block_cfg)
        }
        (false, None) => {
            let design = MultiTaskDesign::new(csc, k_tasks);
            solve_block_path_timed(&design, &datafit, make_pen, &groups, &block_cfg)
        }
    });

    let n_lambdas_out = betas.nrows();
    let mut coefs_out = Array2::<f64>::zeros((n_lambdas_out, p * k_tasks));
    let mut intercepts_out = Array2::<f64>::zeros((n_lambdas_out, k_tasks));
    for lam_idx in 0..n_lambdas_out {
        for j in 0..p {
            let inv_s = scales_user.as_ref().map(|s| 1.0 / s[j]).unwrap_or(1.0);
            for task in 0..k_tasks {
                coefs_out[[lam_idx, j * k_tasks + task]] =
                    betas[[lam_idx, j * k_tasks + task]] * inv_s;
            }
        }
        if fit_intercept {
            for task in 0..k_tasks {
                intercepts_out[[lam_idx, task]] = betas[[lam_idx, p * k_tasks + task]];
            }
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("iters", report.iters)?;
    info.set_item("converged", report.converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;
    info.set_item("times_ns", times_ns)?;
    info.set_item("n_tasks", k_tasks)?;
    info.set_item("n_features", p)?;

    Ok((
        coefs_out.into_pyarray_bound(py),
        intercepts_out.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_multitask_path_lla_outputs_sparse<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    screening: &str,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    make_inner: F,
) -> PyResult<MultiTaskPathOutput<'py>>
where
    F: Fn(ArrayView1<f64>, &Groups, f64) -> Box<dyn GroupPenalty> + Send,
{
    let y_arr = y.as_array().to_owned();
    if y_arr.nrows() != n_rows {
        return Err(PyValueError::new_err(format!(
            "Y must have {} rows (matching X), got {}",
            n_rows,
            y_arr.nrows()
        )));
    }
    let k_tasks = y_arr.ncols();
    if k_tasks < 1 {
        return Err(PyValueError::new_err("Y must have at least one task"));
    }
    let p = n_cols;

    let user_weights = match &weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    p
                )));
            }
            arr
        }
        None => Array1::ones(p),
    };

    let mut y_stacked = Array1::<f64>::zeros(n_rows * k_tasks);
    for task in 0..k_tasks {
        for i in 0..n_rows {
            y_stacked[task * n_rows + i] = y_arr[[i, task]];
        }
    }

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, x_data, x_indices, x_indptr)?;
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let n_features_aug = if fit_intercept { p + 1 } else { p };
    let mut weights_eff = Array1::<f64>::zeros(n_features_aug);
    for j in 0..p {
        weights_eff[j] = user_weights[j];
        if let Some(scales) = &scales_user {
            weights_eff[j] /= scales[j];
        }
    }
    if fit_intercept {
        weights_eff[p] = 0.0;
    }

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: crate::ls::parse_screening(screening)?,
        parallel,
    };

    let datafit = LeastSquares::new(y_stacked);
    let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(n_features_aug, k_tasks);

    let scale_vec = scales_user.as_ref().map(|scales| {
        let mut v = Array1::<f64>::ones(n_features_aug);
        for j in 0..p {
            v[j] = scales[j];
        }
        v
    });

    let (betas, report) = py.allow_threads(|| match (fit_intercept, scale_vec) {
        (true, Some(scales)) => {
            let std_design = Standardized::new(Augmented::new(csc), scales);
            let design = MultiTaskDesign::new(std_design, k_tasks);
            solve_block_path_lla(
                &design,
                &datafit,
                weights_eff,
                make_inner,
                &groups,
                &block_cfg,
                max_outer,
                outer_tol,
            )
        }
        (true, None) => {
            let augmented = Augmented::new(csc);
            let design = MultiTaskDesign::new(augmented, k_tasks);
            solve_block_path_lla(
                &design,
                &datafit,
                weights_eff,
                make_inner,
                &groups,
                &block_cfg,
                max_outer,
                outer_tol,
            )
        }
        (false, Some(scales)) => {
            let std_design = Standardized::new(csc, scales);
            let design = MultiTaskDesign::new(std_design, k_tasks);
            solve_block_path_lla(
                &design,
                &datafit,
                weights_eff,
                make_inner,
                &groups,
                &block_cfg,
                max_outer,
                outer_tol,
            )
        }
        (false, None) => {
            let design = MultiTaskDesign::new(csc, k_tasks);
            solve_block_path_lla(
                &design,
                &datafit,
                weights_eff,
                make_inner,
                &groups,
                &block_cfg,
                max_outer,
                outer_tol,
            )
        }
    });

    let n_lambdas_out = betas.nrows();
    let mut coefs_out = Array2::<f64>::zeros((n_lambdas_out, p * k_tasks));
    let mut intercepts_out = Array2::<f64>::zeros((n_lambdas_out, k_tasks));
    for lam_idx in 0..n_lambdas_out {
        for j in 0..p {
            let inv_s = scales_user.as_ref().map(|s| 1.0 / s[j]).unwrap_or(1.0);
            for task in 0..k_tasks {
                coefs_out[[lam_idx, j * k_tasks + task]] =
                    betas[[lam_idx, j * k_tasks + task]] * inv_s;
            }
        }
        if fit_intercept {
            for task in 0..k_tasks {
                intercepts_out[[lam_idx, task]] = betas[[lam_idx, p * k_tasks + task]];
            }
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;
    info.set_item("n_tasks", k_tasks)?;
    info.set_item("n_features", p)?;

    Ok((
        coefs_out.into_pyarray_bound(py),
        intercepts_out.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_lasso_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray2<f64>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<MultiTaskPathOutput<'py>> {
    build_multitask_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(GroupLasso::with_weights(lam, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_mcp_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray2<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<MultiTaskPathOutput<'py>> {
    let n_features_aug = if fit_intercept { n_cols + 1 } else { n_cols };
    let mut base_weights_eff = match &weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            let mut eff = Array1::<f64>::zeros(n_features_aug);
            for j in 0..n_cols {
                eff[j] = arr[j];
            }
            eff
        }
        None => {
            let mut eff = Array1::<f64>::ones(n_features_aug);
            if fit_intercept {
                eff[n_cols] = 0.0;
            }
            eff
        }
    };
    if standardize_x {
        let csc = crate::ls::read_csc_arrays(
            n_rows,
            n_cols,
            x_data.clone(),
            x_indices.clone(),
            x_indptr.clone(),
        )?;
        let scales = crate::ls::compute_csc_glmnet_scales(&csc);
        for j in 0..n_cols {
            base_weights_eff[j] /= scales[j];
        }
    }
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_weights_eff.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };
    build_multitask_path_lla_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        make_inner,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_scad_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray2<f64>,
    a: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<MultiTaskPathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    let n_features_aug = if fit_intercept { n_cols + 1 } else { n_cols };
    let mut base_weights_eff = match &weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            let mut eff = Array1::<f64>::zeros(n_features_aug);
            for j in 0..n_cols {
                eff[j] = arr[j];
            }
            eff
        }
        None => {
            let mut eff = Array1::<f64>::ones(n_features_aug);
            if fit_intercept {
                eff[n_cols] = 0.0;
            }
            eff
        }
    };
    if standardize_x {
        let csc = crate::ls::read_csc_arrays(
            n_rows,
            n_cols,
            x_data.clone(),
            x_indices.clone(),
            x_indptr.clone(),
        )?;
        let scales = crate::ls::compute_csc_glmnet_scales(&csc);
        for j in 0..n_cols {
            base_weights_eff[j] /= scales[j];
        }
    }
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_scad(beta, g, lam, a, base_weights_eff.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };
    build_multitask_path_lla_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        make_inner,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multitask_elastic_net_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray2<f64>,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<MultiTaskPathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_multitask_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        screening,
        parallel,
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(GroupElasticNet::with_weights(lam, alpha, w)),
    )
}
