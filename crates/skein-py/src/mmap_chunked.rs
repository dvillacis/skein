//! Memory-mapped + row-block chunked PyO3 entry points (M4.x).
//!
//! Extracted from `lib.rs` in the M12 P4 refactor. `MmapMatrix` reads
//! `X` directly from a column-major raw `f64` (or `f32`) file — no
//! in-RAM copy, no scipy.sparse marshalling. `Chunked<C>` lets the
//! solver treat a list of equal-`p` row-block chunks as one design
//! matrix. Both wrap any `DesignMatrix` and compose with `Augmented`
//! (intercept) / `Standardized` (scaling) the same as every other
//! backend.
//!
//! v1 surface: scalar LS+MCP and scalar logistic+MCP × {f64, f32}. The
//! other 22 entry points (group, sparse-group, EN, SCAD, Poisson, Cox,
//! …) would follow the same pattern; defer until there's user demand.
//!
//! Cross-cutting helpers (`parse_screening`, `split_intercept`,
//! `validate_y_binary`, `PathOutput`) live in `lib.rs` as `pub(crate)`;
//! called here as `crate::name`.

use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skein_core::{
    datafit::{BinomialLogit, LeastSquares},
    design::{Augmented, Chunked, DesignMatrix, MmapMatrix, MmapMatrixF32, Standardized},
    penalty::Mcp,
    solver::{prox_newton_solve_path_timed, solve_path_timed, CdConfig, PathConfig},
    Penalty,
};

/// Compute glmnet-convention per-column scales for any
/// [`DesignMatrix`] backend that already cached `col_sq_norm`. One
/// `columns(&[j])` call per column for the mean (page cache hit on
/// the second pass for mmap; trivial for in-memory backends).
fn compute_design_glmnet_scales<D: DesignMatrix>(d: &D) -> Array1<f64> {
    let n = d.n_samples();
    let p = d.n_features();
    let n_f = n as f64;
    let mut s = Array1::<f64>::ones(p);
    for j in 0..p {
        let col = d.columns(&[j]);
        let mut sum = 0.0_f64;
        for &v in col.iter() {
            sum += v;
        }
        let mean = sum / n_f;
        let sq = d.col_sq_norm(j);
        let var = ((sq - n_f * mean * mean) / n_f).max(0.0);
        let sd = var.sqrt();
        s[j] = if sd > 1e-12 { sd } else { 1.0 };
    }
    s
}

/// LS+MCP path body shared between f64 and f32 mmap entry points.
/// Generic over the opened design; the entry-point shells just
/// validate y / weights, call `Mmap*::open`, and forward here.
#[allow(clippy::too_many_arguments)]
fn mmap_ls_mcp_path_inner<'py, D>(
    py: Python<'py>,
    design: D,
    n_rows: usize,
    n_cols: usize,
    y_arr: Array1<f64>,
    user_weights: Option<Array1<f64>>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    D: DesignMatrix + 'static,
{
    debug_assert_eq!(design.n_samples(), n_rows);
    debug_assert_eq!(design.n_features(), n_cols);

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_design_glmnet_scales(&design))
    } else {
        None
    };

    let mut pen_weights = Array1::<f64>::ones(if fit_intercept { n_cols + 1 } else { n_cols });
    if let Some(uw) = &user_weights {
        for j in 0..n_cols {
            pen_weights[j] = uw[j];
        }
    }
    if fit_intercept {
        pen_weights[n_cols] = 0.0;
    }
    if let Some(scales) = &scales_user {
        for j in 0..n_cols {
            pen_weights[j] /= scales[j];
        }
    }

    let path_cfg = PathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: crate::ls::parse_screening(screening)?,
        p0: 10,
    };
    let datafit = LeastSquares::new(y_arr);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> {
        Box::new(Mcp::with_weights(lam, gamma, pen_weights.clone()))
    };

    let p_eff = if fit_intercept { n_cols + 1 } else { n_cols };
    let (betas_aug, report, times_ns) =
        py.allow_threads(|| match (fit_intercept, scales_user.as_ref()) {
            (false, None) => solve_path_timed(&design, &datafit, make_pen, &path_cfg),
            (false, Some(scales)) => {
                let std_design = Standardized::new(design, scales.clone());
                solve_path_timed(&std_design, &datafit, make_pen, &path_cfg)
            }
            (true, None) => {
                let aug = Augmented::new(design);
                solve_path_timed(&aug, &datafit, make_pen, &path_cfg)
            }
            (true, Some(scales)) => {
                let aug = Augmented::new(design);
                let mut x_scale_eff = Array1::<f64>::ones(p_eff);
                for j in 0..n_cols {
                    x_scale_eff[j] = scales[j];
                }
                let std_design = Standardized::new(aug, x_scale_eff);
                solve_path_timed(&std_design, &datafit, make_pen, &path_cfg)
            }
        });

    let (mut coefs, intercepts) = crate::glm::split_intercept(betas_aug, fit_intercept);
    if let Some(scales) = scales_user.as_ref() {
        for k in 0..coefs.nrows() {
            for j in 0..coefs.ncols() {
                coefs[[k, j]] /= scales[j];
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

/// Validate y length and unpack user weights — shared shell across the
/// four mmap entry points (LS×{f64,f32} and logistic×{f64,f32}).
fn mmap_validate_inputs(
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(Array1<f64>, Option<Array1<f64>>)> {
    let y_arr = y.as_array().to_owned();
    if y_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_rows {}",
            y_arr.len(),
            n_rows
        )));
    }
    let user_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    n_cols
                )));
            }
            Some(arr)
        }
        None => None,
    };
    Ok((y_arr, user_weights))
}

#[pyfunction]
#[pyo3(signature = (
    path, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_mcp_ls_path_mmap<'py>(
    py: Python<'py>,
    path: &str,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    let mmap = MmapMatrix::open(path, n_rows, n_cols)
        .map_err(|e| PyValueError::new_err(format!("MmapMatrix::open failed: {}", e)))?;
    mmap_ls_mcp_path_inner(
        py,
        mmap,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        screening,
        acceleration,
        fit_intercept,
        standardize_x,
    )
}

#[pyfunction]
#[pyo3(signature = (
    path, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_mcp_ls_path_mmap_f32<'py>(
    py: Python<'py>,
    path: &str,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    let mmap = MmapMatrixF32::open(path, n_rows, n_cols)
        .map_err(|e| PyValueError::new_err(format!("MmapMatrixF32::open failed: {}", e)))?;
    mmap_ls_mcp_path_inner(
        py,
        mmap,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        screening,
        acceleration,
        fit_intercept,
        standardize_x,
    )
}

/// Logistic+MCP path body shared between f64 and f32 mmap entry
/// points. Same shape as the LS helper above.
#[allow(clippy::too_many_arguments)]
fn mmap_logistic_mcp_path_inner<'py, D>(
    py: Python<'py>,
    design: D,
    n_rows: usize,
    n_cols: usize,
    y_arr: Array1<f64>,
    user_weights: Option<Array1<f64>>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    D: DesignMatrix + 'static,
{
    debug_assert_eq!(design.n_samples(), n_rows);
    debug_assert_eq!(design.n_features(), n_cols);

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_design_glmnet_scales(&design))
    } else {
        None
    };

    let mut pen_weights = Array1::<f64>::ones(if fit_intercept { n_cols + 1 } else { n_cols });
    if let Some(uw) = &user_weights {
        for j in 0..n_cols {
            pen_weights[j] = uw[j];
        }
    }
    if fit_intercept {
        pen_weights[n_cols] = 0.0;
    }
    if let Some(scales) = &scales_user {
        for j in 0..n_cols {
            pen_weights[j] /= scales[j];
        }
    }

    let glm = BinomialLogit::new(y_arr);
    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());
    let make_pen = move |lam: f64| -> Box<dyn Penalty> {
        Box::new(Mcp::with_weights(lam, gamma, pen_weights.clone()))
    };

    let p_eff = if fit_intercept { n_cols + 1 } else { n_cols };
    let (betas_aug, report, times_ns) =
        py.allow_threads(|| match (fit_intercept, scales_user.as_ref()) {
            (false, None) => prox_newton_solve_path_timed(
                &design,
                &glm,
                make_pen,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            ),
            (false, Some(scales)) => {
                let std_design = Standardized::new(design, scales.clone());
                prox_newton_solve_path_timed(
                    &std_design,
                    &glm,
                    make_pen,
                    n_lambdas,
                    lambda_min_ratio,
                    lambdas_vec,
                    &cd_cfg,
                    max_outer,
                    outer_tol,
                )
            }
            (true, None) => {
                let aug = Augmented::new(design);
                prox_newton_solve_path_timed(
                    &aug,
                    &glm,
                    make_pen,
                    n_lambdas,
                    lambda_min_ratio,
                    lambdas_vec,
                    &cd_cfg,
                    max_outer,
                    outer_tol,
                )
            }
            (true, Some(scales)) => {
                let aug = Augmented::new(design);
                let mut x_scale_eff = Array1::<f64>::ones(p_eff);
                for j in 0..n_cols {
                    x_scale_eff[j] = scales[j];
                }
                let std_design = Standardized::new(aug, x_scale_eff);
                prox_newton_solve_path_timed(
                    &std_design,
                    &glm,
                    make_pen,
                    n_lambdas,
                    lambda_min_ratio,
                    lambdas_vec,
                    &cd_cfg,
                    max_outer,
                    outer_tol,
                )
            }
        });

    let (mut coefs, intercepts) = crate::glm::split_intercept(betas_aug, fit_intercept);
    if let Some(scales) = scales_user.as_ref() {
        for k in 0..coefs.nrows() {
            for j in 0..coefs.ncols() {
                coefs[[k, j]] /= scales[j];
            }
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;
    info.set_item("times_ns", times_ns)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    path, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_mcp_path_mmap<'py>(
    py: Python<'py>,
    path: &str,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    crate::glm::validate_y_binary(y_arr.view())?;
    let mmap = MmapMatrix::open(path, n_rows, n_cols)
        .map_err(|e| PyValueError::new_err(format!("MmapMatrix::open failed: {}", e)))?;
    mmap_logistic_mcp_path_inner(
        py,
        mmap,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
    )
}

#[pyfunction]
#[pyo3(signature = (
    path, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_mcp_path_mmap_f32<'py>(
    py: Python<'py>,
    path: &str,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    crate::glm::validate_y_binary(y_arr.view())?;
    let mmap = MmapMatrixF32::open(path, n_rows, n_cols)
        .map_err(|e| PyValueError::new_err(format!("MmapMatrixF32::open failed: {}", e)))?;
    mmap_logistic_mcp_path_inner(
        py,
        mmap,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
    )
}

// ---- Chunked (row-block streaming) entry points ------------------------

/// Open a list of `(path, n_rows)` pairs as `Chunked<MmapMatrix>` (f64).
fn open_chunked_f64(chunks: Vec<(String, usize)>, n_cols: usize) -> PyResult<Chunked<MmapMatrix>> {
    if chunks.is_empty() {
        return Err(PyValueError::new_err("chunks list must not be empty"));
    }
    let mut opened = Vec::with_capacity(chunks.len());
    for (i, (path, n_rows)) in chunks.into_iter().enumerate() {
        let mmap = MmapMatrix::open(&path, n_rows, n_cols).map_err(|e| {
            PyValueError::new_err(format!(
                "MmapMatrix::open failed for chunk {i} ({path}): {e}"
            ))
        })?;
        opened.push(mmap);
    }
    Ok(Chunked::new(opened))
}

/// Same as `open_chunked_f64` but for `Chunked<MmapMatrixF32>`.
fn open_chunked_f32(
    chunks: Vec<(String, usize)>,
    n_cols: usize,
) -> PyResult<Chunked<MmapMatrixF32>> {
    if chunks.is_empty() {
        return Err(PyValueError::new_err("chunks list must not be empty"));
    }
    let mut opened = Vec::with_capacity(chunks.len());
    for (i, (path, n_rows)) in chunks.into_iter().enumerate() {
        let mmap = MmapMatrixF32::open(&path, n_rows, n_cols).map_err(|e| {
            PyValueError::new_err(format!(
                "MmapMatrixF32::open failed for chunk {i} ({path}): {e}"
            ))
        })?;
        opened.push(mmap);
    }
    Ok(Chunked::new(opened))
}

#[pyfunction]
#[pyo3(signature = (
    chunks, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_mcp_ls_path_chunked<'py>(
    py: Python<'py>,
    chunks: Vec<(String, usize)>,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let design = open_chunked_f64(chunks, n_cols)?;
    let n_rows = design.n_samples();
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    mmap_ls_mcp_path_inner(
        py,
        design,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        screening,
        acceleration,
        fit_intercept,
        standardize_x,
    )
}

#[pyfunction]
#[pyo3(signature = (
    chunks, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_mcp_ls_path_chunked_f32<'py>(
    py: Python<'py>,
    chunks: Vec<(String, usize)>,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let design = open_chunked_f32(chunks, n_cols)?;
    let n_rows = design.n_samples();
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    mmap_ls_mcp_path_inner(
        py,
        design,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        screening,
        acceleration,
        fit_intercept,
        standardize_x,
    )
}

#[pyfunction]
#[pyo3(signature = (
    chunks, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_mcp_path_chunked<'py>(
    py: Python<'py>,
    chunks: Vec<(String, usize)>,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let design = open_chunked_f64(chunks, n_cols)?;
    let n_rows = design.n_samples();
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    crate::glm::validate_y_binary(y_arr.view())?;
    mmap_logistic_mcp_path_inner(
        py,
        design,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
    )
}

#[pyfunction]
#[pyo3(signature = (
    chunks, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_mcp_path_chunked_f32<'py>(
    py: Python<'py>,
    chunks: Vec<(String, usize)>,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let design = open_chunked_f32(chunks, n_cols)?;
    let n_rows = design.n_samples();
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    crate::glm::validate_y_binary(y_arr.view())?;
    mmap_logistic_mcp_path_inner(
        py,
        design,
        n_rows,
        n_cols,
        y_arr,
        user_weights,
        gamma,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
    )
}
