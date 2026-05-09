//! Python bindings for skein-core.
//!
//! Compiled extension lives at `skein_glm._core` on the Python side
//! (the PyPI distribution name is `skein-glm` because the `skein`
//! name is taken on PyPI by an unrelated YARN-interface library).
//! The Rust crate names (`skein-core`, `skein-py`) keep the short
//! form.
//!
//! Exposes the smallest surface needed for the Python
//! `skein_glm.estimators` layer: build a problem (design + datafit +
//! penalty), solve it (or solve along a λ-path), return β plus an
//! info dict.

// clippy 1.95's `useless_conversion` lint is a false positive against
// PyO3 0.22's macro-generated `PyResult<...>` returns — it points at the
// opening paren of the return type but there's no `.into()` to remove
// without breaking the signature. Suppress at module level.
#![allow(clippy::useless_conversion)]

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ndarray::{Array2, ArrayView1};
use skein_core::{
    datafit::{BinomialLogit, CoxPH, GlmDatafit, LeastSquares, PoissonLog},
    design::{
        Augmented, Chunked, DenseMatrix, DesignMatrix, MmapMatrix, MmapMatrixF32, SparseCSC,
        Standardized,
    },
    groups::Groups,
    penalty::{ElasticNet, GroupElasticNet, GroupLasso, GroupPenalty, Mcp, Scad, SparseGroupLasso},
    solver::{
        cd_solve, prox_newton_block_solve_path, prox_newton_solve_path, solve_block_path,
        solve_block_path_lla, solve_path, surrogate_sparse_group_mcp, surrogate_weights_group_mcp,
        BlockPathConfig, CdConfig, PathConfig, Screening,
    },
    standardize::{
        destandardize_path, rescale_weights_for_standardize, standardize, StandardizeConfig,
    },
    Penalty,
};

type PathOutput<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyDict>,
);

#[pyfunction]
#[pyo3(signature = (x, y, lambda_, gamma, *, weights=None, max_iter=100, tol=1e-6))]
#[allow(clippy::too_many_arguments)]
fn solve_mcp_ls<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    lambda_: f64,
    gamma: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p = x_arr.ncols();

    let design = DenseMatrix::new(x_arr);
    let datafit = LeastSquares::new(y_arr);
    let penalty: Box<dyn Penalty> = match weights {
        Some(w) => Box::new(Mcp::with_weights(lambda_, gamma, w.as_array().to_owned())),
        None => Box::new(Mcp::new(lambda_, gamma, p)),
    };

    let cfg = CdConfig {
        max_iter,
        tol,
        acceleration: Some(5),
    };
    let (beta, report) = cd_solve(&design, &datafit, &*penalty, &cfg);

    let info = PyDict::new_bound(py);
    info.set_item("iter", report.iter)?;
    info.set_item("converged", report.converged)?;
    info.set_item("final_obj", report.final_obj)?;
    Ok((beta.into_pyarray_bound(py), info))
}

#[pyfunction]
#[pyo3(signature = (x, y, lambda_, a, *, weights=None, max_iter=100, tol=1e-6))]
#[allow(clippy::too_many_arguments)]
fn solve_scad_ls<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    lambda_: f64,
    a: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p = x_arr.ncols();

    let design = DenseMatrix::new(x_arr);
    let datafit = LeastSquares::new(y_arr);
    let penalty: Box<dyn Penalty> = match weights {
        Some(w) => Box::new(Scad::with_weights(lambda_, a, w.as_array().to_owned())),
        None => Box::new(Scad::new(lambda_, a, p)),
    };

    let cfg = CdConfig {
        max_iter,
        tol,
        acceleration: Some(5),
    };
    let (beta, report) = cd_solve(&design, &datafit, &*penalty, &cfg);

    let info = PyDict::new_bound(py);
    info.set_item("iter", report.iter)?;
    info.set_item("converged", report.converged)?;
    info.set_item("final_obj", report.final_obj)?;
    Ok((beta.into_pyarray_bound(py), info))
}

fn parse_screening(s: &str) -> PyResult<Screening> {
    match s {
        "off" => Ok(Screening::Off),
        "strong" => Ok(Screening::Strong),
        "gap_safe" => Ok(Screening::GapSafe),
        other => Err(PyValueError::new_err(format!(
            "screening must be one of 'off', 'strong', 'gap_safe' (got '{}')",
            other
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_path_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    screening: &str,
    fit_intercept: bool,
    standardize_x: bool,
    make_penalty: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, Array1<f64>) -> Box<dyn Penalty>,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p = x_arr.ncols();

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

    let std_cfg = StandardizeConfig {
        center_x: fit_intercept,
        scale_x: standardize_x,
        fit_intercept,
    };
    let (xs, ys, stats) = standardize(x_arr.view(), y_arr.view(), &std_cfg);
    let weights_std = rescale_weights_for_standardize(weights_orig.view(), &stats);

    let path_cfg = PathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: parse_screening(screening)?,
    };

    let design = DenseMatrix::new(xs);
    let datafit = LeastSquares::new(ys);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, weights_std.clone()) };
    let (betas_std, report) = solve_path(&design, &datafit, make_pen, &path_cfg);
    let (coefs, intercepts) = destandardize_path(betas_std.view(), &stats);

    let info = PyDict::new_bound(py);
    info.set_item("iters", report.iters)?;
    info.set_item("converged", report.converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, gamma=3.0,
    lambdas=None,
    n_lambdas=100,
    lambda_min_ratio=1e-3,
    weights=None,
    max_iter=100,
    tol=1e-6,
    screening="strong",
    acceleration=Some(5),
    fit_intercept=true,
    standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_mcp_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
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
) -> PyResult<PathOutput<'py>> {
    build_path_outputs(
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
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, a=3.7,
    lambdas=None,
    n_lambdas=100,
    lambda_min_ratio=1e-3,
    weights=None,
    max_iter=100,
    tol=1e-6,
    screening="strong",
    acceleration=Some(5),
    fit_intercept=true,
    standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_scad_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    a: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_path_outputs(
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
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, alpha=0.5,
    lambdas=None,
    n_lambdas=100,
    lambda_min_ratio=1e-3,
    weights=None,
    max_iter=100,
    tol=1e-6,
    screening="strong",
    acceleration=Some(5),
    fit_intercept=true,
    standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_elastic_net_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    alpha: f64,
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
) -> PyResult<PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_path_outputs(
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
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(ElasticNet::with_weights(lam, alpha, w)),
    )
}

// ---------------------------------------------------------------------
// Group-penalty path solvers
// ---------------------------------------------------------------------

/// Convert a length-p label vector into a `Groups` (CSR-style). Group
/// labels must form a contiguous range starting at 0.
fn groups_from_labels(labels: &[i64]) -> PyResult<Groups> {
    let p = labels.len();
    if p == 0 {
        return Ok(Groups::singletons(0));
    }
    let max_label = *labels.iter().max().unwrap();
    if labels.iter().any(|&v| v < 0) {
        return Err(PyValueError::new_err("groups labels must be non-negative"));
    }
    let n_groups = (max_label as usize) + 1;
    let mut counts = vec![0usize; n_groups];
    for &lab in labels {
        counts[lab as usize] += 1;
    }
    if counts.contains(&0) {
        return Err(PyValueError::new_err(
            "groups labels must form 0..n_groups (no empty groups)",
        ));
    }
    let mut ptr = vec![0usize; n_groups + 1];
    for g in 0..n_groups {
        ptr[g + 1] = ptr[g] + counts[g];
    }
    let mut idx = vec![0usize; p];
    let mut filled = vec![0usize; n_groups];
    for (j, &lab) in labels.iter().enumerate() {
        let g = lab as usize;
        idx[ptr[g] + filled[g]] = j;
        filled[g] += 1;
    }
    Groups::from_csr(ptr, idx).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn build_block_path_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn GroupPenalty>,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p = x_arr.ncols();

    let labels = groups_labels.as_array().to_owned().to_vec();
    if labels.len() != p {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels.len(),
            p
        )));
    }
    let groups = groups_from_labels(&labels)?;
    let n_groups = groups.n_groups();

    let weights_orig = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups
                )));
            }
            arr
        }
        None => ndarray::Array1::ones(n_groups),
    };

    let std_cfg = StandardizeConfig {
        center_x: fit_intercept,
        scale_x: standardize_x,
        fit_intercept,
    };
    let (xs, ys, stats) = standardize(x_arr.view(), y_arr.view(), &std_cfg);

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: parse_screening(screening)?,
        parallel,
    };

    let design = DenseMatrix::new(xs);
    let datafit = LeastSquares::new(ys);
    let make_pen =
        move |lam: f64| -> Box<dyn GroupPenalty> { make_inner(lam, weights_orig.clone()) };
    let (betas_std, report) = solve_block_path(&design, &datafit, make_pen, &groups, &block_cfg);
    let (coefs, intercepts) = destandardize_path(betas_std.view(), &stats);

    let info = PyDict::new_bound(py);
    info.set_item("iters", report.iters)?;
    info.set_item("converged", report.converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        ndarray::Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_block_path_lla_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>>
where
    F: Fn(ArrayView1<f64>, &Groups, f64) -> Box<dyn GroupPenalty>,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p = x_arr.ncols();

    let labels = groups_labels.as_array().to_owned().to_vec();
    if labels.len() != p {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels.len(),
            p
        )));
    }
    let groups = groups_from_labels(&labels)?;
    let n_groups = groups.n_groups();

    let base_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups
                )));
            }
            arr
        }
        None => ndarray::Array1::ones(n_groups),
    };

    let std_cfg = StandardizeConfig {
        center_x: fit_intercept,
        scale_x: standardize_x,
        fit_intercept,
    };
    let (xs, ys, stats) = standardize(x_arr.view(), y_arr.view(), &std_cfg);

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: parse_screening(screening)?,
        parallel,
    };

    let design = DenseMatrix::new(xs);
    let datafit = LeastSquares::new(ys);
    let (betas_std, report) = solve_block_path_lla(
        &design,
        &datafit,
        base_weights,
        make_inner,
        &groups,
        &block_cfg,
        max_outer,
        outer_tol,
    );
    let (coefs, intercepts) = destandardize_path(betas_std.view(), &stats);

    let info = PyDict::new_bound(py);
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        ndarray::Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_group_lasso_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_block_path_outputs(
        py,
        x,
        y,
        groups,
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
    x, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_sparse_group_lasso_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_block_path_outputs(
        py,
        x,
        y,
        groups,
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
        move |lam, w| Box::new(SparseGroupLasso::with_weights(lam, alpha, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_group_elastic_net_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_block_path_outputs(
        py,
        x,
        y,
        groups,
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

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_group_mcp_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    let labels_owned = groups.as_array().to_owned();
    let groups_obj = groups_from_labels(&labels_owned.to_vec())?;
    let n_groups = groups_obj.n_groups();
    let _ = groups_obj; // groups_obj is rebuilt inside the helper; we just validated.

    let base_weights = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => ndarray::Array1::ones(n_groups),
    };
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_mcp(beta, g, lam, gamma, base_weights.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };
    build_block_path_lla_outputs(
        py,
        x,
        y,
        groups,
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
    x, y, groups, *, gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    coord_weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_sparse_group_mcp_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let labels_owned = groups.as_array().to_owned();
    let groups_obj = groups_from_labels(&labels_owned.to_vec())?;
    let n_groups = groups_obj.n_groups();
    let p = x.as_array().ncols();

    let base_group = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => ndarray::Array1::ones(n_groups),
    };
    let base_coord = match &coord_weights {
        Some(w) => w.as_array().to_owned(),
        None => ndarray::Array1::ones(p),
    };
    if base_group.len() != n_groups {
        return Err(PyValueError::new_err(format!(
            "weights length {} does not match n_groups {}",
            base_group.len(),
            n_groups
        )));
    }
    if base_coord.len() != p {
        return Err(PyValueError::new_err(format!(
            "coord_weights length {} does not match n_features {}",
            base_coord.len(),
            p
        )));
    }
    let _ = groups_obj;

    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let (gw, cw) = surrogate_sparse_group_mcp(
            beta,
            g,
            lam,
            gamma,
            alpha,
            base_group.view(),
            base_coord.view(),
        );
        Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
    };
    build_block_path_lla_outputs(
        py,
        x,
        y,
        groups,
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

// ---------------------------------------------------------------------
// Logistic regression (binomial logit) via prox-Newton
// ---------------------------------------------------------------------

/// Augment X with a column of ones at the right edge (the intercept column).
fn append_intercept_column(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((n, p + 1));
    out.slice_mut(ndarray::s![.., ..p]).assign(x);
    out.column_mut(p).fill(1.0);
    out
}

/// Build the per-feature penalty weight vector for the (possibly
/// intercept-augmented) feature space. User weights are `Some` only for
/// the original `p` features; the augmented intercept column gets weight
/// 0 so it stays unpenalized.
fn build_logistic_penalty_weights(
    user_weights: &Option<ndarray::Array1<f64>>,
    p_user: usize,
    fit_intercept: bool,
) -> ndarray::Array1<f64> {
    let p_eff = if fit_intercept { p_user + 1 } else { p_user };
    let mut w = ndarray::Array1::<f64>::ones(p_eff);
    if let Some(uw) = user_weights {
        for j in 0..p_user {
            w[j] = uw[j];
        }
    }
    if fit_intercept {
        w[p_user] = 0.0;
    }
    w
}

/// Split (coefs_aug, intercepts) from a (n_lambdas, p_eff) matrix when
/// `fit_intercept`: last column becomes `intercepts`, rest stays as
/// `coefs`.
fn split_intercept(
    betas_aug: ndarray::Array2<f64>,
    fit_intercept: bool,
) -> (ndarray::Array2<f64>, ndarray::Array1<f64>) {
    let n_lams = betas_aug.nrows();
    if !fit_intercept {
        let intercepts = ndarray::Array1::<f64>::zeros(n_lams);
        return (betas_aug, intercepts);
    }
    let p_eff = betas_aug.ncols();
    let p_user = p_eff - 1;
    let coefs = betas_aug.slice(ndarray::s![.., ..p_user]).to_owned();
    let intercepts = betas_aug.column(p_user).to_owned();
    (coefs, intercepts)
}

/// Validate that y ∈ {0, 1} (logistic regression).
fn validate_y_binary(y: ndarray::ArrayView1<'_, f64>) -> PyResult<()> {
    for &v in y.iter() {
        if v != 0.0 && v != 1.0 {
            return Err(PyValueError::new_err(
                "logistic regression requires y ∈ {0, 1}",
            ));
        }
    }
    Ok(())
}

/// Validate that y ≥ 0 and finite (Poisson regression). `!is_finite`
/// catches NaN and ±∞; `v < 0.0` catches negatives (NaN < 0.0 is false,
/// but the finiteness check has already rejected NaN).
fn validate_y_nonneg(y: ndarray::ArrayView1<'_, f64>) -> PyResult<()> {
    for &v in y.iter() {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(
                "Poisson regression requires y ≥ 0 (finite)",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_glm_path_outputs<'py, F, V, G>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
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
    validate_y: V,
    make_glm: G,
    make_penalty: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty>,
    V: Fn(ndarray::ArrayView1<'_, f64>) -> PyResult<()>,
    G: FnOnce(ndarray::Array1<f64>) -> Box<dyn GlmDatafit>,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p_user = x_arr.ncols();
    let n = x_arr.nrows();
    if y_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_samples {}",
            y_arr.len(),
            n
        )));
    }
    validate_y(y_arr.view())?;

    let user_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p_user {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    p_user
                )));
            }
            Some(arr)
        }
        None => None,
    };

    // Compute per-column scales BEFORE intercept augmentation (intercept
    // column is never scaled).
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };

    // Augment X + per-feature weights for the intercept.
    let x_eff = if fit_intercept {
        append_intercept_column(&x_arr)
    } else {
        x_arr
    };
    let mut pen_weights = build_logistic_penalty_weights(&user_weights, p_user, fit_intercept);
    // Penalty weights live in original space; standardization sends
    // β̃ = s · β, so the standardized-space penalty is `(w_j / s_j) |β̃_j|`.
    // Intercept entry stays 0.
    if let Some(scales) = &scales_user {
        for j in 0..p_user {
            pen_weights[j] /= scales[j];
        }
    }

    let design = DenseMatrix::new(x_eff);
    let glm = make_glm(y_arr);

    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(design.n_features());
            for j in 0..p_user {
                x_scale_eff[j] = scales[j];
            }
            // Intercept column (last) stays at 1.0.
            let std_design = Standardized::new(design, x_scale_eff);
            prox_newton_solve_path(
                &std_design,
                &*glm,
                make_pen,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => prox_newton_solve_path(
            &design,
            &*glm,
            make_pen,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
    if let Some(scales) = scales_user.as_ref() {
        // β_orig = β̃ / s for the non-intercept columns.
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        ndarray::Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    a: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

// ---------------------------------------------------------------------
// Logistic regression + group / sparse-group penalties (M3.3)
// ---------------------------------------------------------------------

/// Append the intercept column to a label vector by adding it as a new
/// singleton group (so it sits in `groups[p]` with label `n_groups`).
fn append_intercept_group(labels: &[i64], n_groups: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(labels.len() + 1);
    out.extend_from_slice(labels);
    out.push(n_groups as i64);
    out
}

/// Build per-group L2 weights for the intercept-augmented group set.
/// User group weights are `Some` only for the original `n_groups`; the
/// new singleton group gets weight 0.
fn build_logistic_group_weights(
    user_weights: &Option<ndarray::Array1<f64>>,
    n_groups_user: usize,
    fit_intercept: bool,
) -> ndarray::Array1<f64> {
    let n_eff = if fit_intercept {
        n_groups_user + 1
    } else {
        n_groups_user
    };
    let mut w = ndarray::Array1::<f64>::ones(n_eff);
    if let Some(uw) = user_weights {
        for g in 0..n_groups_user {
            w[g] = uw[g];
        }
    }
    if fit_intercept {
        w[n_groups_user] = 0.0;
    }
    w
}

/// Build per-coord L1 weights for sparse-group penalties on the
/// intercept-augmented feature space. The intercept column gets weight 0.
fn build_logistic_coord_weights(
    user_coord_weights: &Option<ndarray::Array1<f64>>,
    p_user: usize,
    fit_intercept: bool,
) -> ndarray::Array1<f64> {
    let p_eff = if fit_intercept { p_user + 1 } else { p_user };
    let mut w = ndarray::Array1::<f64>::ones(p_eff);
    if let Some(uw) = user_coord_weights {
        for j in 0..p_user {
            w[j] = uw[j];
        }
    }
    if fit_intercept {
        w[p_user] = 0.0;
    }
    w
}

#[allow(clippy::too_many_arguments)]
fn build_glm_block_path_outputs<'py, F, V, G>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
    weights: Option<PyReadonlyArray1<f64>>,
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
    validate_y: V,
    make_glm: G,
    make_inner: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty>,
    V: Fn(ndarray::ArrayView1<'_, f64>) -> PyResult<()>,
    G: FnOnce(ndarray::Array1<f64>) -> Box<dyn GlmDatafit>,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let p_user = x_arr.ncols();
    let n = x_arr.nrows();
    if y_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_samples {}",
            y_arr.len(),
            n
        )));
    }
    validate_y(y_arr.view())?;

    let labels_user = groups_labels.as_array().to_owned().to_vec();
    if labels_user.len() != p_user {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels_user.len(),
            p_user
        )));
    }
    let n_groups_user = groups_from_labels(&labels_user)?.n_groups();

    let user_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups_user {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups_user
                )));
            }
            Some(arr)
        }
        None => None,
    };

    // Compute per-column scales BEFORE intercept augmentation. The
    // group penalty is applied in standardized space, so per-group
    // weights stay unchanged (matches the LS sparse-group standardize
    // convention at lib.rs:2515-2517).
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };

    // Augment X + group labels + per-group weights for the intercept.
    let x_eff = if fit_intercept {
        append_intercept_column(&x_arr)
    } else {
        x_arr
    };
    let labels_eff = if fit_intercept {
        append_intercept_group(&labels_user, n_groups_user)
    } else {
        labels_user
    };
    let groups = groups_from_labels(&labels_eff)?;
    let group_w_eff = build_logistic_group_weights(&user_weights, n_groups_user, fit_intercept);

    let design = DenseMatrix::new(x_eff);
    let glm = make_glm(y_arr);

    let group_w_for_closure = group_w_eff.clone();
    let make_inner_wrapped =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            make_inner(beta, g, lam, &group_w_for_closure)
        };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(design.n_features());
            for j in 0..p_user {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(design, x_scale_eff);
            prox_newton_block_solve_path(
                &std_design,
                &*glm,
                group_w_eff,
                make_inner_wrapped,
                &groups,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => prox_newton_block_solve_path(
            &design,
            &*glm,
            group_w_eff,
            make_inner_wrapped,
            &groups,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        ndarray::Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        // Plain group lasso: ignore β, build weighted GroupLasso.
        |_beta, _groups, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        // Group MCP: LLA surrogate weights then weighted GroupLasso.
        move |beta, g, lam, group_w| {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w.view());
            Box::new(GroupLasso::with_weights(lam, w))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_sparse_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |_beta, _groups, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_sparse_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let p_user = x.as_array().ncols();
    let user_coord = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p_user {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    p_user
                )));
            }
            Some(arr)
        }
        None => None,
    };
    let coord_w_eff = build_logistic_coord_weights(&user_coord, p_user, fit_intercept);

    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                group_w.view(),
                coord_w_eff.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}

// ---------------------------------------------------------------------
// Poisson regression (log link) via prox-Newton (M3.4)
// ---------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    x, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    a: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        |_beta, _groups, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |beta, g, lam, group_w| {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w.view());
            Box::new(GroupLasso::with_weights(lam, w))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_sparse_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |_beta, _groups, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_sparse_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let p_user = x.as_array().ncols();
    let user_coord = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p_user {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    p_user
                )));
            }
            Some(arr)
        }
        None => None,
    };
    let coord_w_eff = build_logistic_coord_weights(&user_coord, p_user, fit_intercept);

    build_glm_block_path_outputs(
        py,
        x,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                group_w.view(),
                coord_w_eff.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}

// ---------------------------------------------------------------------
// Cox proportional hazards (Breslow ties) via prox-Newton (M3.5)
// ---------------------------------------------------------------------

/// Validate Cox outcomes: `time` finite ≥ 0, `event ∈ {0, 1}`, at least
/// one event observed. Length consistency is checked by the caller.
fn validate_cox_outcomes(
    time: ndarray::ArrayView1<'_, f64>,
    event: ndarray::ArrayView1<'_, f64>,
) -> PyResult<()> {
    let mut n_events = 0_usize;
    for i in 0..time.len() {
        let t = time[i];
        if !t.is_finite() || t < 0.0 {
            return Err(PyValueError::new_err("Cox PH requires time ≥ 0 (finite)"));
        }
        let d = event[i];
        if d != 0.0 && d != 1.0 {
            return Err(PyValueError::new_err("Cox PH requires event ∈ {0, 1}"));
        }
        if d > 0.5 {
            n_events += 1;
        }
    }
    if n_events == 0 {
        return Err(PyValueError::new_err(
            "Cox PH requires at least one event (event = 1)",
        ));
    }
    Ok(())
}

/// Common Cox path scaffold for scalar penalties. Cox has no intercept
/// (the baseline hazard absorbs constants), so β has length `p_user`
/// and we always return zero intercepts to keep the 4-tuple
/// `(coefs, intercepts, lambdas, info)` shape consistent with logistic /
/// Poisson on the Python side.
#[allow(clippy::too_many_arguments)]
fn build_cox_path_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    make_penalty: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty>,
{
    let x_arr = x.as_array().to_owned();
    let time_arr = time.as_array().to_owned();
    let event_arr = event.as_array().to_owned();
    let p = x_arr.ncols();
    let n = x_arr.nrows();
    if time_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "time length {} does not match n_samples {}",
            time_arr.len(),
            n
        )));
    }
    if event_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "event length {} does not match n_samples {}",
            event_arr.len(),
            n
        )));
    }
    validate_cox_outcomes(time_arr.view(), event_arr.view())?;

    let mut pen_weights = match weights {
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

    // Cox has no intercept (the baseline hazard absorbs it), so we wrap
    // the user matrix directly — no augmentation step.
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };
    if let Some(scales) = &scales_user {
        for j in 0..p {
            pen_weights[j] /= scales[j];
        }
    }

    let design = DenseMatrix::new(x_arr);
    let glm = CoxPH::new(time_arr, event_arr);

    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (mut betas, report) = match scales_user.as_ref() {
        Some(scales) => {
            let std_design = Standardized::new(design, scales.clone());
            prox_newton_solve_path(
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
        None => prox_newton_solve_path(
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
    };

    if let Some(scales) = scales_user.as_ref() {
        for k in 0..betas.nrows() {
            for j in 0..betas.ncols() {
                betas[[k, j]] /= scales[j];
            }
        }
    }

    let n_lams = report.lambdas.len();
    let intercepts = Array1::<f64>::zeros(n_lams);

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;

    Ok((
        betas.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_cox_block_path_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    make_inner: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty>,
{
    let x_arr = x.as_array().to_owned();
    let time_arr = time.as_array().to_owned();
    let event_arr = event.as_array().to_owned();
    let p = x_arr.ncols();
    let n = x_arr.nrows();
    if time_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "time length {} does not match n_samples {}",
            time_arr.len(),
            n
        )));
    }
    if event_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "event length {} does not match n_samples {}",
            event_arr.len(),
            n
        )));
    }
    validate_cox_outcomes(time_arr.view(), event_arr.view())?;

    let labels = groups_labels.as_array().to_owned().to_vec();
    if labels.len() != p {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels.len(),
            p
        )));
    }
    let groups = groups_from_labels(&labels)?;
    let n_groups = groups.n_groups();

    let group_w = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups
                )));
            }
            arr
        }
        None => Array1::ones(n_groups),
    };

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };

    let design = DenseMatrix::new(x_arr);
    let glm = CoxPH::new(time_arr, event_arr);

    let group_w_for_closure = group_w.clone();
    let make_inner_wrapped =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            make_inner(beta, g, lam, &group_w_for_closure)
        };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (mut betas, report) = match scales_user.as_ref() {
        Some(scales) => {
            let std_design = Standardized::new(design, scales.clone());
            prox_newton_block_solve_path(
                &std_design,
                &glm,
                group_w,
                make_inner_wrapped,
                &groups,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => prox_newton_block_solve_path(
            &design,
            &glm,
            group_w,
            make_inner_wrapped,
            &groups,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    if let Some(scales) = scales_user.as_ref() {
        for k in 0..betas.nrows() {
            for j in 0..betas.ncols() {
                betas[[k, j]] /= scales[j];
            }
        }
    }

    let n_lams = report.lambdas.len();
    let intercepts = Array1::<f64>::zeros(n_lams);

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;

    Ok((
        betas.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_path_outputs(
        py,
        x,
        time,
        event,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    a: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_path_outputs(
        py,
        x,
        time,
        event,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_block_path_outputs(
        py,
        x,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        |_beta, _groups, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_block_path_outputs(
        py,
        x,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |beta, g, lam, group_w| {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w.view());
            Box::new(GroupLasso::with_weights(lam, w))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_sparse_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_block_path_outputs(
        py,
        x,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |_beta, _groups, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_sparse_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let p = x.as_array().ncols();
    let coord_w = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != p {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    p
                )));
            }
            arr
        }
        None => Array1::ones(p),
    };

    build_cox_block_path_outputs(
        py,
        x,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                group_w.view(),
                coord_w.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}

// ---------------------------------------------------------------------
// Sparse design matrix support (M4.2)
// ---------------------------------------------------------------------
//
// scipy.sparse.csc_matrix arrives as three numpy arrays (`data`,
// `indices`, `indptr`) plus a shape. We build a `SparseCSC` from those
// and route the standard solver scaffolding through it. Standardization
// would densify the matrix, so `standardize_x = True` is rejected by
// the Python layer before reaching here. Intercept handling for sparse
// uses column augmentation (1s column with penalty weight 0) — the
// same scheme the GLM dense path already uses — rather than the
// dense LS path's centering trick, since centering also densifies.

/// Read a scipy.sparse.csc_matrix's `(data, indices, indptr)` arrays
/// into an owned `SparseCSC`. `indices` and `indptr` arrive as `i64`
/// (or `i32` widened by Python before the call); we validate
/// non-negativity and convert to `usize`.
fn read_csc_arrays(
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
) -> PyResult<SparseCSC> {
    let data_arr = data.as_array().to_owned();
    let indices_raw = indices.as_array();
    let indptr_raw = indptr.as_array();

    if indptr_raw.len() != n_cols + 1 {
        return Err(PyValueError::new_err(format!(
            "indptr length {} does not match n_cols + 1 = {}",
            indptr_raw.len(),
            n_cols + 1
        )));
    }
    if indices_raw.len() != data_arr.len() {
        return Err(PyValueError::new_err(format!(
            "indices length {} does not match data length {}",
            indices_raw.len(),
            data_arr.len()
        )));
    }

    let mut indices_usize = Vec::with_capacity(indices_raw.len());
    for (k, &v) in indices_raw.iter().enumerate() {
        if v < 0 {
            return Err(PyValueError::new_err(format!(
                "indices[{}] = {} is negative",
                k, v
            )));
        }
        indices_usize.push(v as usize);
    }
    let mut indptr_usize = Vec::with_capacity(indptr_raw.len());
    for (k, &v) in indptr_raw.iter().enumerate() {
        if v < 0 {
            return Err(PyValueError::new_err(format!(
                "indptr[{}] = {} is negative",
                k, v
            )));
        }
        indptr_usize.push(v as usize);
    }

    Ok(SparseCSC::new(
        n_rows,
        data_arr,
        Array1::from(indices_usize),
        Array1::from(indptr_usize),
    ))
}

/// Append a dense column of 1s as the rightmost column of a CSC matrix.
/// Used to add an intercept feature for sparse paths. Cost: `n_rows`
/// extra non-zeros (so the augmented matrix is no longer fully sparse,
/// but only one column densifies).
fn append_intercept_to_csc(csc: SparseCSC) -> SparseCSC {
    let n_rows = csc.n_samples();
    let nnz_old = csc.nnz();
    let mut new_data = Vec::with_capacity(nnz_old + n_rows);
    new_data.extend(csc.data().iter());
    new_data.extend(std::iter::repeat_n(1.0_f64, n_rows));
    let mut new_indices = Vec::with_capacity(nnz_old + n_rows);
    new_indices.extend(csc.indices().iter());
    new_indices.extend(0..n_rows);
    let mut new_indptr: Vec<usize> = csc.indptr().to_vec();
    new_indptr.push(nnz_old + n_rows);
    SparseCSC::new(
        n_rows,
        Array1::from(new_data),
        Array1::from(new_indices),
        Array1::from(new_indptr),
    )
}

/// glmnet-style per-column std for a `SparseCSC`:
/// `s_j = sqrt((‖X[:,j]‖² − n · x̄_j²) / n)`. Constant columns (s ≈ 0)
/// fall back to `1.0` so downstream `Standardized<...>` doesn't divide
/// by zero. Computed in O(nnz) time directly off the CSC arrays.
fn compute_csc_glmnet_scales(csc: &SparseCSC) -> ndarray::Array1<f64> {
    let n = csc.n_samples();
    let n_f = n as f64;
    let p = csc.n_features();
    let data = csc.data();
    let indptr = csc.indptr();
    let mut scales = ndarray::Array1::<f64>::ones(p);
    for j in 0..p {
        let start = indptr[j];
        let end = indptr[j + 1];
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for k in start..end {
            let v = data[k];
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n_f;
        // Var = (Σ x_ij² − n·x̄²) / n; clamp at 0 for FP safety.
        let var = ((sum_sq - n_f * mean * mean) / n_f).max(0.0);
        let s = var.sqrt();
        scales[j] = if s > 1e-12 { s } else { 1.0 };
    }
    scales
}

/// glmnet-style per-column std for a dense `Array2<f64>`:
/// `s_j = sqrt((‖X[:,j]‖² − n · x̄_j²) / n)`. Constant columns clamp to
/// `1.0` so `Standardized<...>` doesn't divide by zero. Mirrors
/// `compute_csc_glmnet_scales` for the dense backend so dense GLMs
/// with `standardize=True` use the same scale-only
/// `Standardized<DenseMatrix>` recipe as the sparse path — keeping
/// dense and sparse identical at convergence.
fn compute_dense_glmnet_scales(x: &ndarray::Array2<f64>) -> ndarray::Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let n_f = n as f64;
    let mut scales = ndarray::Array1::<f64>::ones(p);
    for j in 0..p {
        let col = x.column(j);
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for &v in col.iter() {
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n_f;
        let var = ((sum_sq - n_f * mean * mean) / n_f).max(0.0);
        let s = var.sqrt();
        scales[j] = if s > 1e-12 { s } else { 1.0 };
    }
    scales
}

/// Build per-feature penalty weights for the (possibly intercept-
/// augmented) sparse feature space. User weights apply to the original
/// `p_user` features; the augmented intercept column gets weight 0.
fn build_sparse_penalty_weights(
    user_weights: &Option<ndarray::Array1<f64>>,
    p_user: usize,
    fit_intercept: bool,
) -> ndarray::Array1<f64> {
    let p_eff = if fit_intercept { p_user + 1 } else { p_user };
    let mut w = ndarray::Array1::<f64>::ones(p_eff);
    if let Some(uw) = user_weights {
        for j in 0..p_user {
            w[j] = uw[j];
        }
    }
    if fit_intercept {
        w[p_user] = 0.0;
    }
    w
}

#[allow(clippy::too_many_arguments)]
fn build_path_outputs_sparse_ls<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    screening: &str,
    fit_intercept: bool,
    standardize_x: bool,
    make_penalty: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty>,
{
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

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    // Compute per-column scales BEFORE intercept augmentation (we
    // never scale the intercept column).
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        append_intercept_to_csc(csc)
    } else {
        csc
    };
    let mut pen_weights = build_sparse_penalty_weights(&user_weights, n_cols, fit_intercept);
    // Scalar L1-style penalty weights live in original space; the
    // standardization changes variables to β̃ = s · β, so the scaled-
    // space penalty is `(w_j / s_j) |β̃_j|`. Intercept entry stays 0.
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
        screening: parse_screening(screening)?,
    };

    let datafit = LeastSquares::new(y_arr);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            // Intercept column stays at 1.0 (already initialized by `ones`).
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            solve_path(&std_design, &datafit, make_pen, &path_cfg)
        }
        None => solve_path(&csc_eff, &datafit, make_pen, &path_cfg),
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
    if let Some(scales) = scales_user.as_ref() {
        // β_orig = β̃ / s for the non-intercept columns.
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_mcp_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_path_outputs_sparse_ls(
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
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_scad_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    a: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_path_outputs_sparse_ls(
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
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_elastic_net_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    alpha: f64,
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
) -> PyResult<PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_path_outputs_sparse_ls(
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
        fit_intercept,
        standardize_x,
        move |lam, w| Box::new(ElasticNet::with_weights(lam, alpha, w)),
    )
}

// ---- Sparse LS group penalties (M4.2b) ------------------------------

/// Per-group weights for the (possibly intercept-augmented) sparse
/// group set. Mirrors `build_logistic_group_weights` but lives here so
/// the LS sparse path doesn't depend on the GLM helpers' naming.
fn build_sparse_group_weights(
    user_weights: &Option<ndarray::Array1<f64>>,
    n_groups_user: usize,
    fit_intercept: bool,
) -> ndarray::Array1<f64> {
    let n_eff = if fit_intercept {
        n_groups_user + 1
    } else {
        n_groups_user
    };
    let mut w = ndarray::Array1::<f64>::ones(n_eff);
    if let Some(uw) = user_weights {
        for g in 0..n_groups_user {
            w[g] = uw[g];
        }
    }
    if fit_intercept {
        w[n_groups_user] = 0.0;
    }
    w
}

/// Per-coord L1 weights for sparse-group penalties on the
/// intercept-augmented sparse feature space. Mirrors
/// `build_logistic_coord_weights`.
fn build_sparse_coord_weights(
    user_coord_weights: &Option<ndarray::Array1<f64>>,
    p_user: usize,
    fit_intercept: bool,
) -> ndarray::Array1<f64> {
    let p_eff = if fit_intercept { p_user + 1 } else { p_user };
    let mut w = ndarray::Array1::<f64>::ones(p_eff);
    if let Some(uw) = user_coord_weights {
        for j in 0..p_user {
            w[j] = uw[j];
        }
    }
    if fit_intercept {
        w[p_user] = 0.0;
    }
    w
}

#[allow(clippy::too_many_arguments)]
fn build_block_path_outputs_sparse_ls<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn GroupPenalty>,
{
    let y_arr = y.as_array().to_owned();
    if y_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_rows {}",
            y_arr.len(),
            n_rows
        )));
    }

    let labels_user = groups_labels.as_array().to_owned().to_vec();
    if labels_user.len() != n_cols {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels_user.len(),
            n_cols
        )));
    }
    let n_groups_user = groups_from_labels(&labels_user)?.n_groups();

    let user_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups_user {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups_user
                )));
            }
            Some(arr)
        }
        None => None,
    };

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        append_intercept_to_csc(csc)
    } else {
        csc
    };
    let labels_eff = if fit_intercept {
        append_intercept_group(&labels_user, n_groups_user)
    } else {
        labels_user
    };
    let groups = groups_from_labels(&labels_eff)?;
    // Per-group weights: glmnet/grpreg apply the group penalty in the
    // standardized space — the user-facing per-group weights stay
    // unscaled. Coefs are returned in original scale via the per-column
    // divide at the end.
    let group_w_eff = build_sparse_group_weights(&user_weights, n_groups_user, fit_intercept);

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: parse_screening(screening)?,
        parallel,
    };

    let datafit = LeastSquares::new(y_arr);
    let make_pen =
        move |lam: f64| -> Box<dyn GroupPenalty> { make_inner(lam, group_w_eff.clone()) };

    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            solve_block_path(&std_design, &datafit, make_pen, &groups, &block_cfg)
        }
        None => solve_block_path(&csc_eff, &datafit, make_pen, &groups, &block_cfg),
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_block_path_lla_outputs_sparse_ls<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64) -> Box<dyn GroupPenalty>,
{
    let y_arr = y.as_array().to_owned();
    if y_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_rows {}",
            y_arr.len(),
            n_rows
        )));
    }

    let labels_user = groups_labels.as_array().to_owned().to_vec();
    if labels_user.len() != n_cols {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels_user.len(),
            n_cols
        )));
    }
    let n_groups_user = groups_from_labels(&labels_user)?.n_groups();

    let user_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups_user {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups_user
                )));
            }
            Some(arr)
        }
        None => None,
    };

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        append_intercept_to_csc(csc)
    } else {
        csc
    };
    let labels_eff = if fit_intercept {
        append_intercept_group(&labels_user, n_groups_user)
    } else {
        labels_user
    };
    let groups = groups_from_labels(&labels_eff)?;
    let group_w_eff = build_sparse_group_weights(&user_weights, n_groups_user, fit_intercept);

    let block_cfg = BlockPathConfig {
        n_lambdas,
        lambda_min_ratio,
        lambdas: lambdas.map(|a| a.as_array().to_vec()),
        cd: CdConfig {
            max_iter,
            tol,
            acceleration,
        },
        screening: parse_screening(screening)?,
        parallel,
    };

    let datafit = LeastSquares::new(y_arr);
    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            solve_block_path_lla(
                &std_design,
                &datafit,
                group_w_eff,
                make_inner,
                &groups,
                &block_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => solve_block_path_lla(
            &csc_eff,
            &datafit,
            group_w_eff,
            make_inner,
            &groups,
            &block_cfg,
            max_outer,
            outer_tol,
        ),
    };
    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
    if let Some(scales) = scales_user.as_ref() {
        for k in 0..coefs.nrows() {
            for j in 0..coefs.ncols() {
                coefs[[k, j]] /= scales[j];
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_group_lasso_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_block_path_outputs_sparse_ls(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
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
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_sparse_group_lasso_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_block_path_outputs_sparse_ls(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
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
        move |lam, w| Box::new(SparseGroupLasso::with_weights(lam, alpha, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_group_elastic_net_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_block_path_outputs_sparse_ls(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
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

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_group_mcp_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    let labels_owned = groups.as_array().to_owned();
    let groups_obj = groups_from_labels(&labels_owned.to_vec())?;
    let n_groups_user = groups_obj.n_groups();
    let _ = groups_obj;

    let base_weights_for_lla = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => Array1::ones(n_groups_user),
    };
    let group_w_eff_for_lla =
        build_sparse_group_weights(&Some(base_weights_for_lla), n_groups_user, fit_intercept);

    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w_eff_for_lla.view());
        Box::new(GroupLasso::with_weights(lam, w))
    };

    build_block_path_lla_outputs_sparse_ls(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
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
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    coord_weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_sparse_group_mcp_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    screening: &str,
    acceleration: Option<usize>,
    parallel: bool,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let labels_owned = groups.as_array().to_owned();
    let groups_obj = groups_from_labels(&labels_owned.to_vec())?;
    let n_groups_user = groups_obj.n_groups();
    let _ = groups_obj;

    let base_group = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => Array1::ones(n_groups_user),
    };
    let user_coord = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    n_cols
                )));
            }
            Some(arr)
        }
        None => None,
    };
    let group_w_eff = build_sparse_group_weights(&Some(base_group), n_groups_user, fit_intercept);
    let coord_w_eff = build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let (gw, cw) = surrogate_sparse_group_mcp(
            beta,
            g,
            lam,
            gamma,
            alpha,
            group_w_eff.view(),
            coord_w_eff.view(),
        );
        Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
    };

    build_block_path_lla_outputs_sparse_ls(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
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

// ---- Sparse GLM scalar helpers (M4.2c) ----------------------------------

#[allow(clippy::too_many_arguments)]
fn build_glm_path_outputs_sparse<'py, F, V, G>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
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
    validate_y: V,
    make_glm: G,
    make_penalty: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty>,
    V: Fn(ndarray::ArrayView1<'_, f64>) -> PyResult<()>,
    G: FnOnce(ndarray::Array1<f64>) -> Box<dyn GlmDatafit>,
{
    let y_arr = y.as_array().to_owned();
    if y_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_rows {}",
            y_arr.len(),
            n_rows
        )));
    }
    validate_y(y_arr.view())?;

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

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    // Per-column scales computed BEFORE intercept augmentation; intercept
    // column is never scaled.
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        append_intercept_to_csc(csc)
    } else {
        csc
    };
    let mut pen_weights = build_sparse_penalty_weights(&user_weights, n_cols, fit_intercept);
    if let Some(scales) = &scales_user {
        for j in 0..n_cols {
            pen_weights[j] /= scales[j];
        }
    }

    let glm = make_glm(y_arr);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            prox_newton_solve_path(
                &std_design,
                &*glm,
                make_pen,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => prox_newton_solve_path(
            &csc_eff,
            &*glm,
            make_pen,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_glm_block_path_outputs_sparse<'py, F, V, G>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    y: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
    weights: Option<PyReadonlyArray1<f64>>,
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
    validate_y: V,
    make_glm: G,
    make_inner: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty>,
    V: Fn(ndarray::ArrayView1<'_, f64>) -> PyResult<()>,
    G: FnOnce(ndarray::Array1<f64>) -> Box<dyn GlmDatafit>,
{
    let y_arr = y.as_array().to_owned();
    if y_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "y length {} does not match n_rows {}",
            y_arr.len(),
            n_rows
        )));
    }
    validate_y(y_arr.view())?;

    let labels_user = groups_labels.as_array().to_owned().to_vec();
    if labels_user.len() != n_cols {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels_user.len(),
            n_cols
        )));
    }
    let n_groups_user = groups_from_labels(&labels_user)?.n_groups();

    let user_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups_user {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups_user
                )));
            }
            Some(arr)
        }
        None => None,
    };

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        append_intercept_to_csc(csc)
    } else {
        csc
    };
    let labels_eff = if fit_intercept {
        append_intercept_group(&labels_user, n_groups_user)
    } else {
        labels_user
    };
    let groups = groups_from_labels(&labels_eff)?;
    // Per-group weights stay unchanged: the group penalty applies in
    // standardized space.
    let group_w_eff = build_sparse_group_weights(&user_weights, n_groups_user, fit_intercept);

    let glm = make_glm(y_arr);
    let group_w_for_closure = group_w_eff.clone();
    let make_inner_wrapped =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            make_inner(beta, g, lam, &group_w_for_closure)
        };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (betas_aug, report) = match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            prox_newton_block_solve_path(
                &std_design,
                &*glm,
                group_w_eff,
                make_inner_wrapped,
                &groups,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => prox_newton_block_solve_path(
            &csc_eff,
            &*glm,
            group_w_eff,
            make_inner_wrapped,
            &groups,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
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

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

// ---- Sparse Cox helpers (M4.2c) -----------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_cox_path_outputs_sparse<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    make_penalty: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty>,
{
    let time_arr = time.as_array().to_owned();
    let event_arr = event.as_array().to_owned();
    if time_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "time length {} does not match n_rows {}",
            time_arr.len(),
            n_rows
        )));
    }
    if event_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "event length {} does not match n_rows {}",
            event_arr.len(),
            n_rows
        )));
    }
    validate_cox_outcomes(time_arr.view(), event_arr.view())?;

    let mut pen_weights = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_features {}",
                    arr.len(),
                    n_cols
                )));
            }
            arr
        }
        None => Array1::ones(n_cols),
    };

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };
    if let Some(scales) = &scales_user {
        for j in 0..n_cols {
            pen_weights[j] /= scales[j];
        }
    }

    let glm = CoxPH::new(time_arr, event_arr);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (mut betas, report) = match scales_user.as_ref() {
        Some(scales) => {
            let std_design = Standardized::new(csc, scales.clone());
            prox_newton_solve_path(
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
        None => prox_newton_solve_path(
            &csc,
            &glm,
            make_pen,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    if let Some(scales) = scales_user.as_ref() {
        for k in 0..betas.nrows() {
            for j in 0..betas.ncols() {
                betas[[k, j]] /= scales[j];
            }
        }
    }

    let n_lams = report.lambdas.len();
    let intercepts = Array1::<f64>::zeros(n_lams);

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;

    Ok((
        betas.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_cox_block_path_outputs_sparse<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups_labels: PyReadonlyArray1<i64>,
    weights: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    make_inner: F,
) -> PyResult<PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty>,
{
    let time_arr = time.as_array().to_owned();
    let event_arr = event.as_array().to_owned();
    if time_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "time length {} does not match n_rows {}",
            time_arr.len(),
            n_rows
        )));
    }
    if event_arr.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "event length {} does not match n_rows {}",
            event_arr.len(),
            n_rows
        )));
    }
    validate_cox_outcomes(time_arr.view(), event_arr.view())?;

    let labels = groups_labels.as_array().to_owned().to_vec();
    if labels.len() != n_cols {
        return Err(PyValueError::new_err(format!(
            "groups length {} does not match n_features {}",
            labels.len(),
            n_cols
        )));
    }
    let groups = groups_from_labels(&labels)?;
    let n_groups = groups.n_groups();

    let group_w = match weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_groups {
                return Err(PyValueError::new_err(format!(
                    "weights length {} does not match n_groups {}",
                    arr.len(),
                    n_groups
                )));
            }
            arr
        }
        None => Array1::ones(n_groups),
    };

    let csc = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let glm = CoxPH::new(time_arr, event_arr);

    let group_w_for_closure = group_w.clone();
    let make_inner_wrapped =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            make_inner(beta, g, lam, &group_w_for_closure)
        };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (mut betas, report) = match scales_user.as_ref() {
        Some(scales) => {
            let std_design = Standardized::new(csc, scales.clone());
            prox_newton_block_solve_path(
                &std_design,
                &glm,
                group_w,
                make_inner_wrapped,
                &groups,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => prox_newton_block_solve_path(
            &csc,
            &glm,
            group_w,
            make_inner_wrapped,
            &groups,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
    };

    if let Some(scales) = scales_user.as_ref() {
        for k in 0..betas.nrows() {
            for j in 0..betas.ncols() {
                betas[[k, j]] /= scales[j];
            }
        }
    }

    let n_lams = report.lambdas.len();
    let intercepts = Array1::<f64>::zeros(n_lams);

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;

    Ok((
        betas.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

// ---- Sparse logistic PyO3 wrappers (M4.2c) ------------------------------

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs_sparse(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_scad_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    a: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs_sparse(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        |_beta, _g, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |beta, g, lam, group_w| {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w.view());
            Box::new(GroupLasso::with_weights(lam, w))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_sparse_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |_beta, _g, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_logistic_sparse_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let user_coord = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    n_cols
                )));
            }
            Some(arr)
        }
        None => None,
    };
    let coord_w_eff = build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_binary,
        |y_arr| Box::new(BinomialLogit::new(y_arr)),
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                group_w.view(),
                coord_w_eff.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}

// ---- Sparse Poisson PyO3 wrappers (M4.2c) -------------------------------

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs_sparse(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_scad_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    a: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_path_outputs_sparse(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        |_beta, _g, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |beta, g, lam, group_w| {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w.view());
            Box::new(GroupLasso::with_weights(lam, w))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_sparse_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
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
) -> PyResult<PathOutput<'py>> {
    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |_beta, _g, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_poisson_sparse_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let user_coord = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    n_cols
                )));
            }
            Some(arr)
        }
        None => None,
    };
    let coord_w_eff = build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

    build_glm_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        y,
        groups,
        weights,
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
        validate_y_nonneg,
        |y_arr| Box::new(PoissonLog::new(y_arr)),
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                group_w.view(),
                coord_w_eff.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}

// ---- Sparse Cox PyO3 wrappers (M4.2c) -----------------------------------

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        time,
        event,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_scad_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    a: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        time,
        event,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        |_beta, _g, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |beta, g, lam, group_w| {
            let w = surrogate_weights_group_mcp(beta, g, lam, gamma, group_w.view());
            Box::new(GroupLasso::with_weights(lam, w))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_sparse_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    build_cox_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |_beta, _g, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *,
    gamma=3.0, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
fn solve_cox_sparse_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    coord_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<PathOutput<'py>> {
    let coord_w = match &coord_weights {
        Some(w) => {
            let arr = w.as_array().to_owned();
            if arr.len() != n_cols {
                return Err(PyValueError::new_err(format!(
                    "coord_weights length {} does not match n_features {}",
                    arr.len(),
                    n_cols
                )));
            }
            arr
        }
        None => Array1::ones(n_cols),
    };

    build_cox_block_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        time,
        event,
        groups,
        weights,
        lambdas,
        n_lambdas,
        lambda_min_ratio,
        max_iter,
        tol,
        acceleration,
        standardize_x,
        max_outer,
        outer_tol,
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_mcp(
                beta,
                g,
                lam,
                gamma,
                alpha,
                group_w.view(),
                coord_w.view(),
            );
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}

// =====================================================================
// Memory-mapped backend entry points (M4.x mmap, f64 column-major)
// =====================================================================
//
// `MmapMatrix` reads `X` directly from a column-major raw `f64` file —
// no in-RAM copy, no scipy.sparse marshalling. The Python side passes
// a path string; we open it here, optionally wrap in `Augmented` for
// the intercept column and `Standardized` for column scaling, then
// hand the trait object to the same path solver every other backend
// uses.
//
// v1 surface: scalar LS (MCP) and scalar logistic (MCP). Adding the
// other 22 entry points is mechanical mirroring of the `_sparse`
// surface; defer until there's user demand.

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
) -> PyResult<PathOutput<'py>>
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

    let mut pen_weights =
        ndarray::Array1::<f64>::ones(if fit_intercept { n_cols + 1 } else { n_cols });
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
        screening: parse_screening(screening)?,
    };
    let datafit = LeastSquares::new(y_arr);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> {
        Box::new(Mcp::with_weights(lam, gamma, pen_weights.clone()))
    };

    let p_eff = if fit_intercept { n_cols + 1 } else { n_cols };
    let (betas_aug, report) = match (fit_intercept, scales_user.as_ref()) {
        (false, None) => solve_path(&design, &datafit, make_pen, &path_cfg),
        (false, Some(scales)) => {
            let std_design = Standardized::new(design, scales.clone());
            solve_path(&std_design, &datafit, make_pen, &path_cfg)
        }
        (true, None) => {
            let aug = Augmented::new(design);
            solve_path(&aug, &datafit, make_pen, &path_cfg)
        }
        (true, Some(scales)) => {
            let aug = Augmented::new(design);
            let mut x_scale_eff = Array1::<f64>::ones(p_eff);
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(aug, x_scale_eff);
            solve_path(&std_design, &datafit, make_pen, &path_cfg)
        }
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
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
fn solve_mcp_ls_path_mmap<'py>(
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
) -> PyResult<PathOutput<'py>> {
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
fn solve_mcp_ls_path_mmap_f32<'py>(
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
) -> PyResult<PathOutput<'py>> {
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
) -> PyResult<PathOutput<'py>>
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

    let mut pen_weights =
        ndarray::Array1::<f64>::ones(if fit_intercept { n_cols + 1 } else { n_cols });
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
    let (betas_aug, report) = match (fit_intercept, scales_user.as_ref()) {
        (false, None) => prox_newton_solve_path(
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
            prox_newton_solve_path(
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
            prox_newton_solve_path(
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
            prox_newton_solve_path(
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
    };

    let (mut coefs, intercepts) = split_intercept(betas_aug, fit_intercept);
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
fn solve_logistic_mcp_path_mmap<'py>(
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
) -> PyResult<PathOutput<'py>> {
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    validate_y_binary(y_arr.view())?;
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
fn solve_logistic_mcp_path_mmap_f32<'py>(
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
) -> PyResult<PathOutput<'py>> {
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    validate_y_binary(y_arr.view())?;
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

// =====================================================================
// Chunked (row-block streaming) entry points
// =====================================================================
//
// `Chunked<C>` lets the solver treat a list of equal-`p` row-block
// chunks as one design matrix. Each chunk is a separate column-major
// raw f64/f32 file on disk; the wrapper routes hot-path calls to the
// chunks and stitches the results.
//
// PyO3 surface mirrors the mmap entries 1:1 — same Augmented +
// Standardized tower-of-wrappers logic from `mmap_*_path_inner`,
// just with the design built from a list of chunks instead of a
// single mmap. v1: LS-MCP and logistic-MCP × {f64, f32}.

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
fn solve_mcp_ls_path_chunked<'py>(
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
) -> PyResult<PathOutput<'py>> {
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
fn solve_mcp_ls_path_chunked_f32<'py>(
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
) -> PyResult<PathOutput<'py>> {
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
fn solve_logistic_mcp_path_chunked<'py>(
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
) -> PyResult<PathOutput<'py>> {
    let design = open_chunked_f64(chunks, n_cols)?;
    let n_rows = design.n_samples();
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    validate_y_binary(y_arr.view())?;
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
fn solve_logistic_mcp_path_chunked_f32<'py>(
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
) -> PyResult<PathOutput<'py>> {
    let design = open_chunked_f32(chunks, n_cols)?;
    let n_rows = design.n_samples();
    let (y_arr, user_weights) = mmap_validate_inputs(n_rows, n_cols, y, weights)?;
    validate_y_binary(y_arr.view())?;
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_mcp_ls, m)?)?;
    m.add_function(wrap_pyfunction!(solve_scad_ls, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_scad_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_elastic_net_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_group_lasso_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_group_elastic_net_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_group_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sparse_group_lasso_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_sparse_group_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_sparse_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_sparse_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_sparse_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_sparse_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_sparse_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_sparse_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mcp_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_scad_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_elastic_net_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_group_lasso_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_group_elastic_net_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_group_mcp_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        solve_sparse_group_lasso_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(solve_sparse_group_mcp_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_scad_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_group_lasso_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_group_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        solve_logistic_sparse_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        solve_logistic_sparse_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_scad_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_group_lasso_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_poisson_group_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        solve_poisson_sparse_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        solve_poisson_sparse_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(solve_cox_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_scad_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_group_lasso_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_cox_group_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        solve_cox_sparse_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(solve_cox_sparse_group_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mcp_ls_path_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_mcp_path_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mcp_ls_path_mmap_f32, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_mcp_path_mmap_f32, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mcp_ls_path_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mcp_ls_path_chunked_f32, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_mcp_path_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(solve_logistic_mcp_path_chunked_f32, m)?)?;
    Ok(())
}
