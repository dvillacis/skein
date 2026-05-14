//! Graphical lasso (M11) PyO3 bindings: single-population
//! [`solve_glasso_*`] and joint multi-population [`solve_joint_glasso_*`].
//!
//! Extracted from `lib.rs` in the M12 P4 refactor — fully self-contained:
//! all helpers ([`build_glasso_config`], `*_info_dict`, `collect_2d_views`)
//! and type aliases ([`GlassoOutput`], [`JointGlassoOutput`]) are used
//! only by the five pyfunctions in this module.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skein_core::{
    penalty::{GroupLassoFactory, GroupMcpFactory, LassoFactory, McpFactory, ScadFactory},
    solver::{glasso_solve, joint_glasso_solve, CdConfig, GlassoConfig, JointGlassoConfig},
};

type GlassoOutput<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyDict>,
);

type JointGlassoOutput<'py> = (Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyDict>);

fn glasso_info_dict<'py>(
    py: Python<'py>,
    report: &skein_core::solver::GlassoReport,
) -> PyResult<Bound<'py, PyDict>> {
    let info = PyDict::new_bound(py);
    info.set_item("outer_iter", report.outer_iter)?;
    info.set_item("converged", report.converged)?;
    info.set_item("max_w_delta", report.max_w_delta)?;
    Ok(info)
}

fn joint_glasso_info_dict<'py>(
    py: Python<'py>,
    report: &skein_core::solver::JointGlassoReport,
) -> PyResult<Bound<'py, PyDict>> {
    let info = PyDict::new_bound(py);
    info.set_item("iter", report.iter)?;
    info.set_item("converged", report.converged)?;
    info.set_item("primal_residual", report.primal_residual)?;
    info.set_item("dual_residual", report.dual_residual)?;
    Ok(info)
}

fn build_glasso_config(
    max_outer_iter: usize,
    outer_tol: f64,
    diag_offset: f64,
    inner_max_iter: usize,
    inner_tol: f64,
) -> GlassoConfig {
    GlassoConfig {
        max_outer_iter,
        outer_tol,
        diag_offset,
        inner: CdConfig {
            max_iter: inner_max_iter,
            tol: inner_tol,
            acceleration: None,
        },
        warm_start: None,
    }
}

#[pyfunction]
#[pyo3(signature = (sample_cov, lambda_, *, edge_weights=None, diag_offset=0.0,
    max_outer_iter=100, outer_tol=1e-4, inner_max_iter=200, inner_tol=1e-6))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_glasso_lasso<'py>(
    py: Python<'py>,
    sample_cov: PyReadonlyArray2<f64>,
    lambda_: f64,
    edge_weights: Option<PyReadonlyArray2<f64>>,
    diag_offset: f64,
    max_outer_iter: usize,
    outer_tol: f64,
    inner_max_iter: usize,
    inner_tol: f64,
) -> PyResult<GlassoOutput<'py>> {
    let s = sample_cov.as_array();
    let factory = LassoFactory { lambda: lambda_ };
    let cfg = build_glasso_config(
        max_outer_iter,
        outer_tol,
        diag_offset,
        inner_max_iter,
        inner_tol,
    );
    let ew = edge_weights.as_ref().map(|w| w.as_array());
    let (theta, w, report) = glasso_solve(s, ew, &factory, &cfg);
    let info = glasso_info_dict(py, &report)?;
    Ok((theta.into_pyarray_bound(py), w.into_pyarray_bound(py), info))
}

#[pyfunction]
#[pyo3(signature = (sample_cov, lambda_, gamma, *, edge_weights=None, diag_offset=0.0,
    max_outer_iter=100, outer_tol=1e-4, inner_max_iter=200, inner_tol=1e-6))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_glasso_mcp<'py>(
    py: Python<'py>,
    sample_cov: PyReadonlyArray2<f64>,
    lambda_: f64,
    gamma: f64,
    edge_weights: Option<PyReadonlyArray2<f64>>,
    diag_offset: f64,
    max_outer_iter: usize,
    outer_tol: f64,
    inner_max_iter: usize,
    inner_tol: f64,
) -> PyResult<GlassoOutput<'py>> {
    let s = sample_cov.as_array();
    let factory = McpFactory {
        lambda: lambda_,
        gamma,
    };
    let cfg = build_glasso_config(
        max_outer_iter,
        outer_tol,
        diag_offset,
        inner_max_iter,
        inner_tol,
    );
    let ew = edge_weights.as_ref().map(|w| w.as_array());
    let (theta, w, report) = glasso_solve(s, ew, &factory, &cfg);
    let info = glasso_info_dict(py, &report)?;
    Ok((theta.into_pyarray_bound(py), w.into_pyarray_bound(py), info))
}

#[pyfunction]
#[pyo3(signature = (sample_cov, lambda_, a, *, edge_weights=None, diag_offset=0.0,
    max_outer_iter=100, outer_tol=1e-4, inner_max_iter=200, inner_tol=1e-6))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_glasso_scad<'py>(
    py: Python<'py>,
    sample_cov: PyReadonlyArray2<f64>,
    lambda_: f64,
    a: f64,
    edge_weights: Option<PyReadonlyArray2<f64>>,
    diag_offset: f64,
    max_outer_iter: usize,
    outer_tol: f64,
    inner_max_iter: usize,
    inner_tol: f64,
) -> PyResult<GlassoOutput<'py>> {
    let s = sample_cov.as_array();
    let factory = ScadFactory { lambda: lambda_, a };
    let cfg = build_glasso_config(
        max_outer_iter,
        outer_tol,
        diag_offset,
        inner_max_iter,
        inner_tol,
    );
    let ew = edge_weights.as_ref().map(|w| w.as_array());
    let (theta, w, report) = glasso_solve(s, ew, &factory, &cfg);
    let info = glasso_info_dict(py, &report)?;
    Ok((theta.into_pyarray_bound(py), w.into_pyarray_bound(py), info))
}

fn collect_2d_views<'a>(arrays: &'a [PyReadonlyArray2<f64>]) -> Vec<ndarray::ArrayView2<'a, f64>> {
    arrays.iter().map(|a| a.as_array()).collect()
}

#[pyfunction]
#[pyo3(signature = (sample_covs, n_samples, lambda_, *, edge_weights=None,
    rho=1.0, diag_offset=0.0, max_iter=200, primal_tol=1e-5, dual_tol=1e-5))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_joint_glasso_lasso<'py>(
    py: Python<'py>,
    sample_covs: Vec<PyReadonlyArray2<f64>>,
    n_samples: Vec<f64>,
    lambda_: f64,
    edge_weights: Option<PyReadonlyArray2<f64>>,
    rho: f64,
    diag_offset: f64,
    max_iter: usize,
    primal_tol: f64,
    dual_tol: f64,
) -> PyResult<JointGlassoOutput<'py>> {
    if sample_covs.is_empty() {
        return Err(PyValueError::new_err(
            "sample_covs must contain at least one population",
        ));
    }
    if sample_covs.len() != n_samples.len() {
        return Err(PyValueError::new_err(
            "sample_covs and n_samples must have the same length",
        ));
    }
    let views = collect_2d_views(&sample_covs);
    let ew = edge_weights.as_ref().map(|w| w.as_array());
    let factory = GroupLassoFactory { lambda: lambda_ };
    let cfg = JointGlassoConfig {
        max_iter,
        primal_tol,
        dual_tol,
        rho,
        diag_offset,
    };
    let (thetas, report) = joint_glasso_solve(&views, &n_samples, ew, &factory, &cfg);
    let info = joint_glasso_info_dict(py, &report)?;
    let py_thetas: Vec<_> = thetas
        .into_iter()
        .map(|t| t.into_pyarray_bound(py))
        .collect();
    Ok((py_thetas, info))
}

#[pyfunction]
#[pyo3(signature = (sample_covs, n_samples, lambda_, gamma, *, edge_weights=None,
    rho=1.0, diag_offset=0.0, max_iter=200, primal_tol=1e-5, dual_tol=1e-5))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_joint_glasso_mcp<'py>(
    py: Python<'py>,
    sample_covs: Vec<PyReadonlyArray2<f64>>,
    n_samples: Vec<f64>,
    lambda_: f64,
    gamma: f64,
    edge_weights: Option<PyReadonlyArray2<f64>>,
    rho: f64,
    diag_offset: f64,
    max_iter: usize,
    primal_tol: f64,
    dual_tol: f64,
) -> PyResult<JointGlassoOutput<'py>> {
    if sample_covs.is_empty() {
        return Err(PyValueError::new_err(
            "sample_covs must contain at least one population",
        ));
    }
    if sample_covs.len() != n_samples.len() {
        return Err(PyValueError::new_err(
            "sample_covs and n_samples must have the same length",
        ));
    }
    let views = collect_2d_views(&sample_covs);
    let ew = edge_weights.as_ref().map(|w| w.as_array());
    let factory = GroupMcpFactory {
        lambda: lambda_,
        gamma,
    };
    let cfg = JointGlassoConfig {
        max_iter,
        primal_tol,
        dual_tol,
        rho,
        diag_offset,
    };
    let (thetas, report) = joint_glasso_solve(&views, &n_samples, ew, &factory, &cfg);
    let info = joint_glasso_info_dict(py, &report)?;
    let py_thetas: Vec<_> = thetas
        .into_iter()
        .map(|t| t.into_pyarray_bound(py))
        .collect();
    Ok((py_thetas, info))
}
