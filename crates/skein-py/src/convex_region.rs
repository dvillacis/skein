//! PyO3 bindings for nonconvex-path convex-region detection.
//!
//! Mirrors `skein_core::solver::convex_region`: given the coefficients
//! returned by a path solver plus the per-coordinate or per-group Lipschitz
//! constants, report the smallest λ-index along the path at which the local
//! objective ceases to be convex (grpreg/ncvreg's `convex.min`).
//!
//! These helpers are pure post-fit utilities — they read coefficients and a
//! cached Lipschitz vector and walk the path; the solver hot path is left
//! untouched.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use skein_core::{
    design::{DenseMatrix, SparseCSC},
    solver::{group_convex_min_idx, group_lipschitz_cache, scalar_convex_min_idx},
};

use crate::ls::{groups_from_labels, read_csc_arrays};

/// Smallest λ-index along the path at which an active coordinate violates
/// the local-convexity bound `col_lip[j] ≥ penalty_concavity`.
///
/// `betas` is shape `(n_lambdas, n_features)`; row `k` corresponds to λ_k
/// in the decreasing-λ ordering produced by the path solver.
/// `penalty_concavity` is `1/γ` for MCP, `1/(a−1)` for SCAD, or `0` (or
/// negative) for any convex penalty — in which case the function returns
/// `None` without scanning the path.
///
/// Returns `None` when the path stays locally convex everywhere.
#[pyfunction]
#[pyo3(signature = (betas, col_lip, penalty_concavity, zero_tol=1e-8))]
pub(crate) fn convex_min_idx_scalar(
    betas: PyReadonlyArray2<f64>,
    col_lip: PyReadonlyArray1<f64>,
    penalty_concavity: f64,
    zero_tol: f64,
) -> PyResult<Option<usize>> {
    let betas_arr = betas.as_array();
    let lip_vec: Vec<f64> = col_lip.as_array().iter().copied().collect();
    if lip_vec.len() != betas_arr.ncols() {
        return Err(PyValueError::new_err(format!(
            "col_lip length {} does not match betas n_features {}",
            lip_vec.len(),
            betas_arr.ncols()
        )));
    }
    Ok(scalar_convex_min_idx(
        betas_arr,
        &lip_vec,
        penalty_concavity,
        zero_tol,
    ))
}

/// Smallest λ-index along the path at which an active group violates the
/// local-convexity bound `group_lip[g] ≥ penalty_concavity`.
///
/// `group_labels[j]` is the (0-indexed) group containing feature `j`.
/// Group labels must form a contiguous range `0..n_groups`.
#[pyfunction]
#[pyo3(signature = (betas, group_labels, group_lip, penalty_concavity, zero_tol=1e-8))]
pub(crate) fn convex_min_idx_group(
    betas: PyReadonlyArray2<f64>,
    group_labels: PyReadonlyArray1<i64>,
    group_lip: PyReadonlyArray1<f64>,
    penalty_concavity: f64,
    zero_tol: f64,
) -> PyResult<Option<usize>> {
    let betas_arr = betas.as_array();
    let labels: Vec<i64> = group_labels.as_array().iter().copied().collect();
    let groups = groups_from_labels(&labels)?;
    let lip_vec: Vec<f64> = group_lip.as_array().iter().copied().collect();
    if lip_vec.len() != groups.n_groups() {
        return Err(PyValueError::new_err(format!(
            "group_lip length {} does not match n_groups {}",
            lip_vec.len(),
            groups.n_groups()
        )));
    }
    Ok(group_convex_min_idx(
        betas_arr,
        &groups,
        &lip_vec,
        penalty_concavity,
        zero_tol,
    ))
}

/// Per-group operator-norm Lipschitz cache `L[g] = ‖X_g‖_op² / n` for a
/// dense design matrix. One-time SVD-style power iteration per group; tiny
/// even for hundreds of groups since each block is small.
///
/// Useful for callers that need to feed [`convex_min_idx_group`] without
/// re-running the path solver.
#[pyfunction]
pub(crate) fn group_lipschitz_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    group_labels: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let design = DenseMatrix::new(x.as_array().to_owned());
    let labels: Vec<i64> = group_labels.as_array().iter().copied().collect();
    let groups = groups_from_labels(&labels)?;
    let lip = group_lipschitz_cache(&design, &groups);
    Ok(ndarray::Array1::from(lip).into_pyarray_bound(py))
}

/// Per-group operator-norm Lipschitz for a sparse (CSC) design matrix.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_lipschitz_sparse<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    data: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<i64>,
    indptr: PyReadonlyArray1<i64>,
    group_labels: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let csc: SparseCSC = read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;
    let labels: Vec<i64> = group_labels.as_array().iter().copied().collect();
    let groups = groups_from_labels(&labels)?;
    let lip = group_lipschitz_cache(&csc, &groups);
    Ok(ndarray::Array1::from(lip).into_pyarray_bound(py))
}
