//! PyO3 bindings for per-block (group) orthonormalization
//! (Breheny–Huang `orthogonalize` from grpreg).
//!
//! Workflow on the Python side:
//!   1. `x_orth, packed = orthonormalize_groups_dense(x, groups)`
//!   2. Fit any group-penalty path on `x_orth` as if it were the design.
//!   3. `coefs_orig = back_transform_coefs_path(coefs_orth, packed)`
//!
//! The packed back-transform is a list of `(cols, T_g)` pairs, with
//! `cols` as an `int64` ndarray and `T_g` as a `(|g|, |g|)` float64
//! ndarray — chosen for round-trippability without exposing a custom
//! PyO3 class. `back_transform_coefs_path` re-applies the per-block
//! T_g matvec to map fitted coefficients back to original-feature
//! space; same math as `skein_core::design::BlockBackTransform`.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use ndarray::{Array1, Array2};
use skein_core::design::orthonormalize_groups_dense as core_orthonormalize;

use crate::ls::groups_from_labels;

/// Orthonormalize a dense design matrix block-by-block.
///
/// Returns `(x_orth, packed_back_transform)` where:
///
/// * `x_orth` has the same shape as `x` and satisfies
///   `x_orth_g.T @ x_orth_g / n == I` for every group `g`.
/// * `packed_back_transform` is a `list[tuple[ndarray, ndarray]]`,
///   one entry per group: the original column indices (int64,
///   shape `(|g|,)`) and the per-group transform `T_g` (float64,
///   shape `(|g|, |g|)`). Feed this back into
///   [`back_transform_coefs_path`] to map coefficients fit on `x_orth`
///   into original-feature space.
///
/// Raises `ValueError` if any group's Gram matrix is rank-deficient
/// (perfectly collinear columns within the group).
#[pyfunction]
pub(crate) fn orthonormalize_groups_dense<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    group_labels: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyList>)> {
    let x_view = x.as_array();
    let labels: Vec<i64> = group_labels.as_array().iter().copied().collect();
    if labels.len() != x_view.ncols() {
        return Err(PyValueError::new_err(format!(
            "group_labels length {} does not match n_features {}",
            labels.len(),
            x_view.ncols()
        )));
    }
    let groups = groups_from_labels(&labels)?;
    let (x_orth, bt) = core_orthonormalize(x_view, &groups)
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let packed = PyList::empty_bound(py);
    for g in 0..groups.n_groups() {
        let (cols, t) = bt.block(g);
        let cols_arr: Array1<i64> = cols.iter().map(|&c| c as i64).collect();
        let t_owned: Array2<f64> = t.to_owned();
        packed.append((
            cols_arr.into_pyarray_bound(py),
            t_owned.into_pyarray_bound(py),
        ))?;
    }
    Ok((x_orth.into_pyarray_bound(py), packed))
}

/// Map a full path of coefficients fit in orthonormalized space back
/// to original-feature space. `coefs_orth` is shape
/// `(n_lambdas, n_features)`; `packed_back_transform` is the list
/// returned by [`orthonormalize_groups_dense`].
#[pyfunction]
pub(crate) fn back_transform_coefs_path<'py>(
    py: Python<'py>,
    coefs_orth: PyReadonlyArray2<f64>,
    packed_back_transform: &Bound<'_, PyList>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coefs_view = coefs_orth.as_array();
    let p = coefs_view.ncols();
    let entries = unpack_back_transform_entries(packed_back_transform, p)?;
    let n_lambdas = coefs_view.nrows();
    let mut out = Array2::<f64>::zeros((n_lambdas, p));
    for k in 0..n_lambdas {
        for (cols, t) in &entries {
            let g_size = cols.len();
            let mut b_block = Array1::<f64>::zeros(g_size);
            for (kk, &j) in cols.iter().enumerate() {
                b_block[kk] = coefs_view[[k, j]];
            }
            let b_out = t.dot(&b_block);
            for (kk, &j) in cols.iter().enumerate() {
                out[[k, j]] = b_out[kk];
            }
        }
    }
    Ok(out.into_pyarray_bound(py))
}

/// Map a single β vector back to original-feature space. Convenience
/// wrapper for callers fitting one λ.
#[pyfunction]
pub(crate) fn back_transform_coefs<'py>(
    py: Python<'py>,
    beta_orth: PyReadonlyArray1<f64>,
    packed_back_transform: &Bound<'_, PyList>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let beta_view = beta_orth.as_array();
    let p = beta_view.len();
    let entries = unpack_back_transform_entries(packed_back_transform, p)?;
    let mut out = Array1::<f64>::zeros(p);
    for (cols, t) in &entries {
        let g_size = cols.len();
        let mut b_block = Array1::<f64>::zeros(g_size);
        for (kk, &j) in cols.iter().enumerate() {
            b_block[kk] = beta_view[j];
        }
        let b_out = t.dot(&b_block);
        for (kk, &j) in cols.iter().enumerate() {
            out[j] = b_out[kk];
        }
    }
    Ok(out.into_pyarray_bound(py))
}

/// Decode a packed back-transform list-of-tuples into owned `(cols, T_g)`
/// entries. Validates that every group's column indices stay within
/// `[0, p)` and that each `T_g` is square with side length matching the
/// column count.
fn unpack_back_transform_entries(
    packed: &Bound<'_, PyList>,
    p: usize,
) -> PyResult<Vec<(Vec<usize>, Array2<f64>)>> {
    let mut entries: Vec<(Vec<usize>, Array2<f64>)> = Vec::with_capacity(packed.len());
    for item in packed.iter() {
        let tup = item.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            PyValueError::new_err("packed_back_transform entry must be (cols, T) tuple")
        })?;
        if tup.len() != 2 {
            return Err(PyValueError::new_err(
                "packed_back_transform entry must be (cols, T) tuple of length 2",
            ));
        }
        let cols_obj = tup.get_item(0)?;
        let t_obj = tup.get_item(1)?;
        let cols_arr: PyReadonlyArray1<i64> = cols_obj.extract()?;
        let t_arr: PyReadonlyArray2<f64> = t_obj.extract()?;
        let cols_v: Vec<usize> = cols_arr
            .as_array()
            .iter()
            .map(|&c| {
                if c < 0 || (c as usize) >= p {
                    Err(PyValueError::new_err(format!(
                        "back-transform col index {} out of range [0, {})",
                        c, p
                    )))
                } else {
                    Ok(c as usize)
                }
            })
            .collect::<PyResult<_>>()?;
        let t_owned: Array2<f64> = t_arr.as_array().to_owned();
        if t_owned.nrows() != t_owned.ncols() || t_owned.nrows() != cols_v.len() {
            return Err(PyValueError::new_err(format!(
                "T_g shape {:?} inconsistent with cols length {}",
                t_owned.shape(),
                cols_v.len()
            )));
        }
        entries.push((cols_v, t_owned));
    }
    Ok(entries)
}
