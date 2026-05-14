//! Least-squares PyO3 bindings (M1 + M2 + M4.x): scalar (mcp/scad/EN/bridge)
//! and group (lasso/EN/MCP/SCAD/sparse-group×3) penalties, dense + sparse
//! design backends.
//!
//! Extracted from `lib.rs` in the M12 P4 refactor. Carries the shared LS
//! plumbing (`build_path_outputs`, `build_block_path_outputs`,
//! `build_block_path_lla_outputs`, plus the `_sparse_ls` siblings) and
//! the cross-cutting helpers used by every other family
//! (`parse_screening`, `groups_from_labels`, CSC readers, glmnet scales,
//! intercept builders, `PathOutput` type alias). Those are exposed
//! `pub(crate)` so `glm.rs`, `multinomial.rs`, `multitask.rs`,
//! `mmap_chunked.rs`, and `glasso.rs` can call them as `crate::ls::name`.

use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skein_core::{
    datafit::LeastSquares,
    design::DesignMatrix as _,
    design::{DenseMatrix, SparseCSC, Standardized},
    groups::Groups,
    penalty::{
        ElasticNet, GroupElasticNet, GroupLasso, GroupMcp, GroupPenalty, Mcp, Scad,
        SparseGroupLasso,
    },
    solver::{
        cd_solve, solve_block_path, solve_block_path_lla, solve_path, solve_path_lla,
        surrogate_sparse_group_mcp, surrogate_sparse_group_scad, surrogate_weights_bridge,
        surrogate_weights_group_scad, BlockPathConfig, CdConfig, PathConfig, Screening,
    },
    standardize::{
        destandardize_path, rescale_weights_for_standardize, standardize, StandardizeConfig,
    },
    Penalty,
};

pub(crate) type PathOutput<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyDict>,
);

#[pyfunction]
#[pyo3(signature = (x, y, lambda_, gamma, *, weights=None, max_iter=100, tol=1e-6))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_mcp_ls<'py>(
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
pub(crate) fn solve_scad_ls<'py>(
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

pub(crate) fn parse_screening(s: &str) -> PyResult<Screening> {
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
    sample_weights: Option<PyReadonlyArray1<f64>>,
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
    F: Fn(f64, Array1<f64>) -> Box<dyn Penalty> + Send,
{
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let n = x_arr.nrows();
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

    // Validate optional sample_weights up front; downstream paths reuse it.
    let sw_arr: Option<Array1<f64>> = match sample_weights {
        Some(sw) => {
            let arr = sw.as_array().to_owned();
            if arr.len() != n {
                return Err(PyValueError::new_err(format!(
                    "sample_weights length {} does not match n_samples {}",
                    arr.len(),
                    n
                )));
            }
            for &v in arr.iter() {
                if !v.is_finite() || v < 0.0 {
                    return Err(PyValueError::new_err(
                        "sample_weights must be finite and non-negative",
                    ));
                }
            }
            Some(arr)
        }
        None => None,
    };

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
        p0: 10,
    };

    // sample_weights path: the standardize/destandardize machinery
    // assumes unweighted column means / y mean, which conflicts with a
    // weighted loss. Route around it via the same augmented-intercept
    // trick the GLM solvers use: skip standardize, append a 1s column,
    // weight=0 on it. `standardize_x=True` would need weighted scaling
    // (deferred); we reject the combo rather than silently giving the
    // wrong answer.
    if let Some(sw) = sw_arr {
        if standardize_x {
            return Err(PyValueError::new_err(
                "sample_weights with standardize_x=True is not yet supported; \
                 standardize X yourself before fitting (or fit with standardize_x=False)",
            ));
        }
        let x_eff = if fit_intercept {
            crate::glm::append_intercept_column(&x_arr)
        } else {
            x_arr
        };
        let pen_weights =
            crate::glm::build_logistic_penalty_weights(&Some(weights_orig), p, fit_intercept);
        let design = DenseMatrix::new(x_eff);
        let datafit = LeastSquares::with_sample_weights(y_arr, sw);
        let make_pen =
            move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };
        // Release the GIL during the heavy compute so Python-side
        // thread pools (e.g. CV fold loops via joblib's "threads"
        // backend) can actually run folds in parallel. The closure
        // and its captures are pure Rust — no PyObject references.
        let (betas_aug, report) =
            py.allow_threads(|| solve_path(&design, &datafit, make_pen, &path_cfg));
        let (coefs, intercepts) = crate::glm::split_intercept(betas_aug, fit_intercept);

        let info = PyDict::new_bound(py);
        info.set_item("iters", report.iters)?;
        info.set_item("converged", report.converged)?;
        info.set_item("final_objs", report.final_objs)?;
        info.set_item("working_set_sizes", report.working_set_sizes)?;
        info.set_item("kkt_passes", report.kkt_passes)?;
        info.set_item("sample_weights", true)?;

        return Ok((
            coefs.into_pyarray_bound(py),
            intercepts.into_pyarray_bound(py),
            Array1::from(report.lambdas).into_pyarray_bound(py),
            info,
        ));
    }

    let std_cfg = StandardizeConfig {
        center_x: fit_intercept,
        scale_x: standardize_x,
        fit_intercept,
    };
    let (xs, ys, stats) = standardize(x_arr.view(), y_arr.view(), &std_cfg);
    let weights_std = rescale_weights_for_standardize(weights_orig.view(), &stats);

    let design = DenseMatrix::new(xs);
    let datafit = LeastSquares::new(ys);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, weights_std.clone()) };
    let (betas_std, report) =
        py.allow_threads(|| solve_path(&design, &datafit, make_pen, &path_cfg));
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
    sample_weights=None,
    max_iter=100,
    tol=1e-6,
    screening="strong",
    acceleration=Some(5),
    fit_intercept=true,
    standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_mcp_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
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
        sample_weights,
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
    sample_weights=None,
    max_iter=100,
    tol=1e-6,
    screening="strong",
    acceleration=Some(5),
    fit_intercept=true,
    standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_scad_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    a: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
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
        sample_weights,
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
    sample_weights=None,
    max_iter=100,
    tol=1e-6,
    screening="strong",
    acceleration=Some(5),
    fit_intercept=true,
    standardize_x=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_elastic_net_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    alpha: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
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
        sample_weights,
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
// Bridge / ℓ_q penalty: `λ · Σ_j w_j |β_j|^q` with `q ∈ (0, 1]`.
// Convex at q = 1 (plain weighted lasso); concave/non-convex for q < 1
// (bridge a.k.a. ℓ_q regression). Solved via outer LLA (the M3/M2.3
// scalar analog) wrapping a weighted-lasso inner.
// ---------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    x, y, *, q=0.5, eps=1e-6,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_bridge_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    q: f64,
    eps: f64,
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
    if !(0.0 < q && q <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "bridge q must be in (0, 1]; got {q}"
        )));
    }
    if eps <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "bridge eps must be > 0; got {eps}"
        )));
    }
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

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let design = DenseMatrix::new(xs);
    let datafit = LeastSquares::new(ys);

    let make_inner = move |beta: ArrayView1<'_, f64>,
                           lam: f64,
                           w_base: ArrayView1<'_, f64>|
          -> Box<dyn Penalty> {
        let w = surrogate_weights_bridge(beta, q, eps, w_base);
        Box::new(ElasticNet::with_weights(lam, 1.0, w))
    };
    let (betas_std, report) = py.allow_threads(|| {
        solve_path_lla(
            &design,
            &datafit,
            weights_std,
            make_inner,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        )
    });
    let (coefs, intercepts) = destandardize_path(betas_std.view(), &stats);

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_objs", report.final_objs)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, q=0.5, eps=1e-6,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_bridge_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    q: f64,
    eps: f64,
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
    if !(0.0 < q && q <= 1.0) {
        return Err(PyValueError::new_err(format!(
            "bridge q must be in (0, 1]; got {q}"
        )));
    }
    if eps <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "bridge eps must be > 0; got {eps}"
        )));
    }
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

    let csc = read_csc_arrays(n_rows, n_cols, x_data, x_indices, x_indptr)?;

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

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());
    let datafit = LeastSquares::new(y_arr);

    let make_inner = move |beta: ArrayView1<'_, f64>,
                           lam: f64,
                           w_base: ArrayView1<'_, f64>|
          -> Box<dyn Penalty> {
        let w = surrogate_weights_bridge(beta, q, eps, w_base);
        Box::new(ElasticNet::with_weights(lam, 1.0, w))
    };

    let (betas_aug, report) = py.allow_threads(|| match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            solve_path_lla(
                &std_design,
                &datafit,
                pen_weights,
                make_inner,
                n_lambdas,
                lambda_min_ratio,
                lambdas_vec,
                &cd_cfg,
                max_outer,
                outer_tol,
            )
        }
        None => solve_path_lla(
            &csc_eff,
            &datafit,
            pen_weights,
            make_inner,
            n_lambdas,
            lambda_min_ratio,
            lambdas_vec,
            &cd_cfg,
            max_outer,
            outer_tol,
        ),
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
    info.set_item("final_objs", report.final_objs)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

// ---------------------------------------------------------------------
// Group-penalty path solvers
// ---------------------------------------------------------------------

/// Convert a length-p label vector into a `Groups` (CSR-style). Group
/// labels must form a contiguous range starting at 0.
pub(crate) fn groups_from_labels(labels: &[i64]) -> PyResult<Groups> {
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
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn GroupPenalty> + Send,
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
    let (betas_std, report) =
        py.allow_threads(|| solve_block_path(&design, &datafit, make_pen, &groups, &block_cfg));
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
    F: Fn(ArrayView1<f64>, &Groups, f64) -> Box<dyn GroupPenalty> + Send,
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
    let (betas_std, report) = py.allow_threads(|| {
        solve_block_path_lla(
            &design,
            &datafit,
            base_weights,
            make_inner,
            &groups,
            &block_cfg,
            max_outer,
            outer_tol,
        )
    });
    let (coefs, intercepts) = destandardize_path(betas_std.view(), &stats);

    let info = PyDict::new_bound(py);
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;
    info.set_item("per_lambda_wall_ns", report.per_lambda_wall_ns)?;

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
pub(crate) fn solve_group_lasso_ls_path<'py>(
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
pub(crate) fn solve_sparse_group_lasso_ls_path<'py>(
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
pub(crate) fn solve_group_elastic_net_ls_path<'py>(
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
pub(crate) fn solve_group_mcp_ls_path<'py>(
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
    // M13.4b — native group-MCP block-CD (no LLA outer loop). The
    // `block_cd` inner solver dispatches to `GroupMcp::prox_group`
    // directly (closed-form group MCP prox per Breheny & Huang 2015 §3),
    // yielding a stationary point of the original non-convex objective
    // in one path solve per λ. The `max_outer` / `outer_tol` arguments
    // are now ignored (kept in the signature for backward compat with
    // any caller passing them by keyword); `solve_block_path`'s
    // convergence is governed by `cd.tol` and the path solver's KKT
    // verifier. Strong-rule screening still applies — the rule's
    // β_g=0 KKT subdifferential `λ·[-w_g, w_g]` is identical for
    // GroupLasso and GroupMcp. Profile (n=10k, p=1k, group_size=5,
    // n_groups=200, k_active=5, tol=1e-7, γ=3.0): native 10.45 s vs
    // LLA 36.19 s — **3.46× faster**, identical support, ≤5e-7
    // objective gap.
    let _ = max_outer;
    let _ = outer_tol;
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
        move |lam, w| Box::new(GroupMcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_group_scad_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    let labels_owned = groups.as_array().to_owned();
    let groups_obj = groups_from_labels(&labels_owned.to_vec())?;
    let n_groups = groups_obj.n_groups();
    let _ = groups_obj;

    let base_weights = match &weights {
        Some(w) => w.as_array().to_owned(),
        None => ndarray::Array1::ones(n_groups),
    };
    let make_inner = move |beta: ArrayView1<f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
        let w = surrogate_weights_group_scad(beta, g, lam, a, base_weights.view());
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
pub(crate) fn solve_sparse_group_mcp_ls_path<'py>(
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

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, a=3.7, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    coord_weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_sparse_group_scad_ls_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    a: f64,
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
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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
        let (gw, cw) = surrogate_sparse_group_scad(
            beta,
            g,
            lam,
            a,
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
pub(crate) fn read_csc_arrays(
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
pub(crate) fn append_intercept_to_csc(csc: SparseCSC) -> SparseCSC {
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
pub(crate) fn compute_csc_glmnet_scales(csc: &SparseCSC) -> ndarray::Array1<f64> {
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
pub(crate) fn compute_dense_glmnet_scales(x: &ndarray::Array2<f64>) -> ndarray::Array1<f64> {
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
pub(crate) fn build_sparse_penalty_weights(
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
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty> + Send,
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
        p0: 10,
    };

    let datafit = LeastSquares::new(y_arr);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let (betas_aug, report) = py.allow_threads(|| match scales_user.as_ref() {
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
    });

    let (mut coefs, intercepts) = crate::glm::split_intercept(betas_aug, fit_intercept);
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
pub(crate) fn solve_mcp_ls_path_sparse<'py>(
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
pub(crate) fn solve_scad_ls_path_sparse<'py>(
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
pub(crate) fn solve_elastic_net_ls_path_sparse<'py>(
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
pub(crate) fn build_sparse_group_weights(
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
pub(crate) fn build_sparse_coord_weights(
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
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn GroupPenalty> + Send,
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
        crate::glm::append_intercept_group(&labels_user, n_groups_user)
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

    let (betas_aug, report) = py.allow_threads(|| match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            solve_block_path(&std_design, &datafit, make_pen, &groups, &block_cfg)
        }
        None => solve_block_path(&csc_eff, &datafit, make_pen, &groups, &block_cfg),
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
    F: Fn(ArrayView1<'_, f64>, &Groups, f64) -> Box<dyn GroupPenalty> + Send,
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
        crate::glm::append_intercept_group(&labels_user, n_groups_user)
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
    let (betas_aug, report) = py.allow_threads(|| match scales_user.as_ref() {
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
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("final_objs", report.final_objs)?;
    info.set_item("working_set_sizes", report.working_set_sizes)?;
    info.set_item("kkt_passes", report.kkt_passes)?;
    info.set_item("per_lambda_wall_ns", report.per_lambda_wall_ns)?;

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
pub(crate) fn solve_group_lasso_ls_path_sparse<'py>(
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
pub(crate) fn solve_sparse_group_lasso_ls_path_sparse<'py>(
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
pub(crate) fn solve_group_elastic_net_ls_path_sparse<'py>(
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
pub(crate) fn solve_group_mcp_ls_path_sparse<'py>(
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
    // M13.4b — see `solve_group_mcp_ls_path` for the rationale. The
    // sparse helper passes pre-augmented weights (intercept group has
    // weight 0) into the make_inner closure; we construct GroupMcp on
    // top of that augmented vector identically to the dense path.
    let _ = max_outer;
    let _ = outer_tol;
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
        move |lam, w| Box::new(GroupMcp::with_weights(lam, gamma, w)),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_group_scad_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
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
) -> PyResult<PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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
        let w = surrogate_weights_group_scad(beta, g, lam, a, group_w_eff_for_lla.view());
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
pub(crate) fn solve_sparse_group_mcp_ls_path_sparse<'py>(
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

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    a=3.7, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    coord_weights=None,
    max_iter=100, tol=1e-6, screening="strong", acceleration=Some(5),
    parallel=false, fit_intercept=true, standardize_x=false,
    max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_sparse_group_scad_ls_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    a: f64,
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
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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
        let (gw, cw) = surrogate_sparse_group_scad(
            beta,
            g,
            lam,
            a,
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
