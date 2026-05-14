//! Multinomial / softmax logistic regression (M3.6) PyO3 bindings.
//!
//! Extracted from `lib.rs` in the M12 P4 refactor. K-class softmax with
//! `B ∈ ℝ^{p × K}` row-major (`bvec[jK + k] = B[j,k]`). Reduces — through
//! a Böhning diagonal majorization (constant per-(i,k) Hessian = 1/2) —
//! to a sequence of multi-task LS problems on `MultiTaskDesign<X>`.
//! Per-class intercepts via a 1s column appended at `j = p`, with
//! row-group weight 0 (unpenalized). Per-feature weights act at the
//! row-group level (one per feature).
//!
//! Output shapes: `coefs` is `(n_lambdas, p × K)` row-major bvec layout
//! (`coefs[lam, j × K + k] = B[j, k]`); `intercepts` is `(n_lambdas, K)`.
//!
//! Cross-cutting helpers (`append_intercept_column`, glmnet scales, CSC
//! readers) live in `lib.rs` as `pub(crate)`; called here as `crate::name`.

use ndarray::{Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skein_core::{
    datafit::MultinomialLogit,
    design::{Augmented, DenseMatrix, MultiTaskDesign, Standardized},
    groups::Groups,
    penalty::{GroupElasticNet, GroupLasso, GroupPenalty},
    solver::{
        prox_newton_block_solve_path, surrogate_weights_group_mcp, surrogate_weights_group_scad,
        CdConfig,
    },
};

type MultinomialPathOutput<'py> = (
    Bound<'py, PyArray2<f64>>, // coefs: (n_lambdas, p*K), row-major bvec
    Bound<'py, PyArray2<f64>>, // intercepts: (n_lambdas, K)
    Bound<'py, PyArray1<f64>>, // lambdas
    Bound<'py, PyDict>,
);

fn build_multinomial_one_hot(
    labels: ArrayView1<'_, f64>,
    n_classes: usize,
) -> PyResult<Array2<f64>> {
    let n = labels.len();
    let mut y = Array2::<f64>::zeros((n, n_classes));
    for (i, &lab) in labels.iter().enumerate() {
        let k = lab as usize;
        if (lab - k as f64).abs() > 1e-12 || k >= n_classes {
            return Err(PyValueError::new_err(format!(
                "label {} at row {} is not an integer in [0, {})",
                lab, i, n_classes
            )));
        }
        y[[i, k]] = 1.0;
    }
    Ok(y)
}

/// Split bvec (row-major) into `(coefs[:, p × K], intercepts[:, K])` when
/// the last "feature" of `betas_aug` is the augmented intercept column.
fn split_multinomial_intercept(
    betas_aug: Array2<f64>,
    p_user: usize,
    n_classes: usize,
    fit_intercept: bool,
) -> (Array2<f64>, Array2<f64>) {
    let n_lambdas = betas_aug.nrows();
    if !fit_intercept {
        let intercepts = Array2::<f64>::zeros((n_lambdas, n_classes));
        return (betas_aug, intercepts);
    }
    let mut coefs = Array2::<f64>::zeros((n_lambdas, p_user * n_classes));
    let mut intercepts = Array2::<f64>::zeros((n_lambdas, n_classes));
    for li in 0..n_lambdas {
        for j in 0..p_user {
            for kk in 0..n_classes {
                coefs[[li, j * n_classes + kk]] = betas_aug[[li, j * n_classes + kk]];
            }
        }
        for kk in 0..n_classes {
            intercepts[[li, kk]] = betas_aug[[li, p_user * n_classes + kk]];
        }
    }
    (coefs, intercepts)
}

#[allow(clippy::too_many_arguments)]
fn build_multinomial_path_outputs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
    make_inner: F,
) -> PyResult<MultinomialPathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &Array1<f64>) -> Box<dyn GroupPenalty> + Send,
{
    let x_arr = x.as_array().to_owned();
    let labels_view = labels.as_array();
    let n = x_arr.nrows();
    let p_user = x_arr.ncols();
    if labels_view.len() != n {
        return Err(PyValueError::new_err(format!(
            "labels length {} does not match n_samples {}",
            labels_view.len(),
            n
        )));
    }
    if n_classes < 2 {
        return Err(PyValueError::new_err("n_classes must be ≥ 2"));
    }
    let y_onehot = build_multinomial_one_hot(labels_view, n_classes)?;

    let user_weights: Option<Array1<f64>> = match weights {
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

    // Per-column scales BEFORE intercept augmentation; the intercept
    // column itself is unscaled (s = 1).
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };

    let p_eff = if fit_intercept { p_user + 1 } else { p_user };
    // Per-row-group weights (one per feature): user weights for j < p_user,
    // 0 for the intercept row-group at j = p_user. Weights are rescaled
    // by 1/s_j when standardizing — the LS sparse-group convention.
    let mut row_weights = Array1::<f64>::ones(p_eff);
    if let Some(uw) = &user_weights {
        for j in 0..p_user {
            row_weights[j] = uw[j];
        }
    }
    if let Some(scales) = &scales_user {
        for j in 0..p_user {
            row_weights[j] /= scales[j];
        }
    }
    if fit_intercept {
        row_weights[p_user] = 0.0;
    }

    let x_eff = if fit_intercept {
        crate::glm::append_intercept_column(&x_arr)
    } else {
        x_arr
    };
    let glm = MultinomialLogit::new(y_onehot);
    let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p_eff, n_classes);

    let row_weights_for_closure = row_weights.clone();
    let make_inner_wrapped =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            make_inner(beta, g, lam, &row_weights_for_closure)
        };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    // For Standardized<DenseMatrix>: build the per-augmented-column scale
    // vector with 1.0 at the intercept slot.
    let scale_vec_eff: Option<Array1<f64>> = scales_user.as_ref().map(|s| {
        let mut v = Array1::<f64>::ones(p_eff);
        for j in 0..p_user {
            v[j] = s[j];
        }
        v
    });

    let (betas_aug, report) = py.allow_threads(|| match scale_vec_eff {
        Some(scales) => {
            let std_design = Standardized::new(DenseMatrix::new(x_eff), scales);
            let design = MultiTaskDesign::new(std_design, n_classes);
            prox_newton_block_solve_path(
                &design,
                &glm,
                row_weights,
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
        None => {
            let design = MultiTaskDesign::new(DenseMatrix::new(x_eff), n_classes);
            prox_newton_block_solve_path(
                &design,
                &glm,
                row_weights,
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
    });

    let (mut coefs, intercepts) =
        split_multinomial_intercept(betas_aug, p_user, n_classes, fit_intercept);
    if let Some(scales) = &scales_user {
        for li in 0..coefs.nrows() {
            for j in 0..p_user {
                let inv_s = 1.0 / scales[j];
                for kk in 0..n_classes {
                    coefs[[li, j * n_classes + kk]] *= inv_s;
                }
            }
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;
    info.set_item("n_classes", n_classes)?;
    info.set_item("n_features", p_user)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_multinomial_path_outputs_sparse<'py, F>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
    make_inner: F,
) -> PyResult<MultinomialPathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &Array1<f64>) -> Box<dyn GroupPenalty> + Send,
{
    let labels_view = labels.as_array();
    if labels_view.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "labels length {} does not match n_samples {}",
            labels_view.len(),
            n_rows
        )));
    }
    if n_classes < 2 {
        return Err(PyValueError::new_err("n_classes must be ≥ 2"));
    }
    let y_onehot = build_multinomial_one_hot(labels_view, n_classes)?;
    let p_user = n_cols;

    let user_weights: Option<Array1<f64>> = match &weights {
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

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, x_data, x_indices, x_indptr)?;
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let p_eff = if fit_intercept { p_user + 1 } else { p_user };
    let mut row_weights = Array1::<f64>::ones(p_eff);
    if let Some(uw) = &user_weights {
        for j in 0..p_user {
            row_weights[j] = uw[j];
        }
    }
    if let Some(scales) = &scales_user {
        for j in 0..p_user {
            row_weights[j] /= scales[j];
        }
    }
    if fit_intercept {
        row_weights[p_user] = 0.0;
    }

    let glm = MultinomialLogit::new(y_onehot);
    let groups = MultiTaskDesign::<DenseMatrix>::auto_groups(p_eff, n_classes);
    let row_weights_for_closure = row_weights.clone();
    let make_inner_wrapped =
        move |beta: ArrayView1<'_, f64>, g: &Groups, lam: f64| -> Box<dyn GroupPenalty> {
            make_inner(beta, g, lam, &row_weights_for_closure)
        };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    // Per-column scale vector for the augmented design — intercept = 1.
    let scale_vec_eff: Option<Array1<f64>> = scales_user.as_ref().map(|s| {
        let mut v = Array1::<f64>::ones(p_eff);
        for j in 0..p_user {
            v[j] = s[j];
        }
        v
    });

    let (betas_aug, report) = py.allow_threads(|| match (fit_intercept, scale_vec_eff) {
        (true, Some(scales)) => {
            let std_design = Standardized::new(Augmented::new(csc), scales);
            let design = MultiTaskDesign::new(std_design, n_classes);
            prox_newton_block_solve_path(
                &design,
                &glm,
                row_weights,
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
        (true, None) => {
            let augmented = Augmented::new(csc);
            let design = MultiTaskDesign::new(augmented, n_classes);
            prox_newton_block_solve_path(
                &design,
                &glm,
                row_weights,
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
        (false, Some(scales)) => {
            let std_design = Standardized::new(csc, scales);
            let design = MultiTaskDesign::new(std_design, n_classes);
            prox_newton_block_solve_path(
                &design,
                &glm,
                row_weights,
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
        (false, None) => {
            let design = MultiTaskDesign::new(csc, n_classes);
            prox_newton_block_solve_path(
                &design,
                &glm,
                row_weights,
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
    });

    let (mut coefs, intercepts) =
        split_multinomial_intercept(betas_aug, p_user, n_classes, fit_intercept);
    if let Some(scales) = &scales_user {
        for li in 0..coefs.nrows() {
            for j in 0..p_user {
                let inv_s = 1.0 / scales[j];
                for kk in 0..n_classes {
                    coefs[[li, j * n_classes + kk]] *= inv_s;
                }
            }
        }
    }

    let info = PyDict::new_bound(py);
    info.set_item("outer_iters", report.outer_iters)?;
    info.set_item("outer_converged", report.outer_converged)?;
    info.set_item("inner_iters", report.inner_iters)?;
    info.set_item("final_losses", report.final_losses)?;
    info.set_item("n_classes", n_classes)?;
    info.set_item("n_features", p_user)?;

    Ok((
        coefs.into_pyarray_bound(py),
        intercepts.into_pyarray_bound(py),
        Array1::from(report.lambdas).into_pyarray_bound(py),
        info,
    ))
}

// ---- Multinomial dense entry points -------------------------------------

#[pyfunction]
#[pyo3(signature = (
    x, labels, n_classes, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    build_multinomial_path_outputs(
        py,
        x,
        labels,
        n_classes,
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
        |_beta, _g, lam, w| Box::new(GroupLasso::with_weights(lam, w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, labels, n_classes, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    build_multinomial_path_outputs(
        py,
        x,
        labels,
        n_classes,
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
        move |beta, g, lam, w| {
            let surrogate = surrogate_weights_group_mcp(beta, g, lam, gamma, w.view());
            Box::new(GroupLasso::with_weights(lam, surrogate))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, labels, n_classes, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    build_multinomial_path_outputs(
        py,
        x,
        labels,
        n_classes,
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
        move |beta, g, lam, w| {
            let surrogate = surrogate_weights_group_scad(beta, g, lam, a, w.view());
            Box::new(GroupLasso::with_weights(lam, surrogate))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, labels, n_classes, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_elastic_net_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_multinomial_path_outputs(
        py,
        x,
        labels,
        n_classes,
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
        move |_beta, _g, lam, w| Box::new(GroupElasticNet::with_weights(lam, alpha, w.clone())),
    )
}

// ---- Multinomial sparse entry points ------------------------------------

#[pyfunction]
#[pyo3(signature = (
    n_rows, n_cols, x_data, x_indices, x_indptr, labels, n_classes, *,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_lasso_path_sparse<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    build_multinomial_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        labels,
        n_classes,
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
        |_beta, _g, lam, w| Box::new(GroupLasso::with_weights(lam, w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    n_rows, n_cols, x_data, x_indices, x_indptr, labels, n_classes, *, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_mcp_path_sparse<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    build_multinomial_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        labels,
        n_classes,
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
        move |beta, g, lam, w| {
            let surrogate = surrogate_weights_group_mcp(beta, g, lam, gamma, w.view());
            Box::new(GroupLasso::with_weights(lam, surrogate))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    n_rows, n_cols, x_data, x_indices, x_indptr, labels, n_classes, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_scad_path_sparse<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    build_multinomial_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        labels,
        n_classes,
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
        move |beta, g, lam, w| {
            let surrogate = surrogate_weights_group_scad(beta, g, lam, a, w.view());
            Box::new(GroupLasso::with_weights(lam, surrogate))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    n_rows, n_cols, x_data, x_indices, x_indptr, labels, n_classes, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_multinomial_elastic_net_path_sparse<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    labels: PyReadonlyArray1<f64>,
    n_classes: usize,
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
) -> PyResult<MultinomialPathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_multinomial_path_outputs_sparse(
        py,
        n_rows,
        n_cols,
        x_data,
        x_indices,
        x_indptr,
        labels,
        n_classes,
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
        move |_beta, _g, lam, w| Box::new(GroupElasticNet::with_weights(lam, alpha, w.clone())),
    )
}
