//! GLM PyO3 bindings — logistic, Poisson, Huber, Cox proportional-hazards.
//! Dense + sparse, scalar + group, M3.x family.
//!
//! Extracted from `lib.rs` in the M12 P4 refactor. Each datafit gets a
//! prox-Newton outer loop wrapping the M1 separable-penalty CD (or M2
//! group block-CD), with the surrogate built from `GlmDatafit::surrogate_at`.
//!
//! GLM-shared helpers (`build_glm_path_outputs`, `build_glm_block_path_outputs`,
//! and their `_sparse` siblings) take the validate / glm-factory / make-penalty
//! closures so logistic / Poisson / Huber share the same plumbing. Cox has
//! its own `build_cox_*` siblings because the (time, event) outcome and
//! ties handling don't fit the standard scalar-y GLM signature.
//!
//! Cross-cutting helpers (`parse_screening`, `groups_from_labels`,
//! `append_intercept_column`, `append_intercept_to_csc`, glmnet scales,
//! CSC readers, validators, weight builders, `split_intercept`,
//! `PathOutput`) live in `lib.rs` as `pub(crate)`; called here as
//! `crate::name`.

use ndarray::{Array1, Array2, ArrayView1};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use skein_core::{
    datafit::{BinomialLogit, CoxPH, Datafit as _, GlmDatafit, Huber, PoissonLog, TieHandling},
    design::{DenseMatrix, DesignMatrix as _, Standardized},
    groups::Groups,
    penalty::{
        ElasticNet, GroupLasso, GroupMcp, GroupPenalty, Mcp, Scad, SparseGroupLasso, SparseGroupMcp,
    },
    solver::{
        prox_newton_block_solve_path_timed, prox_newton_fused_solve_path_timed,
        prox_newton_solve_path_timed, surrogate_sparse_group_scad, CdConfig,
    },
    Penalty,
};

// ---------------------------------------------------------------------
// Logistic regression (binomial logit) via prox-Newton
// ---------------------------------------------------------------------

/// Augment X with a column of ones at the right edge (the intercept column).
pub(crate) fn append_intercept_column(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
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
pub(crate) fn build_logistic_penalty_weights(
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
pub(crate) fn split_intercept(
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
pub(crate) fn validate_y_binary(y: ndarray::ArrayView1<'_, f64>) -> PyResult<()> {
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
pub(crate) fn validate_y_nonneg(y: ndarray::ArrayView1<'_, f64>) -> PyResult<()> {
    for &v in y.iter() {
        if !v.is_finite() || v < 0.0 {
            return Err(PyValueError::new_err(
                "Poisson regression requires y ≥ 0 (finite)",
            ));
        }
    }
    Ok(())
}

/// Validate that y is finite (Huber regression — any real target).
pub(crate) fn validate_y_finite(y: ndarray::ArrayView1<'_, f64>) -> PyResult<()> {
    for &v in y.iter() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(
                "Huber regression requires finite y (no NaN, no ±∞)",
            ));
        }
    }
    Ok(())
}

/// Build a `make_glm` closure for Poisson regression that wires an
/// optional `offset` (length-`n_samples` array, typically log-exposure)
/// into the underlying `PoissonLog::with_offset` constructor. Returns
/// `Err` on length mismatch or non-finite offset entries.
fn poisson_glm_factory(
    offset: Option<PyReadonlyArray1<f64>>,
    n_samples: usize,
) -> PyResult<impl FnOnce(Array1<f64>) -> Box<dyn GlmDatafit>> {
    let offset_arr: Option<Array1<f64>> = match offset {
        Some(o) => {
            let arr = o.as_array().to_owned();
            if arr.len() != n_samples {
                return Err(PyValueError::new_err(format!(
                    "offset length {} does not match n_samples {}",
                    arr.len(),
                    n_samples
                )));
            }
            for &v in arr.iter() {
                if !v.is_finite() {
                    return Err(PyValueError::new_err("Poisson offset must be finite"));
                }
            }
            Some(arr)
        }
        None => None,
    };
    Ok(move |y_arr: Array1<f64>| -> Box<dyn GlmDatafit> {
        match offset_arr {
            Some(o) => Box::new(PoissonLog::with_offset(y_arr, o)),
            None => Box::new(PoissonLog::new(y_arr)),
        }
    })
}

/// Sample-weights-aware variant of [`poisson_glm_factory`]. Used by the
/// dense scalar Poisson MCP/SCAD path where `build_glm_path_outputs`
/// passes `(y, Option<sw>)` to the factory; the group / sparse paths
/// keep using the original factory because they don't (yet) take
/// `sample_weights`.
#[allow(clippy::type_complexity)]
fn poisson_glm_factory_with_sw(
    offset: Option<PyReadonlyArray1<f64>>,
    n_samples: usize,
) -> PyResult<impl FnOnce(Array1<f64>, Option<Array1<f64>>) -> Box<dyn GlmDatafit>> {
    let offset_arr: Option<Array1<f64>> = match offset {
        Some(o) => {
            let arr = o.as_array().to_owned();
            if arr.len() != n_samples {
                return Err(PyValueError::new_err(format!(
                    "offset length {} does not match n_samples {}",
                    arr.len(),
                    n_samples
                )));
            }
            for &v in arr.iter() {
                if !v.is_finite() {
                    return Err(PyValueError::new_err("Poisson offset must be finite"));
                }
            }
            Some(arr)
        }
        None => None,
    };
    Ok(
        move |y_arr: Array1<f64>, sw: Option<Array1<f64>>| -> Box<dyn GlmDatafit> {
            match (offset_arr, sw) {
                (Some(o), Some(w)) => {
                    Box::new(PoissonLog::with_sample_weights_and_offset(y_arr, w, o))
                }
                (Some(o), None) => Box::new(PoissonLog::with_offset(y_arr, o)),
                (None, Some(w)) => Box::new(PoissonLog::with_sample_weights(y_arr, w)),
                (None, None) => Box::new(PoissonLog::new(y_arr)),
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn build_glm_path_outputs<'py, F, V, G>(
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
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
    validate_y: V,
    make_glm: G,
    make_penalty: F,
    use_fused: bool,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty> + Send,
    V: Fn(ndarray::ArrayView1<'_, f64>) -> PyResult<()>,
    G: FnOnce(ndarray::Array1<f64>, Option<ndarray::Array1<f64>>) -> Box<dyn GlmDatafit>,
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
        Some(crate::ls::compute_dense_glmnet_scales(&x_arr))
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
    let glm = make_glm(y_arr, sw_arr);

    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    // Release the GIL during the heavy compute so Python-side thread
    // pools (CV fold loops, joblib in stability selection, debiased
    // lasso nodewise loop, etc.) actually run concurrently.
    // Branch on `use_fused`: GLM × MCP/SCAD routes through the M14f
    // fused IRLS+CD solver (`prox_newton_fused_solve_path`); GLM ×
    // ElasticNet / Huber × {MCP, SCAD} stay on the classic solver.
    let (betas_aug, report, times_ns) =
        py.allow_threads(|| match (scales_user.as_ref(), use_fused) {
            (Some(scales), true) => {
                let mut x_scale_eff = Array1::<f64>::ones(design.n_features());
                for j in 0..p_user {
                    x_scale_eff[j] = scales[j];
                }
                let std_design = Standardized::new(design, x_scale_eff);
                prox_newton_fused_solve_path_timed(
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
            (Some(scales), false) => {
                let mut x_scale_eff = Array1::<f64>::ones(design.n_features());
                for j in 0..p_user {
                    x_scale_eff[j] = scales[j];
                }
                let std_design = Standardized::new(design, x_scale_eff);
                prox_newton_solve_path_timed(
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
            (None, true) => prox_newton_fused_solve_path_timed(
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
            (None, false) => prox_newton_solve_path_timed(
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
        });

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
    info.set_item("times_ns", times_ns)?;

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
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_mcp_path<'py>(
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
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr, sw| match sw {
            Some(w) => Box::new(BinomialLogit::with_sample_weights(y_arr, w)),
            None => Box::new(BinomialLogit::new(y_arr)),
        },
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_scad_path<'py>(
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
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr, sw| match sw {
            Some(w) => Box::new(BinomialLogit::with_sample_weights(y_arr, w)),
            None => Box::new(BinomialLogit::new(y_arr)),
        },
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_elastic_net_path<'py>(
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
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_binary,
        |y_arr, sw| match sw {
            Some(w) => Box::new(BinomialLogit::with_sample_weights(y_arr, w)),
            None => Box::new(BinomialLogit::new(y_arr)),
        },
        move |lam, w| Box::new(ElasticNet::with_weights(lam, alpha, w)),
        false,
    )
}

// ---------------------------------------------------------------------
// Huber regression (M3.7) — robust LS via prox-Newton on a re-weighted
// quadratic surrogate. Reuses the M3.2 / M3.4 `build_glm_path_outputs`
// machinery verbatim; only the GLM factory and y-validator differ.
// ---------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    x, y, *, delta, gamma=3.0,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_huber_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    delta: f64,
    gamma: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PyValueError::new_err(
            "Huber delta must be a positive finite number",
        ));
    }
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_finite,
        move |y_arr, sw| match sw {
            Some(w) => Box::new(Huber::with_sample_weights(y_arr, delta, w)),
            None => Box::new(Huber::new(y_arr, delta)),
        },
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, delta, a=3.7,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_huber_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    delta: f64,
    a: f64,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if !delta.is_finite() || delta <= 0.0 {
        return Err(PyValueError::new_err(
            "Huber delta must be a positive finite number",
        ));
    }
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_finite,
        move |y_arr, sw| match sw {
            Some(w) => Box::new(Huber::with_sample_weights(y_arr, delta, w)),
            None => Box::new(Huber::new(y_arr, delta)),
        },
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
        false,
    )
}

// ---------------------------------------------------------------------
// Logistic regression + group / sparse-group penalties (M3.3)
// ---------------------------------------------------------------------

/// Append the intercept column to a label vector by adding it as a new
/// singleton group (so it sits in `groups[p]` with label `n_groups`).
pub(crate) fn append_intercept_group(labels: &[i64], n_groups: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(labels.len() + 1);
    out.extend_from_slice(labels);
    out.push(n_groups as i64);
    out
}

/// Build per-group L2 weights for the intercept-augmented group set.
/// User group weights are `Some` only for the original `n_groups`; the
/// new singleton group gets weight 0.
pub(crate) fn build_logistic_group_weights(
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
pub(crate) fn build_logistic_coord_weights(
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

/// Split a flat per-feature coord-weights array into per-group blocks.
/// Used by the native sparse-group MCP closures (M14c.2) to build the
/// `Vec<Array1<f64>>` shape expected by
/// `SparseGroupMcp::with_coord_weights`.
pub(crate) fn split_coord_weights_per_group(
    coord_w_flat: ndarray::ArrayView1<'_, f64>,
    groups: &Groups,
) -> Vec<ndarray::Array1<f64>> {
    (0..groups.n_groups())
        .map(|g| {
            let cols = groups.group(g);
            ndarray::Array1::from_iter(cols.iter().map(|&j| coord_w_flat[j]))
        })
        .collect()
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
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty> + Send,
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
    let n_groups_user = crate::ls::groups_from_labels(&labels_user)?.n_groups();

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
        Some(crate::ls::compute_dense_glmnet_scales(&x_arr))
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
    let groups = crate::ls::groups_from_labels(&labels_eff)?;
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

    let (betas_aug, report, times_ns) = py.allow_threads(|| match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(design.n_features());
            for j in 0..p_user {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(design, x_scale_eff);
            prox_newton_block_solve_path_timed(
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
        None => prox_newton_block_solve_path_timed(
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
    });

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
    info.set_item("times_ns", times_ns)?;

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
pub(crate) fn solve_logistic_group_lasso_path<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
pub(crate) fn solve_logistic_group_mcp_path<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
        // M13.4c — native group-MCP block-CD inside the prox-Newton outer
        // loop. `GroupMcp::prox_group` (closed-form per Breheny & Huang
        // 2015 §3) replaces the LLA-wrapped weighted GroupLasso surrogate.
        // Prox-Newton still iterates on the GLM weighted-LS surrogate;
        // `max_outer` / `outer_tol` retain their semantics as
        // prox-Newton outer caps. Strong-rule screening carries over
        // unchanged — the β=0 KKT subdifferential `λ·[-w_g, w_g]` is
        // identical for GroupLasso and GroupMcp.
        move |_beta, _groups, lam, group_w| {
            Box::new(GroupMcp::with_weights(lam, gamma, group_w.clone()))
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
pub(crate) fn solve_logistic_sparse_group_lasso_path<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
pub(crate) fn solve_logistic_sparse_group_mcp_path<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
            // M14c.2 — native sparse-group MCP block-CD inside the
            // prox-Newton outer loop. `SparseGroupMcp::prox_group`
            // (Breheny & Huang 2015 §3 closed-form composition of
            // per-coord MCP + group-MCP) replaces the LLA-wrapped
            // weighted SparseGroupLasso surrogate. Sibling of M13.4c
            // for the plain group-MCP family.
            let _ = beta;
            let cw = crate::glm::split_coord_weights_per_group(coord_w_eff.view(), g);
            Box::new(SparseGroupMcp::with_coord_weights(
                lam,
                alpha,
                gamma,
                group_w.clone(),
                cw,
            ))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, a=3.7, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_sparse_group_scad_path<'py>(
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
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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
            let (gw, cw) = surrogate_sparse_group_scad(
                beta,
                g,
                lam,
                a,
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
    x, y, *, gamma=3.0, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory_with_sw(offset, n)?;
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        make_glm,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, a=3.7, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    a: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory_with_sw(offset, n)?;
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        make_glm,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, *, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    sample_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_elastic_net_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    alpha: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    lambdas: Option<PyReadonlyArray1<f64>>,
    n_lambdas: usize,
    lambda_min_ratio: f64,
    weights: Option<PyReadonlyArray1<f64>>,
    sample_weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory_with_sw(offset, n)?;
    build_glm_path_outputs(
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
        fit_intercept,
        standardize_x,
        max_outer,
        outer_tol,
        validate_y_nonneg,
        make_glm,
        move |lam, w| Box::new(ElasticNet::with_weights(lam, alpha, w)),
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory(offset, n)?;
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
        make_glm,
        |_beta, _groups, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory(offset, n)?;
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
        make_glm,
        // M13.4c — see solve_logistic_group_mcp_path for rationale.
        move |_beta, _groups, lam, group_w| {
            Box::new(GroupMcp::with_weights(lam, gamma, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_sparse_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory(offset, n)?;
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
        make_glm,
        move |_beta, _groups, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, gamma=3.0, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_sparse_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory(offset, n)?;
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
        make_glm,
        move |beta, g, lam, group_w| {
            // M14c.2 — native sparse-group MCP block-CD inside the
            // prox-Newton outer loop. `SparseGroupMcp::prox_group`
            // (Breheny & Huang 2015 §3 closed-form composition of
            // per-coord MCP + group-MCP) replaces the LLA-wrapped
            // weighted SparseGroupLasso surrogate. Sibling of M13.4c
            // for the plain group-MCP family.
            let _ = beta;
            let cw = crate::glm::split_coord_weights_per_group(coord_w_eff.view(), g);
            Box::new(SparseGroupMcp::with_coord_weights(
                lam,
                alpha,
                gamma,
                group_w.clone(),
                cw,
            ))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, y, groups, *, a=3.7, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_sparse_group_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    a: f64,
    alpha: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    let n = x.as_array().nrows();
    let make_glm = poisson_glm_factory(offset, n)?;
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
        make_glm,
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_scad(
                beta,
                g,
                lam,
                a,
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

/// Parse a `ties` string ("breslow" / "efron") into a `TieHandling`.
fn parse_cox_ties(s: &str) -> PyResult<TieHandling> {
    match s {
        "breslow" => Ok(TieHandling::Breslow),
        "efron" => Ok(TieHandling::Efron),
        other => Err(PyValueError::new_err(format!(
            "ties must be 'breslow' or 'efron'; got {other:?}"
        ))),
    }
}

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

/// Cox partial-likelihood IRLS surrogate at a fitted ``β``.
///
/// Returns the per-sample weights ``w`` and working responses ``z``
/// such that minimizing ``(1/2n) Σ w_i (X β − z_i)²`` is the local
/// quadratic expansion of the Cox negative partial log-likelihood at
/// the supplied ``β``. The diagonal of ``w`` is the per-sample Fisher
/// information of the partial likelihood (Cox PH analog of the
/// logistic ``p(1−p)`` Hessian diagonal) — exactly what the
/// nodewise-Fisher debiased estimator needs to weight the design
/// before constructing ``Θ̂``.
///
/// Used by :func:`skein_glm.debiased_cox_lasso`.
#[pyfunction]
#[pyo3(signature = (x, time, event, beta, *, ties="breslow"))]
#[allow(clippy::type_complexity)]
pub(crate) fn cox_surrogate_weights_at<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    beta: PyReadonlyArray1<f64>,
    ties: &str,
) -> PyResult<(
    pyo3::Bound<'py, numpy::PyArray1<f64>>,
    pyo3::Bound<'py, numpy::PyArray1<f64>>,
)> {
    let x_arr = x.as_array();
    let time_arr = time.as_array().to_owned();
    let event_arr = event.as_array().to_owned();
    let beta_arr = beta.as_array();
    let n = x_arr.nrows();
    let p = x_arr.ncols();
    if time_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "time length {} does not match n_samples {n}",
            time_arr.len()
        )));
    }
    if event_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "event length {} does not match n_samples {n}",
            event_arr.len()
        )));
    }
    if beta_arr.len() != p {
        return Err(PyValueError::new_err(format!(
            "beta length {} does not match n_features {p}",
            beta_arr.len()
        )));
    }
    validate_cox_outcomes(time_arr.view(), event_arr.view())?;
    let ties_enum = parse_cox_ties(ties)?;
    let glm = CoxPH::with_ties(time_arr, event_arr, ties_enum);
    let design = DenseMatrix::new(x_arr.to_owned());
    let surrogate = glm.surrogate_at(&design, beta_arr);
    let w = surrogate
        .sample_weights()
        .expect("Cox surrogate always has sample_weights")
        .to_owned();
    let z = surrogate.y().to_owned();
    Ok((w.into_pyarray_bound(py), z.into_pyarray_bound(py)))
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
    ties: TieHandling,
    make_penalty: F,
    use_fused: bool,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty> + Send,
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
        Some(crate::ls::compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };
    if let Some(scales) = &scales_user {
        for j in 0..p {
            pen_weights[j] /= scales[j];
        }
    }

    let design = DenseMatrix::new(x_arr);
    let glm = CoxPH::with_ties(time_arr, event_arr, ties);

    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (mut betas, report, times_ns) =
        py.allow_threads(|| match (scales_user.as_ref(), use_fused) {
            (Some(scales), true) => {
                let std_design = Standardized::new(design, scales.clone());
                prox_newton_fused_solve_path_timed(
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
            (Some(scales), false) => {
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
            (None, true) => prox_newton_fused_solve_path_timed(
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
            (None, false) => prox_newton_solve_path_timed(
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
        });

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
    info.set_item("times_ns", times_ns)?;

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
    ties: TieHandling,
    make_inner: F,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty> + Send,
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
    let groups = crate::ls::groups_from_labels(&labels)?;
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
        Some(crate::ls::compute_dense_glmnet_scales(&x_arr))
    } else {
        None
    };

    let design = DenseMatrix::new(x_arr);
    let glm = CoxPH::with_ties(time_arr, event_arr, ties);

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

    let (mut betas, report, times_ns) = py.allow_threads(|| match scales_user.as_ref() {
        Some(scales) => {
            let std_design = Standardized::new(design, scales.clone());
            prox_newton_block_solve_path_timed(
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
        None => prox_newton_block_solve_path_timed(
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
    });

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
    info.set_item("times_ns", times_ns)?;

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
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    gamma: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, *, a=3.7,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    a: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        |_beta, _groups, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, gamma=3.0,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        // M13.4c — native group-MCP block-CD inside the prox-Newton outer
        // loop. See solve_logistic_group_mcp_path for the full rationale.
        move |_beta, _groups, lam, group_w| {
            Box::new(GroupMcp::with_weights(lam, gamma, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, alpha=0.5,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_sparse_group_lasso_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |_beta, _groups, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, gamma=3.0, alpha=0.5,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_sparse_group_mcp_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    alpha: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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

    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |beta, g, lam, group_w| {
            // M14c.2 — native sparse-group MCP block-CD; see
            // solve_logistic_sparse_group_mcp_path for the full rationale.
            let _ = beta;
            let cw_per_group = crate::glm::split_coord_weights_per_group(coord_w.view(), g);
            Box::new(SparseGroupMcp::with_coord_weights(
                lam,
                alpha,
                gamma,
                group_w.clone(),
                cw_per_group,
            ))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x, time, event, groups, *, a=3.7, alpha=0.5,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_sparse_group_scad_path<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    a: f64,
    alpha: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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

    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |beta, g, lam, group_w| {
            let (gw, cw) =
                surrogate_sparse_group_scad(beta, g, lam, a, alpha, group_w.view(), coord_w.view());
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
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
    use_fused: bool,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty> + Send,
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

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    // Per-column scales computed BEFORE intercept augmentation; intercept
    // column is never scaled.
    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        crate::ls::append_intercept_to_csc(csc)
    } else {
        csc
    };
    let mut pen_weights =
        crate::ls::build_sparse_penalty_weights(&user_weights, n_cols, fit_intercept);
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

    let (betas_aug, report, times_ns) =
        py.allow_threads(|| match (scales_user.as_ref(), use_fused) {
            (Some(scales), true) => {
                let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
                for j in 0..n_cols {
                    x_scale_eff[j] = scales[j];
                }
                let std_design = Standardized::new(csc_eff, x_scale_eff);
                prox_newton_fused_solve_path_timed(
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
            (Some(scales), false) => {
                let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
                for j in 0..n_cols {
                    x_scale_eff[j] = scales[j];
                }
                let std_design = Standardized::new(csc_eff, x_scale_eff);
                prox_newton_solve_path_timed(
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
            (None, true) => prox_newton_fused_solve_path_timed(
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
            (None, false) => prox_newton_solve_path_timed(
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
        });

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
    info.set_item("times_ns", times_ns)?;

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
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty> + Send,
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
    let n_groups_user = crate::ls::groups_from_labels(&labels_user)?.n_groups();

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

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let csc_eff = if fit_intercept {
        crate::ls::append_intercept_to_csc(csc)
    } else {
        csc
    };
    let labels_eff = if fit_intercept {
        append_intercept_group(&labels_user, n_groups_user)
    } else {
        labels_user
    };
    let groups = crate::ls::groups_from_labels(&labels_eff)?;
    // Per-group weights stay unchanged: the group penalty applies in
    // standardized space.
    let group_w_eff =
        crate::ls::build_sparse_group_weights(&user_weights, n_groups_user, fit_intercept);

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

    let (betas_aug, report, times_ns) = py.allow_threads(|| match scales_user.as_ref() {
        Some(scales) => {
            let mut x_scale_eff = Array1::<f64>::ones(csc_eff.n_features());
            for j in 0..n_cols {
                x_scale_eff[j] = scales[j];
            }
            let std_design = Standardized::new(csc_eff, x_scale_eff);
            prox_newton_block_solve_path_timed(
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
        None => prox_newton_block_solve_path_timed(
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
    });

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
    info.set_item("times_ns", times_ns)?;

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
    ties: TieHandling,
    make_penalty: F,
    use_fused: bool,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(f64, ndarray::Array1<f64>) -> Box<dyn Penalty> + Send,
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

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };
    if let Some(scales) = &scales_user {
        for j in 0..n_cols {
            pen_weights[j] /= scales[j];
        }
    }

    let glm = CoxPH::with_ties(time_arr, event_arr, ties);
    let make_pen = move |lam: f64| -> Box<dyn Penalty> { make_penalty(lam, pen_weights.clone()) };

    let cd_cfg = CdConfig {
        max_iter,
        tol,
        acceleration,
    };
    let lambdas_vec: Option<Vec<f64>> = lambdas.map(|a| a.as_array().to_vec());

    let (mut betas, report, times_ns) =
        py.allow_threads(|| match (scales_user.as_ref(), use_fused) {
            (Some(scales), true) => {
                let std_design = Standardized::new(csc, scales.clone());
                prox_newton_fused_solve_path_timed(
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
            (Some(scales), false) => {
                let std_design = Standardized::new(csc, scales.clone());
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
            (None, true) => prox_newton_fused_solve_path_timed(
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
            (None, false) => prox_newton_solve_path_timed(
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
        });

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
    info.set_item("times_ns", times_ns)?;

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
    ties: TieHandling,
    make_inner: F,
) -> PyResult<crate::ls::PathOutput<'py>>
where
    F: Fn(ArrayView1<'_, f64>, &Groups, f64, &ndarray::Array1<f64>) -> Box<dyn GroupPenalty> + Send,
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
    let groups = crate::ls::groups_from_labels(&labels)?;
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

    let csc = crate::ls::read_csc_arrays(n_rows, n_cols, data, indices, indptr)?;

    let scales_user: Option<Array1<f64>> = if standardize_x {
        Some(crate::ls::compute_csc_glmnet_scales(&csc))
    } else {
        None
    };

    let glm = CoxPH::with_ties(time_arr, event_arr, ties);

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

    let (mut betas, report, times_ns) = py.allow_threads(|| match scales_user.as_ref() {
        Some(scales) => {
            let std_design = Standardized::new(csc, scales.clone());
            prox_newton_block_solve_path_timed(
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
        None => prox_newton_block_solve_path_timed(
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
    });

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
    info.set_item("times_ns", times_ns)?;

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
pub(crate) fn solve_logistic_mcp_path_sparse<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
        true,
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
pub(crate) fn solve_logistic_scad_path_sparse<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_elastic_net_path_sparse<'py>(
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
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
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
        move |lam, w| Box::new(ElasticNet::with_weights(lam, alpha, w)),
        false,
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
pub(crate) fn solve_logistic_group_lasso_path_sparse<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
pub(crate) fn solve_logistic_group_mcp_path_sparse<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
        // M13.4c — native group-MCP block-CD inside the prox-Newton outer
        // loop. See solve_logistic_group_mcp_path for the full rationale.
        move |_beta, _groups, lam, group_w| {
            Box::new(GroupMcp::with_weights(lam, gamma, group_w.clone()))
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
pub(crate) fn solve_logistic_sparse_group_lasso_path_sparse<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
pub(crate) fn solve_logistic_sparse_group_mcp_path_sparse<'py>(
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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
    let coord_w_eff = crate::ls::build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

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
            // M14c.2 — native sparse-group MCP block-CD inside the
            // prox-Newton outer loop. `SparseGroupMcp::prox_group`
            // (Breheny & Huang 2015 §3 closed-form composition of
            // per-coord MCP + group-MCP) replaces the LLA-wrapped
            // weighted SparseGroupLasso surrogate. Sibling of M13.4c
            // for the plain group-MCP family.
            let _ = beta;
            let cw = crate::glm::split_coord_weights_per_group(coord_w_eff.view(), g);
            Box::new(SparseGroupMcp::with_coord_weights(
                lam,
                alpha,
                gamma,
                group_w.clone(),
                cw,
            ))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    a=3.7, alpha=0.5,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_logistic_sparse_group_scad_path_sparse<'py>(
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
    acceleration: Option<usize>,
    fit_intercept: bool,
    standardize_x: bool,
    max_outer: usize,
    outer_tol: f64,
) -> PyResult<crate::ls::PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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
    let coord_w_eff = crate::ls::build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

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
            let (gw, cw) = surrogate_sparse_group_scad(
                beta,
                g,
                lam,
                a,
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
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, gamma=3.0, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    gamma: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
        make_glm,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, a=3.7, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_scad_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    a: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
        make_glm,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, *, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_elastic_net_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    alpha: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(format!(
            "alpha must be in [0, 1]; got {alpha}"
        )));
    }
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
        make_glm,
        move |lam, w| Box::new(ElasticNet::with_weights(lam, alpha, w)),
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
        make_glm,
        |_beta, _g, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, gamma=3.0, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_group_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    gamma: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
        make_glm,
        // M13.4c — native group-MCP block-CD inside the prox-Newton outer
        // loop. See solve_logistic_group_mcp_path for the full rationale.
        move |_beta, _groups, lam, group_w| {
            Box::new(GroupMcp::with_weights(lam, gamma, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_sparse_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    y: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    alpha: f64,
    offset: Option<PyReadonlyArray1<f64>>,
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
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
        make_glm,
        move |_beta, _g, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    gamma=3.0, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_sparse_group_mcp_path_sparse<'py>(
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
    offset: Option<PyReadonlyArray1<f64>>,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
    let coord_w_eff = crate::ls::build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

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
        make_glm,
        move |beta, g, lam, group_w| {
            // M14c.2 — native sparse-group MCP block-CD inside the
            // prox-Newton outer loop. `SparseGroupMcp::prox_group`
            // (Breheny & Huang 2015 §3 closed-form composition of
            // per-coord MCP + group-MCP) replaces the LLA-wrapped
            // weighted SparseGroupLasso surrogate. Sibling of M13.4c
            // for the plain group-MCP family.
            let _ = beta;
            let cw = crate::glm::split_coord_weights_per_group(coord_w_eff.view(), g);
            Box::new(SparseGroupMcp::with_coord_weights(
                lam,
                alpha,
                gamma,
                group_w.clone(),
                cw,
            ))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, y, groups, *,
    a=3.7, alpha=0.5, offset=None,
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    fit_intercept=true, standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_poisson_sparse_group_scad_path_sparse<'py>(
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
    offset: Option<PyReadonlyArray1<f64>>,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
    let make_glm = poisson_glm_factory(offset, n_rows)?;
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
    let coord_w_eff = crate::ls::build_sparse_coord_weights(&user_coord, n_cols, fit_intercept);

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
        make_glm,
        move |beta, g, lam, group_w| {
            let (gw, cw) = surrogate_sparse_group_scad(
                beta,
                g,
                lam,
                a,
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
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_mcp_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    gamma: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |lam, w| Box::new(Mcp::with_weights(lam, gamma, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, *, a=3.7,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_scad_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    a: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |lam, w| Box::new(Scad::with_weights(lam, a, w)),
        true,
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_group_lasso_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        |_beta, _g, lam, group_w| Box::new(GroupLasso::with_weights(lam, group_w.clone())),
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *, gamma=3.0,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_group_mcp_path_sparse<'py>(
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
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        // M13.4c — native group-MCP block-CD inside the prox-Newton outer
        // loop. See solve_logistic_group_mcp_path for the full rationale.
        move |_beta, _groups, lam, group_w| {
            Box::new(GroupMcp::with_weights(lam, gamma, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *, alpha=0.5,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3, weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_sparse_group_lasso_path_sparse<'py>(
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
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |_beta, _g, lam, group_w| {
            Box::new(SparseGroupLasso::with_weights(lam, alpha, group_w.clone()))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *,
    gamma=3.0, alpha=0.5,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_sparse_group_mcp_path_sparse<'py>(
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
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
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

    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |beta, g, lam, group_w| {
            // M14c.2 — native sparse-group MCP block-CD; see
            // solve_logistic_sparse_group_mcp_path for the full rationale.
            let _ = beta;
            let cw_per_group = crate::glm::split_coord_weights_per_group(coord_w.view(), g);
            Box::new(SparseGroupMcp::with_coord_weights(
                lam,
                alpha,
                gamma,
                group_w.clone(),
                cw_per_group,
            ))
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    x_data, x_indices, x_indptr, n_rows, n_cols, time, event, groups, *,
    a=3.7, alpha=0.5,
    ties="breslow",
    lambdas=None, n_lambdas=100, lambda_min_ratio=1e-3,
    weights=None, coord_weights=None,
    max_iter=100, tol=1e-6, acceleration=Some(5),
    standardize_x=false, max_outer=10, outer_tol=1e-6,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_cox_sparse_group_scad_path_sparse<'py>(
    py: Python<'py>,
    x_data: PyReadonlyArray1<f64>,
    x_indices: PyReadonlyArray1<i64>,
    x_indptr: PyReadonlyArray1<i64>,
    n_rows: usize,
    n_cols: usize,
    time: PyReadonlyArray1<f64>,
    event: PyReadonlyArray1<f64>,
    groups: PyReadonlyArray1<i64>,
    a: f64,
    alpha: f64,
    ties: &str,
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
) -> PyResult<crate::ls::PathOutput<'py>> {
    if a <= 2.0 {
        return Err(PyValueError::new_err(format!(
            "SCAD shape parameter `a` must be > 2; got {a}"
        )));
    }
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

    let ties = parse_cox_ties(ties)?;
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
        ties,
        move |beta, g, lam, group_w| {
            let (gw, cw) =
                surrogate_sparse_group_scad(beta, g, lam, a, alpha, group_w.view(), coord_w.view());
            Box::new(SparseGroupLasso::with_coord_weights(lam, alpha, gw, cw))
        },
    )
}
