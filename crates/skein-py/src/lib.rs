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

mod convex_region;
mod glasso;
mod glm;
mod ls;
mod mmap_chunked;
mod multinomial;
mod multitask;
mod orthonormalize;

use pyo3::prelude::*;

/// Returns the BLAS feature flags this wheel was built with.
///
/// Each entry corresponds to a `--features=<name>` passed to `maturin
/// develop` / `cibuildwheel`'s `MATURIN_PEP517_ARGS`:
///
/// - `"blas-accelerate"` — macOS Accelerate framework.
/// - `"blas-openblas"`   — system OpenBLAS (via the system shared lib).
///
/// An empty list means the wheel uses ndarray's pure-Rust `matrixmultiply`
/// fallback — correct, but ~3× slower on the inner-CD hot path than
/// either hardware-BLAS path. Currently the case on Windows wheels;
/// ROADMAP P3 tracks closing that gap.
///
/// Wired through to Python as `skein_glm.__build_features__`. P3
/// acceptance criterion.
#[pyfunction]
fn build_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "blas-accelerate") {
        features.push("blas-accelerate");
    }
    if cfg!(feature = "blas-openblas") {
        features.push("blas-openblas");
    }
    features
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_features, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_mcp_ls, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_scad_ls, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_scad_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_elastic_net_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_bridge_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_bridge_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_cmcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_gel_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_lasso_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_elastic_net_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_scad_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_sparse_group_lasso_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_sparse_group_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_sparse_group_scad_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_lasso_ls_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(multitask::solve_multitask_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_scad_ls_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_elastic_net_ls_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_elastic_net_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_huber_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_huber_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_sparse_group_lasso_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_sparse_group_mcp_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_sparse_group_scad_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_elastic_net_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_sparse_group_lasso_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_sparse_group_mcp_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_sparse_group_scad_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_sparse_group_lasso_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_sparse_group_mcp_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_sparse_group_scad_path, m)?)?;
    m.add_function(wrap_pyfunction!(glm::cox_surrogate_weights_at, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_mcp_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_scad_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_elastic_net_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_lasso_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        ls::solve_group_elastic_net_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_mcp_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_group_scad_ls_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        ls::solve_sparse_group_lasso_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ls::solve_sparse_group_mcp_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        ls::solve_sparse_group_scad_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_lasso_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_mcp_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_scad_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_elastic_net_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_scad_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multinomial::solve_multinomial_elastic_net_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_lasso_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_mcp_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_scad_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        multitask::solve_multitask_elastic_net_ls_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_logistic_scad_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_elastic_net_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_sparse_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_sparse_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_logistic_sparse_group_scad_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_poisson_scad_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_elastic_net_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_sparse_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_sparse_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_poisson_sparse_group_scad_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_scad_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_group_lasso_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(glm::solve_cox_group_mcp_path_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_cox_sparse_group_lasso_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_cox_sparse_group_mcp_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        glm::solve_cox_sparse_group_scad_path_sparse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(mmap_chunked::solve_mcp_ls_path_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_logistic_mcp_path_mmap,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_mcp_ls_path_mmap_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_logistic_mcp_path_mmap_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_mcp_ls_path_chunked,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_mcp_ls_path_chunked_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_logistic_mcp_path_chunked,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        mmap_chunked::solve_logistic_mcp_path_chunked_f32,
        m
    )?)?;
    // Per-block group orthonormalization (grpreg `orthogonalize`).
    m.add_function(wrap_pyfunction!(
        orthonormalize::orthonormalize_groups_dense,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        orthonormalize::back_transform_coefs_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(orthonormalize::back_transform_coefs, m)?)?;
    // Post-fit nonconvex convex-region detection (grpreg `convex.min`).
    m.add_function(wrap_pyfunction!(convex_region::convex_min_idx_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(convex_region::convex_min_idx_group, m)?)?;
    m.add_function(wrap_pyfunction!(convex_region::group_lipschitz_dense, m)?)?;
    m.add_function(wrap_pyfunction!(convex_region::group_lipschitz_sparse, m)?)?;
    // M11 — graphical lasso family.
    m.add_function(wrap_pyfunction!(glasso::solve_glasso_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_glasso_mcp, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_glasso_scad, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_joint_glasso_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_joint_glasso_mcp, m)?)?;
    Ok(())
}
