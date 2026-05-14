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

mod glasso;
mod glm;
mod ls;
mod mmap_chunked;
mod multinomial;
mod multitask;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ls::solve_mcp_ls, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_scad_ls, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_mcp_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_scad_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_elastic_net_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_bridge_ls_path, m)?)?;
    m.add_function(wrap_pyfunction!(ls::solve_bridge_ls_path_sparse, m)?)?;
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
    // M11 — graphical lasso family.
    m.add_function(wrap_pyfunction!(glasso::solve_glasso_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_glasso_mcp, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_glasso_scad, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_joint_glasso_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(glasso::solve_joint_glasso_mcp, m)?)?;
    Ok(())
}
