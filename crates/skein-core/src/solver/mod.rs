//! Solvers.
//!
//! v0.1 ships a minimal coordinate-descent solver for separable penalties as
//! a smoke test for the trait wiring. The headline algorithm — LLA outer +
//! group block-CD inner with a working set — lands in a follow-up; the
//! traits here are what it will plug into.

mod block_cd;
mod block_path;
mod block_path_lla;
mod cd;
mod lla;
mod path;
mod path_lla;
mod prox_newton;
mod prox_newton_block;

pub use block_cd::{block_cd_solve, block_cd_solve_subset, block_cd_solve_subset_parallel};
pub use block_path::{block_lambda_max, solve_block_path, BlockPathConfig, BlockPathReport};
pub use block_path_lla::{solve_block_path_lla, BlockPathLLAReport};
pub use cd::{
    cd_solve, cd_solve_subset, cd_solve_warm, cd_solve_warm_with_residual, CdConfig, CdReport,
};
pub use lla::{
    lla_solve, surrogate_sparse_group_mcp, surrogate_sparse_group_scad, surrogate_weights_bridge,
    surrogate_weights_group_mcp, surrogate_weights_group_scad, LLAReport,
};
pub use path::{lambda_grid, lambda_max, solve_path, PathConfig, PathReport, Screening};
pub use path_lla::{solve_path_lla, PathLLAReport};
pub use prox_newton::{
    prox_newton_solve, prox_newton_solve_path, ProxNewtonPathReport, ProxNewtonReport,
};
pub use prox_newton_block::{prox_newton_block_solve_path, ProxNewtonBlockPathReport};
