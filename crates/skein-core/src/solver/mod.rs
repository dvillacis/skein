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
mod convex_region;
mod glasso;
mod glasso_admm;
mod lla;
mod path;
mod path_lla;
mod prox_newton;
mod prox_newton_block;

// --- Stable public surface (v1.0 freeze; see docs/extending/rust-api.md).
pub use block_cd::{
    block_cd_solve_subset, block_cd_solve_subset_parallel, group_lipschitz, group_lipschitz_cache,
};
pub use block_path::{
    block_lambda_max, solve_block_path, solve_block_path_timed, BlockPathConfig, BlockPathReport,
};
pub use block_path_lla::{solve_block_path_lla, BlockPathLLAReport};
pub use cd::{cd_solve, CdConfig, CdReport};
pub use convex_region::{group_convex_min_idx, scalar_convex_min_idx};
pub use glasso::{glasso_solve, GlassoConfig, GlassoReport};
pub use glasso_admm::{joint_glasso_solve, JointGlassoConfig, JointGlassoReport};
pub use lla::{
    broadcast_group_weights_to_coord, cmcp_lambda_max, lla_solve, surrogate_sparse_group_scad,
    surrogate_weights_bridge, surrogate_weights_cmcp, surrogate_weights_gel,
    surrogate_weights_group_mcp, surrogate_weights_group_scad, LLAReport,
};
pub use path::{
    lambda_grid, lambda_max, solve_path, solve_path_timed, PathConfig, PathReport, Screening,
};
pub use path_lla::{solve_path_lla, PathLLAReport};
pub use prox_newton::{
    prox_newton_fused_solve_path, prox_newton_fused_solve_path_timed, prox_newton_solve,
    prox_newton_solve_path, prox_newton_solve_path_timed, ProxNewtonPathReport, ProxNewtonReport,
};
pub use prox_newton_block::{
    prox_newton_block_solve_path, prox_newton_block_solve_path_timed, ProxNewtonBlockPathReport,
};

// --- Crate-internal solver helpers. Demoted from the v0.x public surface
// during the v1.0 API freeze; intra-crate callers still reach them either
// here (via the short `crate::solver::*` path) or directly through
// `crate::solver::<module>::*`. Re-export gated under `cfg(test)` because
// `block_cd_solve` is only consumed via the short path from `#[cfg(test)]`
// modules in `penalty/sparse_group_*` and `penalty/group_elastic_net`.
#[cfg(test)]
pub(crate) use block_cd::block_cd_solve;
