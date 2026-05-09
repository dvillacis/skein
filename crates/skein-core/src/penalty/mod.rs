//! Penalty traits and implementations.
//!
//! Two traits because the prox signature differs:
//! - `Penalty`: separable scalar penalties (lasso, MCP, SCAD).
//! - `GroupPenalty`: block-separable penalties over a `Groups` partition.
//!
//! Both expose a `weights()` accessor so the solver doesn't need to know
//! whether weights are uniform, adaptive, or supplied externally.

mod elastic_net;
mod group_elastic_net;
mod group_lasso;
mod group_mcp;
mod mcp;
mod scad;
mod sparse_group_lasso;

pub use elastic_net::ElasticNet;
pub use group_elastic_net::GroupElasticNet;
pub use group_lasso::GroupLasso;
pub use group_mcp::GroupMcp;
pub use mcp::Mcp;
pub use scad::Scad;
pub use sparse_group_lasso::SparseGroupLasso;

use crate::groups::Groups;
use ndarray::{ArrayView1, ArrayViewMut1};

pub trait Penalty: Sync + Send {
    /// Total penalty value `Σ_j w_j · p(β_j)`.
    fn value(&self, beta: ArrayView1<f64>) -> f64;

    /// Scalar prox at coordinate `j`.
    fn prox_coord(&self, j: usize, z: f64, step: f64) -> f64;

    /// Per-feature penalty multipliers (length = n_features).
    fn weights(&self) -> ArrayView1<'_, f64>;
}

pub trait GroupPenalty: Sync + Send {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64;

    /// In-place block prox for group `g`. `block` aliases `β` restricted to
    /// the group's feature indices.
    fn prox_group(&self, g: usize, block: ArrayViewMut1<f64>, step: f64);

    /// Per-group penalty multipliers (length = n_groups).
    fn weights(&self) -> ArrayView1<'_, f64>;
}
