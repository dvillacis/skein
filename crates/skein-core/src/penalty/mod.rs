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

    /// Per-feature L1-effective penalty multipliers (length = n_features).
    /// These are the multipliers on the L1 part of the penalty — i.e.,
    /// the threshold values that the gradient must respect at β = 0
    /// (`|grad_j| ≤ λ · w_j` for stationarity). For pure lasso /
    /// MCP / SCAD this is the user-supplied per-feature weight; for
    /// elastic net it's `α · w_raw` since the L2 part contributes 0
    /// to the active-set boundary at β = 0.
    fn weights(&self) -> ArrayView1<'_, f64>;

    /// Smooth-penalty correction subtracted from the LS-form dual obj.
    ///
    /// ```text
    ///     gap = primal − (D_datafit(scaled_θ) − dual_correction(β))
    /// ```
    ///
    /// For penalties that are pure-L1 over their support
    /// (lasso, and the LLA-linearised surrogates of MCP / SCAD that
    /// the path solver actually solves at each LLA step), the
    /// correction is `0`. Elastic net at α < 1 returns
    /// `½ · λ · (1−α) · Σ w_raw_j · β_j²` — the smooth-quadratic
    /// part of the primal that the L1 dual can't account for.
    fn dual_correction(&self, _beta: ArrayView1<'_, f64>) -> f64 {
        0.0
    }

    /// Whether the lasso-form duality gap (built from this penalty's
    /// `weights()` and `dual_correction()`) is a valid stopping
    /// criterion for the path solver.
    ///
    /// `true` ⇒ `Penalty(β) ≤ λ · Σ w_j |β_j|` is **tight at the
    /// solution** for any `β` — i.e. the penalty's effective L1
    /// envelope coincides with the penalty itself in a neighbourhood
    /// of the optimum, so `primal − D_lasso(θ)` collapses to zero at
    /// the true `β*`. Holds for lasso / elastic net (where the
    /// penalty *is* the L1 + L2 envelope).
    ///
    /// `false` ⇒ the L1 envelope is a strict upper bound on the
    /// penalty (concave penalties: MCP / SCAD / bridge with q < 1),
    /// so `primal − D_lasso` doesn't go to zero at `β*` and the
    /// "gap" is misleading as a convergence test. The path solver
    /// falls back to prox-gradient stationarity for these.
    fn has_lasso_form_dual_gap(&self) -> bool {
        false
    }
}

pub trait GroupPenalty: Sync + Send {
    fn value(&self, beta: ArrayView1<f64>, groups: &Groups) -> f64;

    /// In-place block prox for group `g`. `block` aliases `β` restricted to
    /// the group's feature indices.
    fn prox_group(&self, g: usize, block: ArrayViewMut1<f64>, step: f64);

    /// Per-group penalty multipliers (length = n_groups).
    fn weights(&self) -> ArrayView1<'_, f64>;
}
