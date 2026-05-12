//! Penalty factories: build a [`Penalty`] from a per-coordinate weight
//! vector at call time. The outer solver in glasso (and any other
//! algorithm whose inner subproblem rebuilds the penalty per
//! iteration with new weights) takes a factory rather than a
//! concrete penalty, so it stays generic over `Lasso | Mcp | Scad |
//! ElasticNet`.
//!
//! Mirrored by [`GroupPenaltyFactory`] for the block-separable case.

use super::{ElasticNet, GroupLasso, GroupMcp, GroupPenalty, Mcp, Penalty, Scad};
use ndarray::Array1;

pub trait ScalarPenaltyFactory: Sync + Send {
    fn build(&self, weights: Array1<f64>) -> Box<dyn Penalty>;
}

/// L1 / soft-threshold penalty factory (elastic net at α = 1).
pub struct LassoFactory {
    pub lambda: f64,
}

impl ScalarPenaltyFactory for LassoFactory {
    fn build(&self, weights: Array1<f64>) -> Box<dyn Penalty> {
        Box::new(ElasticNet::with_weights(self.lambda, 1.0, weights))
    }
}

pub struct McpFactory {
    pub lambda: f64,
    pub gamma: f64,
}

impl ScalarPenaltyFactory for McpFactory {
    fn build(&self, weights: Array1<f64>) -> Box<dyn Penalty> {
        Box::new(Mcp::with_weights(self.lambda, self.gamma, weights))
    }
}

pub struct ScadFactory {
    pub lambda: f64,
    pub a: f64,
}

impl ScalarPenaltyFactory for ScadFactory {
    fn build(&self, weights: Array1<f64>) -> Box<dyn Penalty> {
        Box::new(Scad::with_weights(self.lambda, self.a, weights))
    }
}

pub trait GroupPenaltyFactory: Sync + Send {
    fn build(&self, weights: Array1<f64>) -> Box<dyn GroupPenalty>;
}

pub struct GroupLassoFactory {
    pub lambda: f64,
}

impl GroupPenaltyFactory for GroupLassoFactory {
    fn build(&self, weights: Array1<f64>) -> Box<dyn GroupPenalty> {
        Box::new(GroupLasso::with_weights(self.lambda, weights))
    }
}

pub struct GroupMcpFactory {
    pub lambda: f64,
    pub gamma: f64,
}

impl GroupPenaltyFactory for GroupMcpFactory {
    fn build(&self, weights: Array1<f64>) -> Box<dyn GroupPenalty> {
        Box::new(GroupMcp::with_weights(self.lambda, self.gamma, weights))
    }
}
