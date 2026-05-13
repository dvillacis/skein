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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::Groups;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn lasso_factory_builds_alpha_one_elastic_net() {
        // LassoFactory must round-trip to ElasticNet at α=1, which
        // means its prox is bit-for-bit soft-threshold.
        let factory = LassoFactory { lambda: 0.5 };
        let pen = factory.build(array![1.0, 2.0]);
        // Threshold at j=0: step·λ·w = 1·0.5·1 = 0.5 ⇒ z=0.3 → 0.
        assert_abs_diff_eq!(pen.prox_coord(0, 0.3, 1.0), 0.0);
        // Threshold at j=1: step·λ·w = 1·0.5·2 = 1.0 ⇒ z=0.8 → 0.
        assert_abs_diff_eq!(pen.prox_coord(1, 0.8, 1.0), 0.0);
        // Above threshold at j=1: z=1.5 → 1.5 − 1.0 = 0.5.
        assert_abs_diff_eq!(pen.prox_coord(1, 1.5, 1.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn mcp_factory_builds_mcp_with_supplied_weights() {
        let factory = McpFactory {
            lambda: 0.4,
            gamma: 2.5,
        };
        let pen = factory.build(array![0.5, 1.0]);
        // Reference: Mcp::with_weights yields the same value() for any β.
        let reference = Mcp::with_weights(0.4, 2.5, array![0.5, 1.0]);
        let beta = array![0.3_f64, -0.7];
        assert_abs_diff_eq!(
            pen.value(beta.view()),
            reference.value(beta.view()),
            epsilon = 1e-12
        );
    }

    #[test]
    fn scad_factory_builds_scad_with_supplied_weights() {
        let factory = ScadFactory {
            lambda: 0.4,
            a: 3.7,
        };
        let pen = factory.build(array![0.5, 1.0]);
        let reference = Scad::with_weights(0.4, 3.7, array![0.5, 1.0]);
        let beta = array![0.3_f64, -1.2];
        assert_abs_diff_eq!(
            pen.value(beta.view()),
            reference.value(beta.view()),
            epsilon = 1e-12
        );
    }

    #[test]
    fn group_lasso_factory_builds_group_lasso_with_supplied_weights() {
        let factory = GroupLassoFactory { lambda: 0.5 };
        let pen = factory.build(array![1.0, 2.0]);
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = array![3.0_f64, 4.0, 0.0, 1.0];
        let reference = GroupLasso::with_weights(0.5, array![1.0, 2.0]);
        assert_abs_diff_eq!(
            pen.value(beta.view(), &groups),
            reference.value(beta.view(), &groups),
            epsilon = 1e-12
        );
    }

    #[test]
    fn group_mcp_factory_builds_group_mcp_with_supplied_weights() {
        let factory = GroupMcpFactory {
            lambda: 0.5,
            gamma: 3.0,
        };
        let pen = factory.build(array![1.0, 2.0]);
        let groups = Groups::contiguous_blocks(4, 2);
        let beta = array![0.3_f64, 0.4, 1.0, 0.0];
        let reference = GroupMcp::with_weights(0.5, 3.0, array![1.0, 2.0]);
        assert_abs_diff_eq!(
            pen.value(beta.view(), &groups),
            reference.value(beta.view(), &groups),
            epsilon = 1e-12
        );
    }

    #[test]
    fn factories_propagate_weights_via_build_argument() {
        // The factory must use the *build-time* weights, not anything
        // captured at construction. Catches a "factory ignores its arg"
        // bug.
        let factory = McpFactory {
            lambda: 1.0,
            gamma: 3.0,
        };
        let pen_w0 = factory.build(Array1::from_elem(2, 0.0));
        let pen_w1 = factory.build(Array1::from_elem(2, 1.0));
        // weight 0 ⇒ identity prox.
        assert_abs_diff_eq!(pen_w0.prox_coord(0, 0.5, 1.0), 0.5, epsilon = 1e-12);
        // weight 1 ⇒ thresholded.
        assert_abs_diff_eq!(pen_w1.prox_coord(0, 0.5, 1.0), 0.0, epsilon = 1e-12);
    }
}
