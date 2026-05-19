//! Detection of the convex region along a nonconvex regularization path.
//!
//! For a nonconvex penalty with curvature parameter γ (MCP) or `a` (SCAD),
//! the local objective `L(β) + P(β)` ceases to be locally convex once the
//! penalty's negative second derivative dominates the data-fit's positive
//! curvature in the active directions.
//!
//! grpreg and ncvreg report `convex.min`: the smallest λ-index along the
//! (decreasing-λ) path at which this happens. Beyond `convex.min` the warm-
//! start chain can still march down the grid, but the solution is no longer
//! the global minimizer of a locally convex problem — the user is warned
//! and may opt to restrict cross-validation / IC selection to the convex
//! portion.
//!
//! Skein computes the same diagnostic as a post-fit utility, leaving the
//! solver hot path untouched. For an active feature `j` (resp. group `g`),
//! the local-convexity check is
//!
//! * scalar MCP/SCAD:  `col_lip[j] ≥ penalty_concavity`
//! * group  MCP/SCAD:  `group_lip[g] ≥ penalty_concavity`
//!
//! where `penalty_concavity` is `1/γ` for MCP and `1/(a−1)` for SCAD. For
//! orthonormalized blocks the per-group / per-column Lipschitz is identically
//! `1`, so the check collapses to a single inequality on the penalty
//! parameter alone.

use crate::groups::Groups;
use ndarray::ArrayView2;

/// Curvature parameter of a nonconvex penalty.
///
/// Use [`PenaltyConcavity::as_value`] to turn into the scalar that the
/// convex-region checks compare against per-feature / per-group Lipschitz
/// bounds.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // test-only since v1.0 demotion
pub(crate) enum PenaltyConcavity {
    /// MCP-like, with `gamma > 0`. Concavity = `1 / gamma`.
    Mcp { gamma: f64 },
    /// SCAD-like, with `a > 2`. Concavity = `1 / (a − 1)`.
    Scad { a: f64 },
    /// Explicit concavity value — caller has already done the conversion.
    /// Useful for composite/exponential penalties whose curvature does not
    /// reduce to a single γ.
    Explicit(f64),
}

impl PenaltyConcavity {
    /// The scalar bound that per-feature / per-group Lipschitz must equal
    /// or exceed for local convexity. Returns `0.0` (always convex) for
    /// degenerate inputs (`gamma = ∞`, `a` very large).
    #[allow(dead_code)] // test-only since v1.0 demoted PenaltyConcavity to pub(crate)
    pub fn as_value(&self) -> f64 {
        match *self {
            PenaltyConcavity::Mcp { gamma } => {
                if gamma > 0.0 && gamma.is_finite() {
                    1.0 / gamma
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Scad { a } => {
                if a > 1.0 && a.is_finite() {
                    1.0 / (a - 1.0)
                } else {
                    0.0
                }
            }
            PenaltyConcavity::Explicit(c) => c.max(0.0),
        }
    }
}

/// Smallest λ-index along the path at which an active *coordinate* violates
/// the local-convexity bound for a scalar nonconvex penalty.
///
/// `betas` is `(n_lambdas, n_features)` from a path solver, row `k` at
/// `λ_k` (path is decreasing in λ). `col_lip[j]` is the data-fit's
/// Lipschitz at coordinate `j` (typically `‖X_{:,j}‖² / n` for LS).
/// Coordinates with `|β_{k,j}| ≤ zero_tol` are treated as inactive.
///
/// Returns `None` when the entire path stays locally convex (including when
/// `penalty_concavity ≤ 0`, i.e. a convex penalty).
pub fn scalar_convex_min_idx(
    betas: ArrayView2<f64>,
    col_lip: &[f64],
    penalty_concavity: f64,
    zero_tol: f64,
) -> Option<usize> {
    if penalty_concavity <= 0.0 {
        return None;
    }
    assert_eq!(
        col_lip.len(),
        betas.ncols(),
        "col_lip length {} does not match n_features {}",
        col_lip.len(),
        betas.ncols()
    );
    for k in 0..betas.nrows() {
        let beta_k = betas.row(k);
        for j in 0..betas.ncols() {
            if beta_k[j].abs() > zero_tol && col_lip[j] < penalty_concavity {
                return Some(k);
            }
        }
    }
    None
}

/// Smallest λ-index along the path at which an active *group* violates the
/// local-convexity bound for a group nonconvex penalty.
///
/// `group_lip[g]` is the per-group operator-norm Lipschitz `‖X_g‖_op² / n`
/// (see [`crate::solver::group_lipschitz_cache`]). A group is "active" when
/// any coefficient inside it exceeds `zero_tol` in magnitude.
///
/// Returns `None` when the entire path stays locally convex.
pub fn group_convex_min_idx(
    betas: ArrayView2<f64>,
    groups: &Groups,
    group_lip: &[f64],
    penalty_concavity: f64,
    zero_tol: f64,
) -> Option<usize> {
    if penalty_concavity <= 0.0 {
        return None;
    }
    let n_groups = groups.n_groups();
    assert_eq!(
        group_lip.len(),
        n_groups,
        "group_lip length {} does not match n_groups {}",
        group_lip.len(),
        n_groups
    );
    for k in 0..betas.nrows() {
        let beta_k = betas.row(k);
        for (g, &lip_g) in group_lip.iter().enumerate().take(n_groups) {
            if lip_g >= penalty_concavity {
                continue;
            }
            let active = groups.group(g).iter().any(|&j| beta_k[j].abs() > zero_tol);
            if active {
                return Some(k);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn convex_penalty_returns_none() {
        let betas = array![[1.0, -1.0], [0.5, -0.5]];
        let col_lip = vec![1.0, 1.0];
        assert_eq!(
            scalar_convex_min_idx(betas.view(), &col_lip, 0.0, 1e-8),
            None
        );
        assert_eq!(
            scalar_convex_min_idx(betas.view(), &col_lip, -0.5, 1e-8),
            None
        );
    }

    #[test]
    fn scalar_detects_first_violator() {
        // col_lip = [2.0, 0.5]. concavity = 1.0 (so MCP γ = 1).
        // At k=0: only β_0 active (|β_1| < zero_tol) — col_lip[0]=2 ≥ 1 → convex.
        // At k=1: β_1 becomes active and col_lip[1] = 0.5 < 1 → non-convex.
        let betas = array![[1.0, 1e-10], [1.0, 0.3]];
        let col_lip = vec![2.0, 0.5];
        assert_eq!(
            scalar_convex_min_idx(betas.view(), &col_lip, 1.0, 1e-8),
            Some(1)
        );
    }

    #[test]
    fn scalar_inactive_violator_is_ignored() {
        // col_lip[1] < concavity but β_1 is exactly zero everywhere — convex.
        let betas = array![[0.5, 0.0], [0.7, 0.0]];
        let col_lip = vec![2.0, 0.5];
        assert_eq!(
            scalar_convex_min_idx(betas.view(), &col_lip, 1.0, 1e-8),
            None
        );
    }

    #[test]
    fn group_detects_violator() {
        // 4 features in two contiguous groups of 2.
        let groups = Groups::contiguous_blocks(4, 2);
        // group_lip = [2.0, 0.5], concavity 1.0.
        // k=0: only group 0 active (all of group 1's coords < zero_tol) → convex.
        // k=1: group 1 becomes active → violates.
        let betas = array![[1.0, -0.5, 1e-12, 0.0], [1.0, -0.5, 0.4, 0.2]];
        let group_lip = vec![2.0, 0.5];
        assert_eq!(
            group_convex_min_idx(betas.view(), &groups, &group_lip, 1.0, 1e-8),
            Some(1)
        );
    }

    #[test]
    fn group_path_fully_convex_returns_none() {
        let groups = Groups::contiguous_blocks(4, 2);
        let betas: Array2<f64> = array![[1.0, -0.5, 0.4, 0.2], [0.7, -0.3, 0.1, 0.05]];
        let group_lip = vec![3.0, 3.0]; // both >= concavity 1.0
        assert_eq!(
            group_convex_min_idx(betas.view(), &groups, &group_lip, 1.0, 1e-8),
            None
        );
    }

    #[test]
    fn penalty_concavity_helpers() {
        assert!((PenaltyConcavity::Mcp { gamma: 3.0 }.as_value() - 1.0 / 3.0).abs() < 1e-12);
        assert!((PenaltyConcavity::Scad { a: 3.7 }.as_value() - 1.0 / 2.7).abs() < 1e-12);
        assert_eq!(
            PenaltyConcavity::Mcp {
                gamma: f64::INFINITY
            }
            .as_value(),
            0.0
        );
        assert!((PenaltyConcavity::Explicit(0.25).as_value() - 0.25).abs() < 1e-12);
        assert_eq!(PenaltyConcavity::Explicit(-1.0).as_value(), 0.0);
    }
}
