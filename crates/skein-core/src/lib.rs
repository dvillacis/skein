//! skein-core: weighted structured nonconvex sparse models.
//!
//! Public surface is intentionally minimal in v0.1: traits + a few concrete
//! impls used by the solver. Everything is `Sync` so the solver can dispatch
//! group-wise work across Rayon threads.

pub mod design;
pub mod groups;
pub mod prox;
pub mod penalty;
pub mod datafit;
pub mod solver;
pub mod standardize;

pub use design::{DenseMatrix, DesignMatrix, SparseCSC, Standardized};
pub use groups::Groups;
pub use penalty::{GroupPenalty, Penalty};
pub use datafit::Datafit;
pub use standardize::{
    destandardize, destandardize_path, rescale_weights_for_standardize, standardize,
    StandardizationStats, StandardizeConfig,
};

#[derive(Debug, thiserror::Error)]
pub enum SkeinError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("solver did not converge after {iter} iterations")]
    NotConverged { iter: usize },
}

pub type Result<T> = std::result::Result<T, SkeinError>;
