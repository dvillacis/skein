//! skein-core: weighted structured nonconvex sparse models.
//!
//! Public surface is intentionally minimal in v0.1: traits + a few concrete
//! impls used by the solver. Everything is `Sync` so the solver can dispatch
//! group-wise work across Rayon threads.

// Anchor the BLAS provider so its build-script `cargo:rustc-link-lib`
// directive reaches the final binary. Without an explicit `use`, rustc's
// dead-code prune skips emitting the link line even though the crate is in
// `Cargo.toml`. `accelerate-src` itself is empty — its only job is the
// build-script directive — so this `use` is purely a linkage anchor.
#[cfg(feature = "blas-accelerate")]
use accelerate_src as _;

pub mod datafit;
pub mod design;
pub mod groups;
pub mod penalty;
pub mod prox;
pub mod solver;
pub mod standardize;

pub use datafit::Datafit;
pub use design::{DenseMatrix, DesignMatrix, SparseCSC, Standardized};
pub use groups::Groups;
pub use penalty::{GroupPenalty, Penalty};
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
