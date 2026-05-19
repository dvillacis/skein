# Stable Rust API contract

`skein-core` is the foundation crate that downstream Rust users
(per-paper crates, custom benchmarks, alternative Python bindings)
depend on. The list below is the **v1.0 freeze surface** — every item
here will follow semver from v1.0 onward: minor releases add only;
breaking changes wait for a major bump and ship with at least one
minor release of deprecation warnings.

Items not listed here are `pub(crate)`; if you find yourself wanting
one, file an issue and we'll consider promoting it. The internal
surface follows no stability guarantee and may be reshuffled freely
in any release.

## Trait surfaces

The five extension traits are the headline contract — implementing
them must continue to work.

- `skein_core::DesignMatrix` — column-access trait for storage
  backends. The required methods are `n_samples`, `n_features`,
  `matvec`, `rmatvec`, `col_dot`, `col_sq_norm`, `columns`. The
  default-provided `col_axpy`, `col_dot_weighted`, `col_sq_norm_weighted`,
  `weighted_col_sq_norms`, `col_axpy_weighted` may be overridden for
  performance. New required methods will not be added without a major
  version bump and a deprecation path.
- `skein_core::Datafit` — loss-function trait. Same stability
  guarantee.
- `skein_core::datafit::GlmDatafit` — the GLM extension trait
  (`surrogate_at` returns the prox-Newton weighted-LS surrogate,
  `loss` evaluates the original non-quadratic objective for outer-loop
  reporting).
- `skein_core::Penalty` — separable scalar-penalty trait
  (`prox_coord`, `value`, `weights`, `n_features`).
- `skein_core::GroupPenalty` — block-separable group-penalty trait
  (`prox_group`, `value`, `weights`).
- `skein_core::penalty::{ScalarPenaltyFactory, GroupPenaltyFactory}`
  — λ-parametric penalty constructors used by the multi-population
  glasso ADMM path. Implementing types build a `Box<dyn Penalty>` (or
  `Box<dyn GroupPenalty>`) from a single `λ` to keep per-λ allocation
  off the hot loop.

## Concrete design backends

All composable with any `D: DesignMatrix` wrapper:

- `skein_core::DenseMatrix`, `SparseCSC` — in-memory backends.
- `skein_core::Standardized<D>`, `Augmented<D>`,
  `MultiTaskDesign<D>` — lazy wrappers. `Standardized<D>` is the
  centering/scaling wrapper; `Augmented<D>` is the elastic-net
  `[X; √λ₂ I]` trick; `MultiTaskDesign<D>` virtually stacks
  multi-response data.
- `skein_core::MmapMatrix`, `MmapMatrixF32`, `Chunked<C>` — out-of-RAM
  backends.
- `skein_core::groups::Groups` — group-partition representation.
- `skein_core::design::{BlockBackTransform, orthonormalize_groups_dense}`
  — per-group orthonormalization preprocess + postprocess (Breheny–Huang
  pipeline; returned `BlockBackTransform` carries the per-group
  `T_g = √n · L_g^{-T}` and exposes `apply_to_coefs` /
  `apply_to_coefs_path`).

## Concrete datafits

- `skein_core::datafit::LeastSquares`, `BinomialLogit`, `PoissonLog`,
  `CoxPH` (with `TieHandling` enum — Breslow / Efron), `MultinomialLogit`,
  `Huber`. Standard constructors: `new`, `with_sample_weights`, etc.
  (See the rustdoc for each type.)

## Concrete penalties

Scalar penalties:

- `skein_core::penalty::{Mcp, Scad, ElasticNet}`.

Group penalties:

- `skein_core::penalty::{GroupLasso, GroupMcp, GroupScad,
  GroupElasticNet}`.

Sparse-group penalties (block + within-block sparsity):

- `skein_core::penalty::{SparseGroupLasso, SparseGroupMcp,
  SparseGroupScad}`.

Penalty factories for the λ-parametric ADMM paths:

- `skein_core::penalty::{LassoFactory, McpFactory, ScadFactory,
  GroupLassoFactory, GroupMcpFactory}`.

All concrete penalties expose `new`, `with_weights`, and (where
relevant) `with_coord_weights` constructors that respect the
per-feature / per-group weighting contract.

## Algorithm entry points

The path-level solvers that downstream estimators dispatch to:

- `skein_core::solver::cd_solve` — scalar CD on a separable penalty
  with a fixed λ.
- `skein_core::solver::solve_path` — scalar CD path with strong-rule
  screening, Anderson acceleration, gap-safe dual checks (`Screening`
  enum).
- `skein_core::solver::solve_path_lla` — scalar LLA outer wrapping
  `solve_path` (powers bridge / adaptive lasso / cMCP / gel).
- `skein_core::solver::solve_block_path` — group block-CD path.
- `skein_core::solver::solve_block_path_lla` — group LLA outer (group
  MCP / SCAD via weighted convex inner problem).
- `skein_core::solver::lla_solve` — scalar LLA without a path (single
  λ; the inner CD's reusable building block).
- `skein_core::solver::prox_newton_solve` — single-λ prox-Newton outer
  for GLMs (used by criterion microbenchmarks).
- `skein_core::solver::prox_newton_solve_path`,
  `prox_newton_block_solve_path` — GLM scalar / group path solvers
  with celer-style gap-safe screening on the weighted-LS surrogate.
- `skein_core::solver::prox_newton_fused_solve_path` — fused IRLS+CD
  GLM path (lazy per-iter weight update, mirrors `ncvreg::cdfit_glm`).
- `skein_core::solver::glasso_solve` — single-population graphical
  lasso (L1 / MCP / SCAD via penalty-factory dispatch).
- `skein_core::solver::joint_glasso_solve` — joint glasso ADMM across
  K populations (Danaher–Wang–Witten group form).

## Solver helpers

These are reachable from estimator code in `skein-py` and form part of
the documented surface:

- `skein_core::solver::{lambda_max, lambda_grid, block_lambda_max,
  cmcp_lambda_max}` — KKT-at-zero bounds and grid construction.
- `skein_core::solver::{surrogate_weights_bridge,
  surrogate_weights_group_mcp, surrogate_weights_group_scad,
  surrogate_weights_cmcp, surrogate_weights_gel,
  surrogate_sparse_group_scad, broadcast_group_weights_to_coord}` —
  LLA surrogate-weight closures for bridge, adaptive, group MCP/SCAD,
  cMCP, group exponential lasso, and sparse-group SCAD.
- `skein_core::solver::{scalar_convex_min_idx, group_convex_min_idx}`
  — post-fit `convex.min` detection for nonconvex paths (grpreg style).
- `skein_core::solver::{group_lipschitz, group_lipschitz_cache}` —
  per-group Lipschitz scan / cache for downstream block-CD callers.
- `skein_core::solver::{block_cd_solve_subset,
  block_cd_solve_subset_parallel}` — serial Gauss–Seidel and Jacobi
  block-CD inner solvers, exercised by the criterion bench tree.

## Configs and reports

Function signatures: every solver entry point above takes a `*Config`
and returns a `*Report` from this list. Adding fields is allowed under
semver if they default sensibly; removing or renaming is breaking.

- `skein_core::solver::{CdConfig, CdReport}`
- `skein_core::solver::{PathConfig, PathReport, Screening}`
- `skein_core::solver::{BlockPathConfig, BlockPathReport}`
- `skein_core::solver::{PathLLAReport, BlockPathLLAReport, LLAReport}`
- `skein_core::solver::{ProxNewtonReport, ProxNewtonPathReport,
  ProxNewtonBlockPathReport}`
- `skein_core::solver::{GlassoConfig, GlassoReport}`
- `skein_core::solver::{JointGlassoConfig, JointGlassoReport}`

## Prox primitives

Scalar and block prox operators — the mathematical building blocks
that custom `Penalty` / `GroupPenalty` impls reuse:

- `skein_core::prox::{soft_threshold, mcp_prox, scad_prox,
  elastic_net_prox}`.
- `skein_core::prox::{group_soft_threshold, group_elastic_net_prox}`.
- `skein_core::prox::{jacobi_eigh, logdet_eigen_prox}` — symmetric
  eigendecomposition + the `(M, ρ, c) ↦ argmin_Θ tr(MΘ) − c log det Θ
  + (ρ/2)‖Θ − Θ_prev‖²` prox used by the graphical-lasso ADMM.

The formulas are mathematical objects, so we treat them as frozen:
their signatures won't change. Function names won't either; if new
prox shapes are needed, they go in as additions.

## Standardize helpers

- `skein_core::standardize::{standardize, destandardize,
  destandardize_path, rescale_weights_for_standardize,
  StandardizeConfig, StandardizationStats}`.

Solvers internally operate in standardized space; results
destandardize on the way out. The bijection
`rescale_weights_for_standardize` maps per-feature penalty weights
across that boundary.

## Numerical guards

- `skein_core::numerics::W_FLOOR` (`1e-4`): the minimum allowed value
  of the IRLS weight `w_i = ∂²ℓ/∂η²` before clamping. Centralised so
  every GLM family shares the same guard.
- `skein_core::numerics::ETA_CLAMP` (`30.0`): the maximum allowed
  absolute value of the GLM linear predictor `η_i = (Xβ)_i` before
  clamping, applied before computing `μ = inv_link(η)` to avoid
  overflow in `exp` / `log1p_exp`.

## Errors

- `skein_core::SkeinError`, `skein_core::Result<T>`.

## Promised invariants

Beyond "we won't remove the symbol", these properties of the API are
part of the contract:

1. **`Sync + Send` on every trait object.** The solver dispatches
   group-wise work across Rayon threads; `&dyn DesignMatrix` and
   `&dyn Datafit` must be safely shareable across threads. Don't
   build trait impls with interior mutability that lacks proper
   synchronization.
2. **Prox convention.** `Penalty::prox_coord(j, z, step)` solves
   `argmin_x { (1/(2·step))(x − z)² + p_j(x) }` where `p_j` is the
   per-coordinate penalty *with the per-feature weight w_j folded
   in*. `GroupPenalty::prox_group` follows the same convention at
   the block level. We won't change this.
3. **Residual convention.** Datafits expose
   `r = X β − y` (note the sign — it's the *negative* residual in
   some conventions). Coordinate updates patch `r` in place after
   each `β_j` update. Solver code relies on this.
4. **`col_sq_norms` must be O(1).** The CD hot path calls
   `col_sq_norm(j)` once per coordinate update; if your backend
   computes it lazily on each call, you've added a factor of `n` to
   every fit.

## Items demoted to `pub(crate)` in the v1.0 freeze

For reference, the following items were `pub` in 0.x but moved to
`pub(crate)` when this contract was drawn — they are internal solver
plumbing that the path-level entry points wrap, and no downstream
user reported depending on them. If you were relying on one, file an
issue and we'll consider re-promoting it.

- `solver::cd_solve_warm`, `cd_solve_warm_with_residual`,
  `cd_solve_subset`, `cd_solve_subset_weighted_ls`,
  `cd_solve_subset_weighted_ls_with_lips` — internal CD variants used
  by the path solvers.
- `solver::block_cd_solve` — internal block-CD wrapper.
- `solver::prox_newton_fused_solve` — single-λ fused-solver primitive;
  the path variant `prox_newton_fused_solve_path` remains stable.
- `solver::lla::{cmcp_value, gel_value, surrogate_sparse_group_mcp}`
  — penalty value-eval helpers (kept for test fixtures).
- `solver::convex_region::PenaltyConcavity` — internal enum; the
  public detection functions `scalar_convex_min_idx` /
  `group_convex_min_idx` take an `f64` concavity directly.
- `design::GramDesign`, `datafit::GramLeastSquares` — internal
  precomputed-Gram path, not exposed via the Python facade.

## How we maintain this

Every release walks `cargo doc -p skein-core --no-deps` and diffs
against this page. New top-level `pub` items that aren't listed here
either get added on intent or stay `pub(crate)`. We treat the diff as
part of the release-prep checklist documented in `CLAUDE.md`.

## See also

- [Extending: backends](backend.md) — implementing `DesignMatrix`.
- [Extending: penalties](penalty.md) — implementing `Penalty` /
  `GroupPenalty`.
- [Extending: datafits](datafit.md) — implementing `Datafit`.
