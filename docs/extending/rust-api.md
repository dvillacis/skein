# Stable Rust API contract

`skein-core` is the foundation crate that downstream Rust users
(per-paper crates, custom benchmarks, alternative Python bindings)
depend on. v0.1 is a **0.x release** — semver doesn't constrain
breaking changes — but the team treats certain pieces of the
public API as stable enough to plan against and reserves the right
to break others freely. This page documents the difference.

## Stable surface (we won't break without a deprecation cycle)

These are what downstream code should depend on:

### Trait surfaces

The four extension traits are the headline contract — implementing
them must continue to work.

- `skein_core::DesignMatrix` — five-method trait for storage backends
  (`matvec`, `rmatvec`, `col_dot`, `col_sq_norm`, `columns`). New
  required methods will not be added without a major version bump
  and a deprecation path.
- `skein_core::Datafit` — loss-function trait. Same stability
  guarantee.
- `skein_core::datafit::GlmDatafit` — the GLM extension trait
  (`surrogate_at`, `loss`).
- `skein_core::Penalty` — separable scalar-penalty trait
  (`prox_coord`, `value`, `weights`, `n_features`).
- `skein_core::GroupPenalty` — block-separable group-penalty trait
  (`prox_group`, `value`, `weights`).

### Concrete types

The concrete types used by every solver path:

- `skein_core::DenseMatrix`, `SparseCSC`
- `skein_core::Standardized<D>`, `Augmented<D>` — generic wrappers
  composable with any `D: DesignMatrix`.
- `skein_core::MmapMatrix`, `MmapMatrixF32`, `Chunked<C>` — out-of-
  RAM backends.
- `skein_core::groups::Groups` — group-partition representation.

### Algorithms (public entry points)

- `skein_core::solver::cd_solve`, `cd_solve_warm`, `solve_path`,
  `solve_block_path`, `solve_block_path_lla`,
  `prox_newton_solve_path`, `prox_newton_block_solve_path`.
- Their config + report structs (`CdConfig`, `CdReport`,
  `PathConfig`, `PathReport`, `BlockPathConfig`, `BlockPathReport`,
  `Screening`, `ProxNewtonReport`, `ProxNewtonPathReport`,
  `ProxNewtonBlockPathReport`).
- `skein_core::standardize::{standardize, destandardize,
  destandardize_path, rescale_weights_for_standardize,
  StandardizeConfig, StandardizationStats}`.

### Concrete penalties + datafits

The shipped concrete impls of the trait surfaces:

- Datafits: `LeastSquares`, `BinomialLogit`, `PoissonLog`, `CoxPH`.
- Penalties: `Mcp`, `Scad`, `GroupLasso`, `GroupMcp`,
  `SparseGroupLasso`. Their constructors (`new`, `with_weights`,
  `with_coord_weights`).

### Errors

- `skein_core::SkeinError`, `skein_core::Result<T>`.

## Incidental `pub` (subject to change)

These items are `pub` because the workspace's other crates
(`skein-py`) need them, but external users shouldn't depend on them
without expecting churn:

- Internal solver helpers (`block_cd_solve_subset`,
  `block_cd_solve_subset_with_cache`, `group_lipschitz_cache`,
  `block_lambda_max`, `lambda_max`, `lambda_grid`,
  `surrogate_weights_group_mcp`, `surrogate_weights_group_scad`,
  `surrogate_sparse_group_mcp`, `surrogate_sparse_group_scad`).
  These are the building blocks the path solvers use; their
  signatures may change as we improve the inner machinery.
- Module-level reports beyond the top-level public ones.
- The `prox` module (`prox::soft_threshold`, `prox::firm_threshold`,
  etc.) — useful primitives, but the exact set may grow / be
  reorganized.
- The `datafit::glm` submodule's working-response helpers if they
  exist as `pub` for testing purposes.

If you find yourself depending on something in this list, file an
issue — we'll consider promoting it to the stable surface, and in
the meantime you'll get a heads-up if we plan to change it.

## Promised invariants

Beyond "we won't remove the symbol", these properties of the API
are part of the contract:

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
   computes it lazily on each call, you've added a factor of `n`
   to every fit.

## Minor-version bump policy

While we're 0.x: any release may break incidental `pub` items
without notice, but the **stable surface** above will only break
in releases announced as breaking, with at least one minor release
of deprecation warnings preceding the removal where feasible.

When we move to 1.0, the stable surface freezes per semver. The
incidental `pub` surface either gets promoted (to the stable
surface) or hidden (`pub(crate)`).

## How we maintain this

The CI runs `cargo doc -p skein-core --no-deps` and we manually
audit the generated rustdoc against this page on each release. A
mechanical "doc-diff" tool that flags new public items not listed
here is on the M8 backlog.

## See also

- [Extending: backends](backend.md) — implementing `DesignMatrix`.
- [Extending: penalties](penalty.md) — implementing `Penalty` /
  `GroupPenalty`.
- [Extending: datafits](datafit.md) — implementing `Datafit`.
