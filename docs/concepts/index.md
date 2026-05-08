# Concepts

`skein` is built around a small set of orthogonal abstractions. Once
you understand the four pieces below, you can predict the behavior
of any combination — including ones we haven't documented as
worked examples — because the solver code never sees the difference.

## The four axes

| Axis            | Trait surface             | Concrete types in v0.1                                        | Page                       |
|-----------------|---------------------------|---------------------------------------------------------------|----------------------------|
| **Penalty**     | `Penalty`, `GroupPenalty` | MCP, SCAD, group lasso, group MCP, sparse-group lasso/MCP     | [Penalties](penalties.md)  |
| **Datafit**     | `Datafit`, `GlmDatafit`   | Least squares, binomial logistic, Poisson, Cox PH (Breslow)   | [Datafits](datafits.md)    |
| **Weights**     | (per-axis on each trait)  | per-sample, per-feature, per-group                            | [Weights](weights.md)      |
| **Backend**     | `DesignMatrix`            | dense, sparse CSC, mmap (f64 + f32), chunked, augmented, standardized | [Backends](backends.md) |

Every estimator class in `skein.*` is a packaging of one
`(datafit, penalty)` pair with optional weights, behind a single
sklearn-compatible `fit` / `predict` interface. The path variants
add warm-starting across a λ-grid; the CV variants wrap a path
in K-fold cross-validation. You don't need to learn 48 separate
classes — they're all the same machinery with different
`(datafit, penalty)` instantiations.

## Why this matters

Two things follow from the abstraction:

1. **New backends are O(1) effort.** The `DesignMatrix` trait has
   five methods (`matvec`, `rmatvec`, `col_dot`, `col_sq_norm`,
   `columns`). Implementing them for a new backend (an HDF5 reader,
   a Parquet column store, a GPU buffer) takes a few hundred lines
   and works with every estimator immediately. The `Augmented` and
   `Standardized` wrappers compose with anything that implements
   the trait.

2. **New penalties are also O(1) effort.** The `Penalty` trait
   exposes `prox`, `value`, and `weights`; the `GroupPenalty` trait
   adds block-prox primitives. Once you implement these, every
   datafit and every backend already supports your penalty — you
   don't write a separate `MyPenaltyOnSparse` or `MyPenaltyForCox`.

The downside of this orthogonality: when something goes wrong,
the bug is in exactly one of the four traits, and the type system
won't help you locate it. The test suite is structured the same
way (199 cargo tests + 138 pytests) so each axis is covered
independently.

## Reading order

If you're new to the conceptual model:

1. **[Datafits](datafits.md)** first — what "the loss function"
   means in `skein` and how prox-Newton turns non-Gaussian losses
   into a sequence of weighted least-squares problems.
2. **[Penalties](penalties.md)** second — convex (lasso, group
   lasso) vs. nonconvex (MCP, SCAD), and how Local Linear
   Approximation reduces nonconvex to weighted convex.
3. **[Weights](weights.md)** third — the three independent axes
   (per-sample, per-feature, per-group), what each one accomplishes
   statistically, and how to combine them.
4. **[Backends](backends.md)** last — the storage + wrapper hierarchy
   (`DenseMatrix`, `SparseCSC`, `MmapMatrix`, `Chunked<C>`, plus the
   `Augmented<D>` and `Standardized<D>` wrappers).

If you're porting from R, skip ahead to the **Porting** section
(coming in commit 2) — it's organized by R package and points back
to the relevant concept pages.
