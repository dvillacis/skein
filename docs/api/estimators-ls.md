# LS estimators (Gaussian)

Least-squares datafit with the six skein penalty families. All 12
classes follow the same shape:

- **Scalar penalty + single λ**: `MCPRegressor`, `SCADRegressor`.
- **Scalar penalty + path**: `MCPPathRegressor`, `SCADPathRegressor`.
- **Group penalty + single λ**: `GroupLassoRegressor`,
  `GroupMCPRegressor`, `SparseGroupLassoRegressor`,
  `SparseGroupMCPRegressor`.
- **Group penalty + path**: their `*PathRegressor` siblings.

Single-λ classes are useful when you've already chosen λ (via CV
externally, prior knowledge, etc.); path classes are the workhorse
for any analysis that picks λ post-hoc.

## Scalar — single λ

::: skein.estimators.MCPRegressor
::: skein.estimators.SCADRegressor

## Scalar — path

::: skein.estimators.MCPPathRegressor
::: skein.estimators.SCADPathRegressor

## Group — single λ

::: skein.estimators.GroupLassoRegressor
::: skein.estimators.GroupMCPRegressor
::: skein.estimators.SparseGroupLassoRegressor
::: skein.estimators.SparseGroupMCPRegressor

## Group — path

::: skein.estimators.GroupLassoPathRegressor
::: skein.estimators.GroupMCPPathRegressor
::: skein.estimators.SparseGroupLassoPathRegressor
::: skein.estimators.SparseGroupMCPPathRegressor
