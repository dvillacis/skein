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

::: skein_glm.estimators.MCPRegressor
::: skein_glm.estimators.SCADRegressor
::: skein_glm.estimators.ElasticNetRegressor

## Scalar — path

::: skein_glm.estimators.MCPPathRegressor
::: skein_glm.estimators.SCADPathRegressor
::: skein_glm.estimators.ElasticNetPathRegressor

## Group — single λ

::: skein_glm.estimators.GroupLassoRegressor
::: skein_glm.estimators.GroupMCPRegressor
::: skein_glm.estimators.SparseGroupLassoRegressor
::: skein_glm.estimators.SparseGroupMCPRegressor

## Group — path

::: skein_glm.estimators.GroupLassoPathRegressor
::: skein_glm.estimators.GroupMCPPathRegressor
::: skein_glm.estimators.SparseGroupLassoPathRegressor
::: skein_glm.estimators.SparseGroupMCPPathRegressor
