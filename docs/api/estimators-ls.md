# LS estimators (Gaussian)

Least-squares datafit with the seven skein penalty families. All 14
classes follow the same shape:

- **Scalar penalty + single λ**: `MCPRegressor`, `SCADRegressor`,
  `ElasticNetRegressor`.
- **Scalar penalty + path**: `MCPPathRegressor`, `SCADPathRegressor`,
  `ElasticNetPathRegressor`.
- **Group penalty + single λ**: `GroupLassoRegressor`,
  `GroupMCPRegressor`, `GroupElasticNetRegressor`,
  `SparseGroupLassoRegressor`, `SparseGroupMCPRegressor`.
- **Group penalty + path**: their `*PathRegressor` siblings.

For multi-response Y (shape `(n, K)`) with joint feature selection,
see [Multi-task estimators](estimators-multitask.md).

Single-λ classes are useful when you've already chosen λ (via CV
externally, prior knowledge, etc.); path classes are the workhorse
for any analysis that picks λ post-hoc.

## Scalar — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.MCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.SCADRegressor
   :members:

.. autoclass:: skein_glm.estimators.ElasticNetRegressor
   :members:
```

## Scalar — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.MCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.SCADPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.ElasticNetPathRegressor
   :members:
```

## Group — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.GroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.GroupMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.GroupElasticNetRegressor
   :members:

.. autoclass:: skein_glm.estimators.SparseGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.SparseGroupMCPRegressor
   :members:
```

## Group — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.GroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.GroupMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.GroupElasticNetPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.SparseGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.SparseGroupMCPPathRegressor
   :members:
```
