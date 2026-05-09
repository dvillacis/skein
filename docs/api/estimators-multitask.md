# Multi-task LS estimators

Multi-response least squares with **joint feature selection across
tasks**. Unlike the scalar LS family — where every output column of
`Y` would be regressed independently — these estimators penalize a
feature's whole row of the coefficient matrix `B ∈ ℝ^(p × K)`, so
that feature `j` is either "active in every task" or "inactive in
every task." Useful for genomics, finance, and any setting where
related outcomes are expected to share a sparse support.

All 8 classes follow the same shape:

- **Lasso (convex)**: `MultiTaskLassoRegressor`, `MultiTaskLassoPathRegressor`.
- **MCP (LLA-wrapped non-convex)**: `MultiTaskMCPRegressor`, `MultiTaskMCPPathRegressor`.
- **SCAD (LLA-wrapped non-convex)**: `MultiTaskSCADRegressor`, `MultiTaskSCADPathRegressor`.
- **Elastic net (convex)**: `MultiTaskElasticNetRegressor`, `MultiTaskElasticNetPathRegressor`.

Single-λ classes are useful when you've already chosen λ; path
classes are the workhorse when picking λ post-hoc. CV wrappers live
on the [CV](cv.md) page.

## Inputs and outputs

`fit(X, Y)` takes `X ∈ ℝ^(n, p)` (dense ndarray or scipy.sparse CSC)
and `Y ∈ ℝ^(n, K)`. After fitting:

- `coef_` has shape `(K, p)` — matches sklearn's `MultiTaskLasso`.
- `intercept_` has shape `(K,)`.
- `predict(X) → (n_pred, K)`.

Path classes have analogous `coefs_` of shape `(n_lambdas, K, p)`
and `intercepts_` of shape `(n_lambdas, K)`.

## Convention vs sklearn

skein uses the natural per-sample objective from the stacked
formulation: `(1/(2nK)) ‖Y - X·B‖²_F + λ · P(B)` where `P(B) = Σ_j
w_j ‖B[j, :]‖_2` for the lasso. sklearn's `MultiTaskLasso` uses
`(1/(2n)) ‖Y - X·W^T‖²_F + α · ‖W‖_{2,1}`. The same minimizer is
reached at `λ_skein = α_sklearn / K`. Verified by a regression test
in the suite.

This convention difference does **not** apply to MCP/SCAD's
shape parameters (`gamma`, `a`) — those mean the same thing as
in scalar MCP/SCAD because they're applied to the per-row L2 norm
of `B[j, :]`, which is on the same scale as the natural
formulation.

See [Concepts → Multi-task](../concepts/multitask.md) for the
algebraic reduction (multi-task LS reduces *exactly* to a
group-lasso problem on a virtual block-replicated design) and
the K-rescale derivation.

## Sparse + standardize

The `MultiTaskLasso/MCP/SCAD/ElasticNet` estimators dispatch
transparently on `scipy.sparse.issparse(X)` — pass a CSC matrix
and the path solver routes through the sparse `MultiTaskDesign`
backend. `standardize=True` works for both dense (physical
center+scale) and sparse (lazy `Standardized` wrapper composed
with the augmented intercept column).

## Lasso — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multitask.MultiTaskLassoRegressor
   :members:

.. autoclass:: skein_glm.multitask.MultiTaskLassoPathRegressor
   :members:
```

## MCP — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multitask.MultiTaskMCPRegressor
   :members:

.. autoclass:: skein_glm.multitask.MultiTaskMCPPathRegressor
   :members:
```

## SCAD — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multitask.MultiTaskSCADRegressor
   :members:

.. autoclass:: skein_glm.multitask.MultiTaskSCADPathRegressor
   :members:
```

## Elastic net — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multitask.MultiTaskElasticNetRegressor
   :members:

.. autoclass:: skein_glm.multitask.MultiTaskElasticNetPathRegressor
   :members:
```
