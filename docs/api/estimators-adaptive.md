# Adaptive estimators

Two-stage adaptive penalty estimators (Zou 2006 for adaptive lasso,
extended here to MCP and SCAD). The recipe:

1. **Pilot fit.** Run a plain lasso path (MCP at γ → ∞) on `(X, y)` and
   read β at a chosen position along the path (default: middle of the
   auto-generated λ-grid).
2. **Adaptive weights.** Compute per-feature
   `w_j = 1 / max(|β_pilot[j]|, ε)^η`. Larger pilot magnitudes mean
   smaller penalty weights — truly active features are shrunk less,
   inactive features (with `β_pilot ≈ 0`) get huge weights and stay at
   zero.
3. **Final fit.** Re-fit the chosen final penalty (Lasso / MCP / SCAD)
   with these adaptive weights. The path is the **final** estimator's
   path, λ-decreasing.

The motivating result is **the oracle property**: under regularity
conditions, adaptive lasso recovers the true sparse support and yields
asymptotically unbiased estimates on the active features — neither of
which plain lasso provides. Adaptive MCP / SCAD inherit this story but
typically need fewer outer iterations because the underlying penalty
already has the "near-unbiased on active features" property.

This family is the headline use of skein's per-feature `weights=`
parameter — the underlying solvers all accept it directly, so adaptive
estimators are pure composition with no Rust changes.

## Pilot strategy

The pilot is a `MCPPathRegressor(gamma=1e9)` fit (i.e. plain lasso) of
length `n_pilot_lambdas` (default 10). The β at `pilot_position`
(default `'mid'` — index `n_pilot_lambdas // 2`) is the pilot estimate.
Other positions:

- `'last'` — smallest λ (closest to OLS for `n > p`, but unstable for
  `n < p`).
- An integer index into the pilot path — full control.

The pilot runs on the **full** data even inside the CV variants — pilot
weights are a data-derived hyperparameter, not a model parameter, and
re-fitting the pilot per-fold would be a different procedure.

## Adaptive lasso — path + CV

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptiveLassoPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveLassoPathCV
   :members:
```

## Adaptive MCP — path + CV

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptiveMCPPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveMCPPathCV
   :members:
```

## Adaptive SCAD — path + CV

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptiveSCADPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveSCADPathCV
   :members:
```

## Adaptive group estimators

For group penalties, the adaptive weights are **per-group**: pilot is
plain group lasso, and the per-group L2 norm `‖β_pilot[g]‖_2` becomes
the weight `w_g = 1 / max(‖β_pilot[g]‖_2, ε)^η`. Active groups receive
small weights and are shrunk less; inactive groups get huge weights and
stay zero.

`GroupLasso` and `GroupMCP` are wired up; `GroupSCAD` is a separate
prerequisite (only the `SparseGroup` variant of SCAD ships today —
plain `GroupSCAD` is a small wiring task on its own).

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptiveGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveGroupMCPPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveGroupMCPPathCV
   :members:
```

## Adaptive GLMs (Logistic, Poisson, Cox)

Same recipe applied to GLM datafits. Pilot is the GLM's lasso path
(e.g., `LogisticMCPPathRegressor(gamma=1e9)`); final is the user's
chosen GLM-penalty path with adaptive weights. CV inherits the
per-family scoring from the existing CV mixins (binomial deviance for
logistic, Poisson deviance for Poisson, Harrell c-index for Cox), and
Cox keeps its `fit(x, time, event)` signature with StratifiedKFold by
event indicator.

### Adaptive logistic — Path + CV

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptiveLogisticLassoPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveLogisticLassoPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveLogisticMCPPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveLogisticMCPPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveLogisticSCADPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveLogisticSCADPathCV
   :members:
```

### Adaptive Poisson — Path + CV

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptivePoissonLassoPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptivePoissonLassoPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptivePoissonMCPPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptivePoissonMCPPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptivePoissonSCADPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptivePoissonSCADPathCV
   :members:
```

### Adaptive Cox PH — Path + CV

```{eval-rst}
.. autoclass:: skein_glm.adaptive.AdaptiveCoxLassoPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveCoxLassoPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveCoxMCPPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveCoxMCPPathCV
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveCoxSCADPathRegressor
   :members:

.. autoclass:: skein_glm.adaptive.AdaptiveCoxSCADPathCV
   :members:
```
