# Cox PH estimators

Cox proportional hazards with Breslow ties. Right-censored survival
data: fit signature is `fit(X, time, event)` instead of `fit(X, y)`.
**No intercept** — the baseline hazard absorbs it.

`predict(X)` returns the prognostic index η = Xβ (higher → shorter
survival); same as `decision_function(X)`. There's no `predict_proba`
on Cox — we don't ship the baseline-hazard estimator yet (M3.7
roadmap), so survival probabilities aren't directly available.

## Scalar — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.CoxMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxSCADRegressor
   :members:
```

## Scalar — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.CoxMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxSCADPathRegressor
   :members:
```

## Group — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.CoxGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxGroupMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxSparseGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxSparseGroupMCPRegressor
   :members:
```

## Group — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.CoxGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxGroupMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxSparseGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.CoxSparseGroupMCPPathRegressor
   :members:
```
