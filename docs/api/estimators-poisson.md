# Poisson estimators

Poisson regression with a log link: `y_i ~ Poisson(exp(η_i))`. Solved
via prox-Newton; same scaffolding as logistic.

Estimators expose `predict` (returns μ = exp(η), the conditional
mean — matches sklearn's `PoissonRegressor.predict`),
`decision_function` (returns η = log-rate). `y` must be ≥ 0.

## Scalar — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.PoissonMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonSCADRegressor
   :members:
```

## Scalar — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.PoissonMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonSCADPathRegressor
   :members:
```

## Group — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.PoissonGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonGroupMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonSparseGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonSparseGroupMCPRegressor
   :members:
```

## Group — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.PoissonGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonGroupMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonSparseGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.PoissonSparseGroupMCPPathRegressor
   :members:
```
