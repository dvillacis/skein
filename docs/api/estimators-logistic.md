# Logistic estimators (binomial)

Binomial logistic datafit (cross-entropy loss with sigmoid link).
Solved via prox-Newton outer iterations around the M1 LS coordinate
descent inner solver — see [Concepts: Datafits](../concepts/datafits.md)
for the algorithm.

All classes share `predict` (class labels {0, 1}), `predict_proba`
(P(y=1)), and `decision_function` (η = Xβ + α) inherited from the
logistic base class.

## Scalar — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.LogisticMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticSCADRegressor
   :members:
```

## Scalar — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.LogisticMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticSCADPathRegressor
   :members:
```

## Group — single λ

```{eval-rst}
.. autoclass:: skein_glm.estimators.LogisticGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticGroupMCPRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticSparseGroupLassoRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticSparseGroupMCPRegressor
   :members:
```

## Group — path

```{eval-rst}
.. autoclass:: skein_glm.estimators.LogisticGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticGroupMCPPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticSparseGroupLassoPathRegressor
   :members:

.. autoclass:: skein_glm.estimators.LogisticSparseGroupMCPPathRegressor
   :members:
```
