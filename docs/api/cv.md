# Cross-validation

`*PathCV` estimators wrap a path solver in K-fold cross-validation.
Two flavors of selection:

- **Lower-is-better metrics** (MSE, binomial deviance, Poisson
  deviance): pick λ minimizing mean test score.
- **Higher-is-better metrics** (Cox concordance / c-index): pick λ
  maximizing.

The metric is auto-selected by family. After fitting, a final refit
on the full data at λ_best is stored on `coef_` / `intercept_`.

24 estimators total — six penalty types × four families. They all
share the same fit/predict surface as their non-CV counterparts.

## LS family

```{eval-rst}
.. autoclass:: skein_glm.cv.MCPPathCV
   :members:

.. autoclass:: skein_glm.cv.SCADPathCV
   :members:

.. autoclass:: skein_glm.cv.ElasticNetPathCV
   :members:

.. autoclass:: skein_glm.cv.GroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.GroupMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.SparseGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.SparseGroupMCPPathCV
   :members:
```

## Logistic family

```{eval-rst}
.. autoclass:: skein_glm.cv.LogisticMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.LogisticSCADPathCV
   :members:

.. autoclass:: skein_glm.cv.LogisticGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.LogisticGroupMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.LogisticSparseGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.LogisticSparseGroupMCPPathCV
   :members:
```

## Poisson family

```{eval-rst}
.. autoclass:: skein_glm.cv.PoissonMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.PoissonSCADPathCV
   :members:

.. autoclass:: skein_glm.cv.PoissonGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.PoissonGroupMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.PoissonSparseGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.PoissonSparseGroupMCPPathCV
   :members:
```

## Cox family

Cox CV uses `StratifiedKFold` by event indicator (so heavy
censoring doesn't produce event-empty train folds). Folds with zero
events are defensively skipped.

```{eval-rst}
.. autoclass:: skein_glm.cv.CoxMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.CoxSCADPathCV
   :members:

.. autoclass:: skein_glm.cv.CoxGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.CoxGroupMCPPathCV
   :members:

.. autoclass:: skein_glm.cv.CoxSparseGroupLassoPathCV
   :members:

.. autoclass:: skein_glm.cv.CoxSparseGroupMCPPathCV
   :members:
```
