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

::: skein_glm.cv.MCPPathCV
::: skein_glm.cv.SCADPathCV
::: skein_glm.cv.ElasticNetPathCV
::: skein_glm.cv.GroupLassoPathCV
::: skein_glm.cv.GroupMCPPathCV
::: skein_glm.cv.SparseGroupLassoPathCV
::: skein_glm.cv.SparseGroupMCPPathCV

## Logistic family

::: skein_glm.cv.LogisticMCPPathCV
::: skein_glm.cv.LogisticSCADPathCV
::: skein_glm.cv.LogisticGroupLassoPathCV
::: skein_glm.cv.LogisticGroupMCPPathCV
::: skein_glm.cv.LogisticSparseGroupLassoPathCV
::: skein_glm.cv.LogisticSparseGroupMCPPathCV

## Poisson family

::: skein_glm.cv.PoissonMCPPathCV
::: skein_glm.cv.PoissonSCADPathCV
::: skein_glm.cv.PoissonGroupLassoPathCV
::: skein_glm.cv.PoissonGroupMCPPathCV
::: skein_glm.cv.PoissonSparseGroupLassoPathCV
::: skein_glm.cv.PoissonSparseGroupMCPPathCV

## Cox family

Cox CV uses `StratifiedKFold` by event indicator (so heavy
censoring doesn't produce event-empty train folds). Folds with zero
events are defensively skipped.

::: skein_glm.cv.CoxMCPPathCV
::: skein_glm.cv.CoxSCADPathCV
::: skein_glm.cv.CoxGroupLassoPathCV
::: skein_glm.cv.CoxGroupMCPPathCV
::: skein_glm.cv.CoxSparseGroupLassoPathCV
::: skein_glm.cv.CoxSparseGroupMCPPathCV
