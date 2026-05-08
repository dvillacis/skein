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

::: skein.cv.MCPPathCV
::: skein.cv.SCADPathCV
::: skein.cv.ElasticNetPathCV
::: skein.cv.GroupLassoPathCV
::: skein.cv.GroupMCPPathCV
::: skein.cv.SparseGroupLassoPathCV
::: skein.cv.SparseGroupMCPPathCV

## Logistic family

::: skein.cv.LogisticMCPPathCV
::: skein.cv.LogisticSCADPathCV
::: skein.cv.LogisticGroupLassoPathCV
::: skein.cv.LogisticGroupMCPPathCV
::: skein.cv.LogisticSparseGroupLassoPathCV
::: skein.cv.LogisticSparseGroupMCPPathCV

## Poisson family

::: skein.cv.PoissonMCPPathCV
::: skein.cv.PoissonSCADPathCV
::: skein.cv.PoissonGroupLassoPathCV
::: skein.cv.PoissonGroupMCPPathCV
::: skein.cv.PoissonSparseGroupLassoPathCV
::: skein.cv.PoissonSparseGroupMCPPathCV

## Cox family

Cox CV uses `StratifiedKFold` by event indicator (so heavy
censoring doesn't produce event-empty train folds). Folds with zero
events are defensively skipped.

::: skein.cv.CoxMCPPathCV
::: skein.cv.CoxSCADPathCV
::: skein.cv.CoxGroupLassoPathCV
::: skein.cv.CoxGroupMCPPathCV
::: skein.cv.CoxSparseGroupLassoPathCV
::: skein.cv.CoxSparseGroupMCPPathCV
