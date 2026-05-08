# Logistic estimators (binomial)

Binomial logistic datafit (cross-entropy loss with sigmoid link).
Solved via prox-Newton outer iterations around the M1 LS coordinate
descent inner solver — see [Concepts: Datafits](../concepts/datafits.md)
for the algorithm.

All classes share `predict` (class labels {0, 1}), `predict_proba`
(P(y=1)), and `decision_function` (η = Xβ + α) inherited from the
logistic base class.

## Scalar — single λ

::: skein.estimators.LogisticMCPRegressor
::: skein.estimators.LogisticSCADRegressor

## Scalar — path

::: skein.estimators.LogisticMCPPathRegressor
::: skein.estimators.LogisticSCADPathRegressor

## Group — single λ

::: skein.estimators.LogisticGroupLassoRegressor
::: skein.estimators.LogisticGroupMCPRegressor
::: skein.estimators.LogisticSparseGroupLassoRegressor
::: skein.estimators.LogisticSparseGroupMCPRegressor

## Group — path

::: skein.estimators.LogisticGroupLassoPathRegressor
::: skein.estimators.LogisticGroupMCPPathRegressor
::: skein.estimators.LogisticSparseGroupLassoPathRegressor
::: skein.estimators.LogisticSparseGroupMCPPathRegressor
