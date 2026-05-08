# Logistic estimators (binomial)

Binomial logistic datafit (cross-entropy loss with sigmoid link).
Solved via prox-Newton outer iterations around the M1 LS coordinate
descent inner solver — see [Concepts: Datafits](../concepts/datafits.md)
for the algorithm.

All classes share `predict` (class labels {0, 1}), `predict_proba`
(P(y=1)), and `decision_function` (η = Xβ + α) inherited from the
logistic base class.

## Scalar — single λ

::: skein_glm.estimators.LogisticMCPRegressor
::: skein_glm.estimators.LogisticSCADRegressor

## Scalar — path

::: skein_glm.estimators.LogisticMCPPathRegressor
::: skein_glm.estimators.LogisticSCADPathRegressor

## Group — single λ

::: skein_glm.estimators.LogisticGroupLassoRegressor
::: skein_glm.estimators.LogisticGroupMCPRegressor
::: skein_glm.estimators.LogisticSparseGroupLassoRegressor
::: skein_glm.estimators.LogisticSparseGroupMCPRegressor

## Group — path

::: skein_glm.estimators.LogisticGroupLassoPathRegressor
::: skein_glm.estimators.LogisticGroupMCPPathRegressor
::: skein_glm.estimators.LogisticSparseGroupLassoPathRegressor
::: skein_glm.estimators.LogisticSparseGroupMCPPathRegressor
