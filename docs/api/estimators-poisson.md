# Poisson estimators

Poisson regression with a log link: `y_i ~ Poisson(exp(η_i))`. Solved
via prox-Newton; same scaffolding as logistic.

Estimators expose `predict` (returns μ = exp(η), the conditional
mean — matches sklearn's `PoissonRegressor.predict`),
`decision_function` (returns η = log-rate). `y` must be ≥ 0.

## Scalar — single λ

::: skein_glm.estimators.PoissonMCPRegressor
::: skein_glm.estimators.PoissonSCADRegressor

## Scalar — path

::: skein_glm.estimators.PoissonMCPPathRegressor
::: skein_glm.estimators.PoissonSCADPathRegressor

## Group — single λ

::: skein_glm.estimators.PoissonGroupLassoRegressor
::: skein_glm.estimators.PoissonGroupMCPRegressor
::: skein_glm.estimators.PoissonSparseGroupLassoRegressor
::: skein_glm.estimators.PoissonSparseGroupMCPRegressor

## Group — path

::: skein_glm.estimators.PoissonGroupLassoPathRegressor
::: skein_glm.estimators.PoissonGroupMCPPathRegressor
::: skein_glm.estimators.PoissonSparseGroupLassoPathRegressor
::: skein_glm.estimators.PoissonSparseGroupMCPPathRegressor
