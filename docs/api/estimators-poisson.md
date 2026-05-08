# Poisson estimators

Poisson regression with a log link: `y_i ~ Poisson(exp(η_i))`. Solved
via prox-Newton; same scaffolding as logistic.

Estimators expose `predict` (returns μ = exp(η), the conditional
mean — matches sklearn's `PoissonRegressor.predict`),
`decision_function` (returns η = log-rate). `y` must be ≥ 0.

## Scalar — single λ

::: skein.estimators.PoissonMCPRegressor
::: skein.estimators.PoissonSCADRegressor

## Scalar — path

::: skein.estimators.PoissonMCPPathRegressor
::: skein.estimators.PoissonSCADPathRegressor

## Group — single λ

::: skein.estimators.PoissonGroupLassoRegressor
::: skein.estimators.PoissonGroupMCPRegressor
::: skein.estimators.PoissonSparseGroupLassoRegressor
::: skein.estimators.PoissonSparseGroupMCPRegressor

## Group — path

::: skein.estimators.PoissonGroupLassoPathRegressor
::: skein.estimators.PoissonGroupMCPPathRegressor
::: skein.estimators.PoissonSparseGroupLassoPathRegressor
::: skein.estimators.PoissonSparseGroupMCPPathRegressor
