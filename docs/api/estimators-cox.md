# Cox PH estimators

Cox proportional hazards with Breslow ties. Right-censored survival
data: fit signature is `fit(X, time, event)` instead of `fit(X, y)`.
**No intercept** — the baseline hazard absorbs it.

`predict(X)` returns the prognostic index η = Xβ (higher → shorter
survival); same as `decision_function(X)`. There's no `predict_proba`
on Cox — we don't ship the baseline-hazard estimator yet (M3.7
roadmap), so survival probabilities aren't directly available.

## Scalar — single λ

::: skein.estimators.CoxMCPRegressor
::: skein.estimators.CoxSCADRegressor

## Scalar — path

::: skein.estimators.CoxMCPPathRegressor
::: skein.estimators.CoxSCADPathRegressor

## Group — single λ

::: skein.estimators.CoxGroupLassoRegressor
::: skein.estimators.CoxGroupMCPRegressor
::: skein.estimators.CoxSparseGroupLassoRegressor
::: skein.estimators.CoxSparseGroupMCPRegressor

## Group — path

::: skein.estimators.CoxGroupLassoPathRegressor
::: skein.estimators.CoxGroupMCPPathRegressor
::: skein.estimators.CoxSparseGroupLassoPathRegressor
::: skein.estimators.CoxSparseGroupMCPPathRegressor
