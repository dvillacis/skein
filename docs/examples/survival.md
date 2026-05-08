# Worked example: survival — Cox PH with sparse-group MCP

Survival analysis with high-dimensional covariates: predict
time-to-event from a feature matrix where features cluster into
biologically meaningful groups (gene pathways, treatment regimens,
etc.) and we want both pathway-level and feature-level sparsity.

This is the same shape as the [genomics example](genomics.md), but
with a Cox PH datafit instead of LS. Cox has no intercept (the
baseline hazard absorbs it), and the fit signature takes
`(X, time, event)` instead of `(X, y)`.

## Setup

```python
import numpy as np
import skein

rng = np.random.default_rng(7)

# 800 patients, 100 features grouped into 10 pathways of 10 features.
n = 800
n_pathways = 10
features_per_pathway = 10
p = n_pathways * features_per_pathway
groups = np.repeat(np.arange(n_pathways), features_per_pathway)

# Standard-normal features (simulating standardized expression data,
# clinical covariates, etc.).
X = rng.standard_normal((n, p))

# Truth: 2 of the 10 pathways have any active features. Within an
# active pathway, 3 features are active.
true_beta = np.zeros(p)
active_pathways = [2, 6]
for path in active_pathways:
    feature_idx = np.where(groups == path)[0]
    active = rng.choice(feature_idx, size=3, replace=False)
    true_beta[active] = rng.normal(0.5, 0.2, size=3)

# Simulate survival times under exponential baseline hazard:
# T_i ~ Exp(λ_0 · exp(η_i)), independent right-censoring at C_i ~ Exp(0.3).
eta = X @ true_beta
hazard_rate = np.exp(eta)
T = rng.exponential(scale=1.0 / hazard_rate)
C = rng.exponential(scale=1.0 / 0.3, size=n)

time = np.minimum(T, C)
event = (T <= C).astype(float)

print(f"n = {n}, p = {p}, n_pathways = {n_pathways}")
print(f"event rate (uncensored): {event.mean():.2%}")
print(f"# truly active features: {int((true_beta != 0).sum())}")
print(f"truly active pathways: {active_pathways}")
```

The event rate (~70-80% with the hazards above) is on the higher
end of typical clinical data; if you're modeling something with
heavy censoring, expect the per-fit information to drop and λ_min
to need tightening.

## The fit

```python
group_sizes = np.bincount(groups)
group_weights = np.sqrt(group_sizes.astype(float))

# Cox + sparse-group MCP, CV-selected λ, standardize for safety.
fit = skein.CoxSparseGroupMCPPathCV(
    groups=groups,
    weights=group_weights,
    gamma=3.0,
    alpha=0.5,
    n_lambdas=40,
    lambda_min_ratio=1e-3,
    cv=10,
    standardize=True,
    random_state=0,
).fit(X, time, event)

print(f"Best λ:                     {fit.lambda_best_:.4f}")
print(f"# selected features:        {int((fit.coef_ != 0).sum())}")
print(f"selected pathways:          "
      f"{sorted(set(int(g) for g in groups[fit.coef_ != 0]))}")
print(f"truly active pathways:      {active_pathways}")
print(f"CV concordance at best λ:   {fit.cv_mean_scores_[fit.cv_mean_scores_.argmax()]:.3f}")
```

Note: Cox CV uses **Harrell's c-index** (concordance) by default,
which is higher-better. `fit.cv_mean_scores_.argmax()` picks the
best-by-c-index λ; for `lower-is-better` family the choice would
be `argmin`.

## Predicting risk

```python
# Build a small held-out set the same way.
X_new = rng.standard_normal((100, p))
risk_score = fit.predict(X_new)        # η = Xβ, the prognostic index

# Higher η → shorter survival. Rank-order patients:
order = np.argsort(risk_score)[::-1]   # descending risk
top10 = order[:10]
print(f"10 highest-risk patients (by η): {top10}")
print(f"η values:                        {risk_score[top10]}")
```

For survival predictions you typically want either:

- **The prognostic index** $\eta = X\beta$ — what `predict()` returns
  for Cox. Higher is shorter survival.
- **Survival probabilities** $S(t \mid x)$ — needs the baseline
  hazard estimate. v0.1 doesn't ship this; M3.7 has Breslow's
  cumulative-baseline-hazard estimator on the roadmap.

For now, the prognostic index is enough for ranking patients (the
"who's at highest risk" question), which is what most clinical
models use anyway.

## Path inspection

If you want to trace which features come on / drop off as λ
decreases (a common diagnostic in genomics):

```python
path = skein.CoxSparseGroupMCPPathRegressor(
    groups=groups,
    weights=group_weights,
    gamma=3.0,
    alpha=0.5,
    n_lambdas=40,
    lambda_min_ratio=1e-3,
    standardize=True,
).fit(X, time, event)

# path.coefs_ has shape (n_lambdas, p). For each λ, count active
# features per pathway.
n_active_by_lambda = (path.coefs_ != 0).sum(axis=1)
n_active_pathways_by_lambda = np.array([
    len(set(groups[path.coefs_[k] != 0]))
    for k in range(len(path.lambdas_))
])

import matplotlib.pyplot as plt    # if you have it
plt.plot(path.lambdas_, n_active_by_lambda, label="features")
plt.plot(path.lambdas_, n_active_pathways_by_lambda, label="pathways")
plt.xscale("log")
plt.xlabel("λ")
plt.ylabel("# selected")
plt.legend()
```

A typical pattern: the "pathways" curve grows in steps (one
pathway turns on at a time), then plateaus; the "features" curve
grows more smoothly within each pathway as within-group sparsity
relaxes.

## Comparing to a non-grouped fit

What happens if you drop the group structure and use scalar Cox MCP?

```python
fit_scalar = skein.CoxMCPPathCV(
    gamma=3.0,
    n_lambdas=40,
    lambda_min_ratio=1e-3,
    cv=10,
    standardize=True,
    random_state=0,
).fit(X, time, event)

selected_groups_scalar = sorted(set(
    int(g) for g in groups[fit_scalar.coef_ != 0]
))
print(f"Scalar Cox MCP selected pathways: {selected_groups_scalar}")
print(f"Sparse-group MCP selected:        "
      f"{sorted(set(int(g) for g in groups[fit.coef_ != 0]))}")
print(f"Truth:                            {active_pathways}")
```

Without the group structure, the scalar MCP often picks individual
features scattered across many pathways — even when, biologically,
features in the same pathway are correlated and either should all
be selected or none. The sparse-group MCP enforces the right
structure: every selected feature lives in an actively-selected
pathway.

## See also

- [Concepts: Datafits](../concepts/datafits.md) — Cox PH partial
  likelihood and Breslow ties.
- [Concepts: Penalties](../concepts/penalties.md) — sparse-group MCP
  shape.
- [Worked example: genomics](genomics.md) — the same group
  structure with a continuous outcome instead.
- [Porting from glmnet](../porting/glmnet.md) for `cv.glmnet(family =
  "cox")` users; [Porting from grpreg](../porting/grpreg.md) for
  `grpsurv` users.
