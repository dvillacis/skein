# 6. Counts and rates

Poisson regression handles non-negative integer outcomes — clicks,
events, infections, defects. This tutorial covers the standard count
case and then the much more useful **rate model** with exposure
offsets.

## Plain Poisson regression

For counts where every observation has the same denominator (or you
don't care to model exposure):

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
n, p = 300, 20
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[[0, 2, 5]] = [0.5, -0.4, 0.3]

eta = X @ true_beta
y = rng.poisson(np.exp(np.clip(eta, -3, 3))).astype(np.float64)

m = skein_glm.PoissonMCPPathRegressor(
    gamma=3.0, n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, y)

m.predict(X).shape         # (n, n_lambdas) — predicted means μ = exp(η)
m.decision_function(X).shape  # (n, n_lambdas) — linear predictor η
```

`predict` returns the **conditional mean** μ = exp(η), matching
sklearn's `PoissonRegressor`. `decision_function` returns η directly.

## Rate models with offsets

In practice, observations rarely have the same exposure. Counts of
hospital admissions are higher in states with more people; click
counts depend on impression volume; defect counts depend on
production-run length. The fix is a **log-exposure offset**:

`E[y_i] = exposure_i · exp(X_i β)` ⇔ `log E[y_i] = log(exposure_i) + X_i β`

The offset enters η as a known per-sample term that doesn't get
penalized.

```python
# Synthetic problem: each observation has a different exposure.
exposure = rng.uniform(0.5, 5.0, n)
true_rate = np.exp(np.clip(eta, -3, 3))   # rate per unit exposure
y = rng.poisson(exposure * true_rate).astype(np.float64)

offset = np.log(exposure)

m = skein_glm.PoissonMCPPathRegressor(
    gamma=3.0, offset=offset,
    n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, y)
```

The coefficients now have **rate-ratio** interpretation: a one-unit
increase in feature `j` multiplies the rate by `exp(coef_[j])`,
*holding exposure constant*. Without the offset, β would absorb
random exposure variation and you'd get biased estimates.

```python
# At the smallest λ, β should approximate true_beta.
last = m.coefs_[-1]
print("estimated β:", last[[0, 2, 5]].round(3))
print("true β:     ", true_beta[[0, 2, 5]])
```

### Predicting expected counts

When you `predict` from a model fit with offset, you're predicting
`exp(X β)` — i.e., the **rate per unit exposure**, not absolute
counts. To get expected counts, multiply by exposure:

```python
predicted_rate = m.predict(X)              # exp(X β); shape (n, n_lambdas)
predicted_count = exposure[:, None] * predicted_rate
```

## When to use offsets

Anywhere the natural quantity is a rate, not a raw count:

| Domain | Outcome | Offset |
|---|---|---|
| Epidemiology | new cases per region | log(person-years) |
| Click-through | clicks per ad slot | log(impressions) |
| Insurance | claims per policy | log(policy-years) |
| Manufacturing | defects per run | log(units produced) |
| Astronomy | photons per source | log(exposure time) |

Without the offset, your "feature effect" is confounded with exposure
variation. A coefficient that should have rate-ratio meaning becomes
uninterpretable.

## Offsets work everywhere in the Poisson family

Every Poisson estimator accepts `offset` — single-λ, path, CV,
scalar penalty, group penalty, sparse-group:

```python
# CV with offset
cv = skein_glm.PoissonMCPPathCV(
    gamma=3.0, offset=offset, cv=5, random_state=0,
    n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X, y)
cv.lambda_best_

# Group penalty with offset
groups = np.repeat(np.arange(4), 5).astype(np.int64)
m_group = skein_glm.PoissonGroupLassoPathRegressor(
    groups=groups, offset=offset,
    n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X, y)
```

CV slices `offset[train_idx]` per fold automatically — you don't
have to think about per-fold subsetting.

## A note on the validation rule

Poisson requires `y >= 0`. skein checks this on every fit and raises
a clean error if you pass negative values. Floating-point counts
(from rounding) work fine; pass `y.astype(np.float64)` if your raw
counts are integers.

## Recap

| Concept | API |
|---|---|
| Plain count regression | `PoissonMCPPathRegressor(...).fit(X, y)` |
| Rate model | add `offset=np.log(exposure)` |
| CV with offset | works automatically; folds are sliced |
| Group penalty + offset | every group/sparse-group Poisson estimator accepts `offset` |
| `predict` returns | rate per unit exposure (`exp(X β)`) — multiply by exposure to get expected counts |
| Coefficient interpretation | rate ratio: `exp(coef_[j])` is the rate multiplier |

## Next

→ [7. Stability selection](07_stability_selection.md) — bootstrap-based
feature selection without picking λ.
