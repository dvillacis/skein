# 8. Adaptive estimators

Plain lasso and MCP have a known weakness: they **shrink active
coefficients** toward zero. Even when feature `j` is unambiguously in
the model, the L1 penalty pulls `β_j` away from its true value. MCP
fixes this asymptotically (no shrinkage once `|β_j| > γλ`), but for
moderate `n` the bias persists.

**Adaptive estimators** (Zou 2006) attack this directly. The recipe
is two stages:

1. **Pilot fit.** Run a plain lasso on `(X, y)` and read off rough
   coefficient magnitudes.
2. **Adaptive refit.** Run the final lasso/MCP/SCAD again, but this
   time with **per-feature weights** `w_j = 1 / max(|β̂_pilot_j|, ε)^η`.
   Features with large pilot magnitudes get small penalty weights
   (less shrinkage); features with `β̂_pilot_j ≈ 0` get huge weights
   (and stay zero).

Under regularity conditions (Zou's Theorem 1), this two-stage
procedure achieves the **oracle property**: the active features are
selected with probability → 1, and their estimates are
asymptotically as good as if you'd known the active set ahead of
time.

## Walk-through

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))

true_beta = np.zeros(p)
true_beta[[0, 5, 12]] = [2.0, -1.5, 1.2]
y = X @ true_beta + 0.3 * rng.standard_normal(n)
```

Compare plain MCP to adaptive MCP:

```python
plain = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3,
).fit(X, y)
adaptive = skein_glm.AdaptiveMCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-3,
).fit(X, y)

last_plain = plain.coefs_[-1, [0, 5, 12]]
last_adapt = adaptive.coefs_[-1, [0, 5, 12]]
print("plain:    ", last_plain.round(3))
print("adaptive: ", last_adapt.round(3))
print("true:     ", true_beta[[0, 5, 12]])
```

The adaptive estimator should be visibly closer to the true values
on the active features — it's *unbiased* on the active set.

## Inspecting the pilot

Adaptive estimators expose two extra attributes:

```python
adaptive.coef_pilot_   # β at pilot_position along the pilot path
adaptive.weights_      # per-feature weights w_j fed to the final fit
```

`weights_[j]` is huge for features with `coef_pilot_[j] ≈ 0` and small
for active features. That's how the inactive features stay zero in
the final fit: their effective penalty is `λ · w_j ≈ ∞`.

## Tuning the recipe

Three knobs:

```python
m = skein_glm.AdaptiveMCPPathRegressor(
    gamma=3.0,
    eta=1.0,                # weight exponent — higher = more aggressive
    eps_pilot=1e-6,         # floor on |β_pilot| (avoid divide-by-zero)
    n_pilot_lambdas=10,     # length of the pilot path
    pilot_position="mid",   # which λ on the pilot path to read β from
    n_lambdas=20,
)
```

- `eta` — Zou's paper suggests `eta = 1` for `n` moderate, `eta = 2`
  for very large `n`. Higher values sparsify more aggressively.
- `pilot_position` — `'mid'` is a reasonable bias/variance compromise;
  `'last'` (smallest λ) is closest to OLS but unstable when `n < p`;
  an integer index gives full control.

## When adaptive helps — and when it doesn't

Adaptive shines when:

- The truth is **sparse** with **strong signals** on the active set.
- You have enough `n` for a halfway-decent pilot fit (`n` larger
  than the active-set size, ideally `n` much larger than `p`).
- Bias on the active set is the cost you're trying to reduce.

Adaptive can hurt when:

- `n < p` and the pilot can't recover *anything* — the weights are
  noise.
- The truth has many weak signals (the pilot misses them, and they
  get penalized out of existence by the adaptive weights).
- You care more about prediction than about coefficient accuracy.

## CV with adaptive

```python
cv = skein_glm.AdaptiveMCPPathCV(
    gamma=3.0, eta=1.0, cv=5, random_state=0,
    n_lambdas=20, lambda_min_ratio=1e-3,
).fit(X, y)
cv.lambda_best_       # CV-selected λ for the final fit
cv.coef_pilot_        # pilot β at full data (not per-fold)
cv.weights_           # adaptive weights derived from full-data pilot
```

The pilot runs **once on the full data**, not per-fold. The pilot
weights are a hyperparameter (a data-derived input to the final
fit), not a model parameter — refitting them per fold would be a
different procedure.

## Adaptive group / GLM variants

Same recipe applies at the group level and for GLMs. 30 adaptive
classes total; a few highlights:

```python
# Group-adaptive: per-group inverse-norm weights.
groups = np.repeat(np.arange(6), 5).astype(np.int64)
m = skein_glm.AdaptiveGroupMCPPathRegressor(
    groups=groups, gamma=3.0, eta=1.0,
    n_lambdas=20, lambda_min_ratio=1e-3,
).fit(X, y)
m.weights_     # length-6 (per group)

# Adaptive logistic.
y_bin = (y > 0).astype(np.float64)
m = skein_glm.AdaptiveLogisticMCPPathRegressor(
    gamma=3.0, eta=1.0, n_lambdas=20,
).fit(X, y_bin)

# Adaptive Cox.
time = rng.exponential(1.0, n)
event = (rng.uniform(size=n) < 0.7).astype(np.float64)
m = skein_glm.AdaptiveCoxMCPPathRegressor(
    gamma=3.0, eta=1.0, n_lambdas=20,
).fit(X, time, event)
```

Every penalty (Lasso, MCP, SCAD) and every GLM datafit (LS, logistic,
Poisson, Cox, plus group variants) has a matching adaptive class.

## Recap

| Goal | Tool |
|---|---|
| Reduce shrinkage on active features | `Adaptive*PathRegressor` |
| Inspect the pilot fit | `coef_pilot_`, `weights_` |
| Group-level adaptive selection | `AdaptiveGroup{Lasso,MCP,SCAD}PathRegressor` |
| GLM adaptive | `Adaptive{Logistic,Poisson,Cox}*PathRegressor` |

Adaptive is the **headline use** of skein's per-feature-weights axis.
The two-stage recipe is pure Python composition — every adaptive
class is a wrapper around an existing path estimator that just
plumbs `weights=` correctly.

## Next

→ [9. Multinomial and multi-task](09_multinomial_and_multitask.md) —
K-class softmax and multi-response regression.
