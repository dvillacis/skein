# 1. Your first fit

The smallest interesting thing skein can do: fit a sparse linear
regression and see which features are active.

## What we're building

A linear model `y = X β + ε` where most components of `β` are zero.
The job is to recover the support — the set of features with nonzero
coefficients — and estimate their values. We'll use `MCPRegressor`
because the **MCP penalty** is unbiased on truly active coefficients,
unlike plain lasso.

## A toy problem

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)

# 200 samples, 50 features. Only features 0, 4, 9, 23 actually matter.
n, p = 200, 50
X = rng.standard_normal((n, p))

true_beta = np.zeros(p)
true_beta[[0, 4, 9, 23]] = [1.5, -1.0, 0.8, -1.2]

y = X @ true_beta + 0.3 * rng.standard_normal(n)
```

Four features carry signal; 46 are pure noise. A regular OLS fit
would give noisy estimates for every coefficient — useless if you
want to know which features actually matter.

## Fit the model

```python
model = skein_glm.MCPRegressor(lambda_=0.05, gamma=3.0).fit(X, y)
```

That's it. `lambda_` is the penalty strength; `gamma` controls how
sharply MCP transitions from "shrink the estimate" to "leave it
alone." The default `gamma=3.0` is what `ncvreg` recommends.

## What you get back

```python
model.coef_                # ndarray (p,)
model.intercept_           # float
model.n_features_in_       # 50

active = np.flatnonzero(np.abs(model.coef_) > 1e-6)
print(active)              # [ 0  4  9 23]
print(model.coef_[active]) # ~[1.5, -1.0, 0.8, -1.2]
```

The exact-zero pattern outside the active set is the headline of
sparse regression — non-zero entries identify the variables that
matter; zeros let you drop the rest.

## Predict and score

```python
y_pred = model.predict(X)
r2 = model.score(X, y)
print(f"R² = {r2:.3f}")
```

`predict` returns `Xβ + α`. `score` returns the sklearn-default
R² (since `MCPRegressor` inherits from `RegressorMixin`).

## What's the catch?

Picking `lambda_=0.05` was a guess. Try `lambda_=0.5` and watch every
coefficient collapse to zero (over-regularized); try `lambda_=0.001`
and the noise features creep back in. The next tutorial covers the
**three principled ways to pick λ**: the full λ-path (and warm
starts), cross-validation, and information criteria.

## Recap

| What | API |
|---|---|
| Fit a sparse linear model | `skein_glm.MCPRegressor(lambda_, gamma).fit(X, y)` |
| Read coefficients | `model.coef_`, `model.intercept_` |
| Predictions | `model.predict(X)` |
| R² score | `model.score(X, y)` |

Same `Regressor / fit / predict / score` shape as scikit-learn — every
skein estimator follows it.

## Next

→ [2. Picking λ](02_picking_lambda.md) — paths, CV, and AIC/BIC/EBIC.
