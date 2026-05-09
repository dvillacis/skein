# 2. Picking λ

The previous tutorial cheated: we picked `lambda_=0.05` out of a hat.
This page shows the three principled options for choosing λ —
**path** (compute many at once), **cross-validation**, and
**information criteria** — and when each is the right tool.

## Setup (same as tutorial 1)

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[[0, 4, 9, 23]] = [1.5, -1.0, 0.8, -1.2]
y = X @ true_beta + 0.3 * rng.standard_normal(n)
```

## Option 1 — The full path

If you want to *see* how the active set changes with λ instead of
collapsing to a single answer, fit a path.

```python
path = skein_glm.MCPPathRegressor(
    gamma=3.0, n_lambdas=50, lambda_min_ratio=1e-3,
).fit(X, y)

path.coefs_.shape       # (50, 50)  — coefs at each λ
path.intercepts_.shape  # (50,)
path.lambdas_.shape     # (50,)     — descending
```

Warm-starts thread β through decreasing λ, so a 50-point path costs
roughly 2-3× a single fit at the smallest λ. Use this when you want
to plot a sparsity-vs-shrinkage trace, or feed the whole path into a
downstream selector.

`predict(X)` on a path returns shape `(n_samples, n_lambdas)`.

## Option 2 — Cross-validation

The workhorse for unsupervised λ choice. K-fold; pick the λ that
minimizes mean test MSE.

```python
cv = skein_glm.MCPPathCV(
    gamma=3.0, cv=5, random_state=0,
    n_lambdas=50, lambda_min_ratio=1e-3,
).fit(X, y)

cv.lambda_best_         # the chosen λ
cv.coef_                # final-refit β at λ_best
cv.cv_mean_scores_      # (n_lambdas,) — mean test MSE per λ
cv.cv_std_scores_       # standard error
```

After fitting, `cv` behaves like a single-λ regressor: `cv.predict(X)`
uses the refit at `lambda_best_`. The full CV grid stays available
on `cv_scores_` (shape `(n_folds, n_lambdas)`) if you want a
one-standard-error rule.

CV is the right call when:

- You don't have a strong prior on the active-set size.
- Pure prediction quality matters more than parsimony.
- You can afford the K× cost of refitting per fold.

## Option 3 — Information criteria

Pick λ by AIC, BIC, or EBIC on a fitted path. No held-out data, no
folds — purely a function of the in-sample log-likelihood and an
active-set complexity penalty.

```python
best_idx, scores = skein_glm.select_by_ic(
    path, X, y, criterion="bic",
)
beta_best = path.coefs_[best_idx]
intercept_best = path.intercepts_[best_idx]
```

Three criteria, all minimized; in increasing order of how much they
prefer parsimony:

- **AIC** = `2k + 2·NLL` — softest; tends to keep more features.
- **BIC** = `log(n)·k + 2·NLL` — standard parsimony.
- **EBIC** = `BIC + 2γ·log C(p, k)` with `γ ∈ [0, 1]` — strict;
  recommended when `p ≫ n` (`ncvreg::BIC`'s default).

```python
best_idx, _ = skein_glm.select_by_ic(path, X, y, criterion="ebic", ebic_gamma=0.5)
```

ICs are the right call when:

- You have a single dataset and don't want to spend folds.
- You want a deterministic answer for a given path.
- `p` is large relative to `n` and you'd rather under-fit than
  over-fit.

## Comparing the three

```python
print(f"CV chose λ = {cv.lambda_best_:.4f}")
bic_idx, _ = skein_glm.select_by_ic(path, X, y, criterion="bic")
print(f"BIC chose λ = {path.lambdas_[bic_idx]:.4f}")
ebic_idx, _ = skein_glm.select_by_ic(path, X, y, criterion="ebic")
print(f"EBIC chose λ = {path.lambdas_[ebic_idx]:.4f}")
```

On this clean synthetic problem all three agree on a roughly similar
λ; the differences show up most when `n` is small or noise is heavy.

## Recap

| Tool | When |
|---|---|
| `MCPPathRegressor` | inspecting the full path, downstream pipelines |
| `MCPPathCV` | predictive λ choice with K-fold |
| `select_by_ic` | parsimonious λ choice without folds |

Each works the same way for `SCAD`, `ElasticNet`, `GroupMCP`, and
every other skein penalty — swap the class, keep the workflow.

## Next

→ [3. Logistic and Cox](03_logistic_and_cox.md) — same workflow, different
datafit.
