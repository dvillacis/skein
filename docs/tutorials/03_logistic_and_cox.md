# 3. Logistic and Cox

The previous two tutorials used Gaussian data: `y = Xβ + ε`. This
page shows that swapping the **datafit** is the only change needed
to handle binary classification or right-censored survival outcomes.
The penalty stays the same; the workflow is identical.

skein is built around an `(datafit) × (penalty)` orthogonality. Once
you know how to fit `MCPRegressor` for least squares, you know how to
fit logistic-MCP, Poisson-MCP, Cox-MCP — they're the same machinery
with a different loss.

## Logistic regression

Binary `y ∈ {0, 1}`. Use `LogisticMCPPathRegressor`:

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
n, p = 300, 30
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[[0, 5, 12]] = [1.5, -1.0, 0.8]

# Bernoulli outcome via the logistic link.
eta = X @ true_beta
prob = 1.0 / (1.0 + np.exp(-eta))
y = (rng.uniform(size=n) < prob).astype(np.float64)

clf = skein_glm.LogisticMCPPathRegressor(
    gamma=3.0, n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, y)

clf.coefs_.shape          # (30, 30)
clf.predict_proba(X).shape  # (n, n_lambdas) — P(y=1) per λ
clf.predict(X).shape        # (n, n_lambdas) — class labels (0/1)
```

Note `predict_proba` returns probabilities and `predict` returns hard
class labels — same convention as sklearn's `LogisticRegression`. The
path version returns one column per λ; the single-λ
`LogisticMCPRegressor` returns a 1-D vector.

### CV scoring is binomial deviance

```python
clf_cv = skein_glm.LogisticMCPPathCV(
    gamma=3.0, cv=5, random_state=0, n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, y)
clf_cv.lambda_best_
clf_cv.predict(X[:5])     # ndarray of 0/1 labels
clf_cv.predict_proba(X[:5])  # P(y=1) for each row
```

CV automatically picks the right loss for the family — binomial
deviance for logistic, Poisson deviance for Poisson, Harrell c-index
(higher-is-better) for Cox.

## Cox proportional hazards

Right-censored survival data has a different shape: each observation
is `(time, event)` where `event ∈ {0, 1}` (1 = observed event,
0 = censored at `time`). The fit signature reflects that:

```python
# Synthetic survival problem.
n, p = 300, 20
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[[0, 3]] = [0.6, -0.5]

eta = X @ true_beta
true_time = rng.exponential(1.0 / np.exp(np.clip(eta, -3, 3)))
censor_time = rng.exponential(2.0, n)
time = np.minimum(true_time, censor_time)
event = (true_time <= censor_time).astype(np.float64)

cox = skein_glm.CoxMCPPathRegressor(
    gamma=3.0, n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, time, event)

cox.coefs_.shape           # (30, 20)
cox.predict(X).shape       # (n, n_lambdas) — Cox prognostic index η
```

Three things to know about Cox:

1. **No intercept.** The baseline hazard absorbs any constant shift
   in η, so there's no `intercept_`.
2. **`predict` returns η = Xβ**, not a probability or a survival
   curve. This is the **prognostic index** — higher values mean
   higher hazard. It matches `glmnet::predict.cox` and what you'd
   feed into Harrell's c-index.
3. **CV uses StratifiedKFold by event indicator.** Heavy censoring
   doesn't produce event-empty folds.

```python
cox_cv = skein_glm.CoxMCPPathCV(
    gamma=3.0, cv=5, random_state=0, n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, time, event)
cox_cv.cv_mean_scores_   # higher is better — Harrell c-index
cox_cv.lambda_best_
```

### Tie handling: Breslow vs Efron

Cox's partial likelihood needs a tie-handling convention. skein
defaults to **Breslow** (matches `glmnet`), but supports **Efron**
(R's `survival::coxph` default; more accurate when many event times
are tied):

```python
cox_efron = skein_glm.CoxMCPPathRegressor(
    gamma=3.0, ties="efron", n_lambdas=30,
).fit(X, time, event)
```

Reduces exactly to Breslow when no ties — a smaller default cost
than you might expect.

## Poisson

The third standard GLM. Non-negative integer `y`, with the canonical
log link. Optional **offset** for rate models:

```python
exposure = rng.uniform(0.5, 2.0, n)
y = rng.poisson(exposure * np.exp(np.clip(X @ true_beta, -3, 3))).astype(np.float64)

pois = skein_glm.PoissonMCPPathRegressor(
    gamma=3.0, offset=np.log(exposure),
    n_lambdas=30, lambda_min_ratio=1e-3,
).fit(X, y)
```

The offset is just `log(exposure)` — see [tutorial 6](06_counts_and_rates.md)
for the rate-model story.

## Recap

| Datafit | Single-λ | Path | CV | Predict semantics |
|---|---|---|---|---|
| Gaussian (LS) | `MCPRegressor` | `MCPPathRegressor` | `MCPPathCV` | `Xβ + α` |
| Logistic | `LogisticMCPRegressor` | `LogisticMCPPathRegressor` | `LogisticMCPPathCV` | `predict` = label, `predict_proba` = P(y=1) |
| Poisson | `PoissonMCPRegressor` | `PoissonMCPPathRegressor` | `PoissonMCPPathCV` | `predict` = μ = exp(η), `decision_function` = η |
| Cox PH | `CoxMCPRegressor` | `CoxMCPPathRegressor` | `CoxMCPPathCV` | `predict = decision_function` = η |

Same row applies to `SCAD`, `ElasticNet`, and the group penalties
covered in the next tutorial.

## Next

→ [4. Group penalties](04_group_penalties.md) — selecting whole groups of
features at once.
