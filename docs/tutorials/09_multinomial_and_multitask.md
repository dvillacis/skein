# 9. Multinomial and multi-task

So far every model has had a single output: one `y` per sample (LS,
logistic, Poisson, Cox). Two situations break that:

- **Multinomial classification.** Each `y` is one of K class labels.
- **Multi-task regression.** Each `y` is a K-vector of related
  continuous outcomes.

Both reduce algebraically to a **row-grouped problem on a virtual
block-replicated design** — the same M2 block-CD machinery used
elsewhere in skein, just with a different design wrapper. Concretely:
the coefficient matrix `B ∈ ℝ^(p×K)` gets penalized at the **row
level** (the row of feature `j`'s K coefficients), so a feature is
either active for every class/task or inactive for all of them.

## Multinomial: K-class softmax

Build a 3-class problem:

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
n, p, K = 300, 30, 3
X = rng.standard_normal((n, p))

# Coefficient matrix B (p × K). Only features 0, 5, 12 are active.
true_B = np.zeros((p, K))
true_B[0]  = [1.5, -1.5, 0.0]
true_B[5]  = [-1.0, 0.0, 1.0]
true_B[12] = [0.0, 1.2, -1.2]

# Sample class labels via softmax.
eta = X @ true_B
labels = np.argmax(eta + 0.1 * rng.standard_normal((n, K)), axis=1)
```

Two ways to get prediction surfaces, depending on how you've picked λ.
The **path classifier** exposes the full sweep but is intentionally
prediction-free — it's for inspecting the trajectory:

```python
path = skein_glm.MultinomialMCPPathClassifier(
    gamma=3.0, n_lambdas=15, lambda_min_ratio=1e-2,
).fit(X, labels)

path.coefs_.shape       # (15, K, p) — coefficients along the path
path.intercepts_.shape  # (15, K)
path.lambdas_.shape     # (15,)
path.classes_           # ndarray (K,) — the unique label values
```

For predictions, use either a single-λ classifier (`MultinomialMCPClassifier`)
or the CV variant, both of which expose `predict_proba` /
`predict` / `decision_function`:

```python
clf = skein_glm.MultinomialMCPClassifier(
    lambda_=path.lambdas_[10],   # pick a λ from the path
    gamma=3.0,
).fit(X, labels)

clf.coef_.shape                  # (K, p) — sklearn convention
clf.intercept_.shape             # (K,)
clf.predict_proba(X).shape       # (n, K) — softmax probabilities
clf.predict(X).shape             # (n,) — argmax class labels
clf.decision_function(X).shape   # (n, K) — η values
```

### Symmetric parameterization

skein follows **glmnet's symmetric / "grouped" multinomial**: all `K`
columns of `B` are estimated freely. The softmax is invariant to
adding a constant to every column, but the L2-row penalty breaks
that redundancy by shrinking everything toward zero. Predictions are
invariant to the redundancy — you don't have to think about it.

This matches `glmnet(family="multinomial", type.multinomial="grouped")`.

### Labels can be any sortable type

`fit(X, y)` accepts integers, strings, or anything `np.unique` can
sort. The estimator stores the sorted unique labels on `classes_`
and decodes predictions back to the original dtype:

```python
labels_str = np.array(["cat", "dog", "fish"])[labels]
clf = skein_glm.MultinomialMCPPathClassifier(
    gamma=3.0, n_lambdas=10, lambda_min_ratio=1e-2,
).fit(X, labels_str)

clf.classes_              # array(['cat', 'dog', 'fish'])
clf.predict(X[:3])        # array(['dog', 'cat', 'fish'])
```

### Multinomial CV

`MultinomialMCPPathCV` scores by **multinomial deviance** (log-loss)
and uses **StratifiedKFold by class** so heavy class imbalance
doesn't produce class-empty folds:

```python
cv = skein_glm.MultinomialMCPPathCV(
    gamma=3.0, cv=5, random_state=0,
    n_lambdas=15, lambda_min_ratio=1e-2,
).fit(X, labels)
cv.lambda_best_
cv.predict(X[:5])
```

## Multi-task LS: K continuous outcomes

Multi-task LS fits `K` related linear regressions **jointly** under
a row-grouped penalty: feature `j` is selected if it's active in
*any* task.

```python
n, p, K = 200, 30, 4
X = rng.standard_normal((n, p))

# Truth: features 0, 5, 12 active; the rest are noise.
true_B = np.zeros((p, K))
true_B[0]  = [1.5, -1.0, 0.8, -1.2]
true_B[5]  = [0.7, -0.5, 1.1, 0.4]
true_B[12] = [0.0, 0.6, -0.5, 0.9]

Y = X @ true_B + 0.3 * rng.standard_normal((n, K))   # (n, K)

path = skein_glm.MultiTaskLassoPathRegressor(
    n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X, Y)

path.coefs_.shape        # (n_lambdas, K, p) — full path
path.intercepts_.shape   # (n_lambdas, K)
path.lambdas_.shape      # (n_lambdas,)

# Single-λ regressor matches sklearn's `MultiTaskLasso` shape — use it
# when you want a fitted model to call `predict` on.
m = skein_glm.MultiTaskLassoRegressor(
    lambda_=path.lambdas_[10],
).fit(X, Y)
m.coef_.shape            # (K, p)
m.intercept_.shape       # (K,)
m.predict(X).shape       # (n, K)
```

Useful when you have multiple correlated outcomes — e.g., gene
expression in different tissues, sales across regions, sensor
readings at multiple time points — and you believe the sparse support
is shared.

### Convention vs sklearn

skein and sklearn use different objective scalings for multi-task LS:

| Library | Data fidelity factor | Penalty constant |
|---|---|---|
| skein | `1/(2nK)` | `λ` |
| sklearn | `1/(2n)` | `α` |

Same minimizer at `λ_skein = α_sklearn / K`. The pytest suite has a
parity test against `sklearn.linear_model.MultiTaskLasso` at the
matching `(α, λ)` pair.

### Other multi-task penalties

```python
# Multi-task MCP via LLA (unbiased on active features) — full path.
path_mcp = skein_glm.MultiTaskMCPPathRegressor(
    gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X, Y)
path_mcp.coefs_.shape     # (n_lambdas, K, p)

# Multi-task elastic net (convex, useful for correlated tasks).
path_en = skein_glm.MultiTaskElasticNetPathRegressor(
    alpha=0.5, n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X, Y)

# CV scoring by mean per-task MSE — exposes `coef_ (K, p)` after refit.
cv = skein_glm.MultiTaskMCPPathCV(
    gamma=3.0, cv=5, random_state=0,
    n_lambdas=20, lambda_min_ratio=1e-2,
).fit(X, Y)
cv.lambda_best_
cv.coef_.shape            # (K, p)
cv.predict(X).shape       # (n, K)
```

## Why these two are the same problem

Both multinomial and multi-task LS share a **row-grouped coefficient
matrix** `B ∈ ℝ^(p×K)` with the penalty `λ · Σ_j w_j ‖B[j, :]‖_2`.
The only difference is the loss:

- Multinomial: cross-entropy on softmax(XB).
- Multi-task LS: squared-error on `Y - XB`.

Both reduce to a **multi-task LS on a virtual `(nK × pK)` design**
that the M2 block-CD machinery handles unchanged. For multinomial,
the prox-Newton outer loop linearizes the cross-entropy via Böhning's
diagonal majorization — every iteration becomes a multi-task LS
sub-problem. The same `MultiTaskDesign<X>` wrapper drives both.

## Stability selection on multi-task

`StabilitySelection` works on multi-task estimators too:

```python
ss = skein_glm.StabilitySelection(
    skein_glm.MultiTaskLassoPathRegressor(
        n_lambdas=15, lambda_min_ratio=1e-2,
    ),
    n_bootstraps=50, threshold=0.7, random_state=0, n_jobs=-1,
).fit(X, Y)
ss.stable_features_
```

The 3D `coefs_ (n_lambdas, K, p)` is collapsed via the **any-class
active** rule: feature `j` counts as selected if any of its K
coefficients is nonzero.

## Recap

| Setting | Class | Coef shape | predict |
|---|---|---|---|
| K-class single λ | `MultinomialMCPClassifier` | `(K, p)` | `(n,)` labels (+ `predict_proba`, `decision_function`) |
| K-class path | `MultinomialMCPPathClassifier` | `(n_lambdas, K, p)` | path attrs only — no `predict` |
| K-class CV | `MultinomialMCPPathCV` | `(K, p)` after refit | `(n,)` labels (+ `predict_proba`) |
| Multi-response single λ | `MultiTaskMCPRegressor` | `(K, p)` | `(n_pred, K)` |
| Multi-response path | `MultiTaskMCPPathRegressor` | `(n_lambdas, K, p)` | path attrs only |
| Multi-task CV | `MultiTaskMCPPathCV` | `(K, p)` after refit | `(n_pred, K)` |

Both families share the same `(datafit, row-grouped penalty)` shape
underneath. Once you understand this tutorial, you understand
everything in `skein_glm.multinomial` and `skein_glm.multitask`.

## Where to go next

You now know all nine tutorials. The natural next reading:

- The applied case studies in **[examples/](../examples/genomics.md)**
  — full analyses combining multiple features.
- The **[concepts](../concepts/index.md)** pages for the conceptual
  model behind the abstractions.
- The **[extending](../extending/penalty.md)** guides for building
  your own penalty / datafit / backend.
- The **[API reference](../api/index.md)** for the complete surface.
