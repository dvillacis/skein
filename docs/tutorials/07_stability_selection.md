# 7. Stability selection

Cross-validation answers: "what's the best single λ for prediction?"
That's a useful question, but it's not the same as: "which features
are reliably part of the active set?" The first is a tuning problem;
the second is a *variable-selection* problem with explicit
false-positive control.

**Stability selection** (Meinshausen-Bühlmann 2010) attacks the
second question by running the path on many bootstrap subsamples and
asking: *across resamples, which features keep showing up?*

## The procedure

1. Subsample the data (default: half-sample without replacement).
2. Fit a path estimator on the subsample over a shared λ-grid.
3. Record the active set at each λ.
4. Repeat `n_bootstraps` times.
5. For each `(feature, λ)`, compute the **selection probability**
   `Π_j(λ_k)` = fraction of bootstraps where feature `j` was active
   at λ_k.
6. The **stable set** is `{j : max_k Π_j(λ_k) ≥ threshold}`.

The crucial detail: stability selection takes the **max over λ** —
which gives a single per-feature score and sidesteps the "pick a λ"
problem entirely.

## Walk-through

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
n, p = 300, 50
X = rng.standard_normal((n, p))

# Sparse truth: 4 features carry signal, 46 are noise.
true_beta = np.zeros(p)
active = np.array([0, 5, 12, 23])
true_beta[active] = [1.5, -1.0, 0.8, -1.2]
y = X @ true_beta + 0.3 * rng.standard_normal(n)
```

Wrap the path estimator in `StabilitySelection`:

```python
ss = skein_glm.StabilitySelection(
    base_estimator=skein_glm.MCPPathRegressor(
        gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2,
    ),
    n_bootstraps=100,
    threshold=0.9,
    n_jobs=-1,        # parallelize across cores
    random_state=0,
).fit(X, y)
```

The base estimator can be **any skein path estimator** — scalar,
group, GLM, multi-task, multinomial, Cox. `StabilitySelection` auto-
detects the type and aggregates correctly.

Read the results:

```python
ss.stable_features_
# array([ 0,  5, 12, 23])

ss.max_probabilities_[active]
# array([1.0, 1.0, 1.0, 1.0])  — the truly active features were
#                               selected in every bootstrap
```

The shape of `selection_probabilities_` is `(n_lambdas, n_features)`:

```python
ss.selection_probabilities_.shape   # (20, 50)
```

## Picking the threshold

Higher threshold → fewer false positives, tighter recall. The MB
paper recommends starting around 0.6 and adjusting upward if you have
a target false-positive budget:

```python
# Conservative (likely to drop borderline features).
ss_strict = skein_glm.StabilitySelection(
    skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2),
    n_bootstraps=100, threshold=0.95, random_state=0, n_jobs=-1,
).fit(X, y)

# Permissive (catches weaker signals).
ss_loose = skein_glm.StabilitySelection(
    skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2),
    n_bootstraps=100, threshold=0.6, random_state=0, n_jobs=-1,
).fit(X, y)

print("strict:", ss_strict.stable_features_)
print("loose: ", ss_loose.stable_features_)
```

The threshold must be > 0.5 — that's the regime where the MB error
bound applies.

## λ-grid choice matters

A subtle pitfall: at very small λ (near OLS), the path keeps every
feature active, so every bootstrap gets `Π_j ≈ 1` for every `j` and
the stable set explodes.

**Avoid `lambda_min_ratio` < 1e-2 for stability selection.** A
moderate path keeps the small-λ end away from the OLS regime, which
keeps noise features unselected across bootstraps.

```python
# DON'T do this:
ss_bad = skein_glm.StabilitySelection(
    skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=50, lambda_min_ratio=1e-4),
    n_bootstraps=50, threshold=0.6, random_state=0,
).fit(X, y)
# Stable set probably contains nearly every feature — useless.
```

## Comparing to CV

Stability selection often gives a **smaller, more confident** set
than CV. CV optimizes prediction error and tends to keep extra weakly
contributing features; stability selection optimizes selection
reliability.

```python
cv = skein_glm.MCPPathCV(
    gamma=3.0, n_lambdas=50, lambda_min_ratio=1e-2,
    cv=5, random_state=0,
).fit(X, y)
cv_active = np.flatnonzero(np.abs(cv.coef_) > 1e-6)

print("CV-best λ active set:", sorted(cv_active.tolist()))
print("Stable set (thresh=0.9):", sorted(ss.stable_features_.tolist()))
```

Both should include the truly active features. CV's set is usually a
**superset** that contains a few borderline noise features.

## Group-level stability

For grouped path estimators, stability is evaluated **at the group
level** — features in the same group share their selection
probability. This is the right behavior: in a group setting, you
either select the whole group or you don't.

```python
groups = np.repeat(np.arange(10), 5).astype(np.int64)  # 10 groups of 5

ss_g = skein_glm.StabilitySelection(
    skein_glm.GroupMCPPathRegressor(
        groups=groups, gamma=3.0, n_lambdas=15, lambda_min_ratio=1e-2,
    ),
    n_bootstraps=50, threshold=0.7, random_state=0, n_jobs=-1,
).fit(X, y)

# Per-feature probabilities are constant within each group.
ss_g.selection_probabilities_[5, :5]  # all the same
ss_g.selection_probabilities_[5, 5:10]  # all the same
```

## Cox PH stability

Pass survival outcomes as a tuple:

```python
time = rng.exponential(1.0, n)
event = (rng.uniform(size=n) < 0.7).astype(np.float64)

ss_cox = skein_glm.StabilitySelection(
    skein_glm.CoxMCPPathRegressor(gamma=3.0, n_lambdas=15, lambda_min_ratio=1e-2),
    n_bootstraps=50, threshold=0.6, random_state=0, n_jobs=-1,
).fit(X, (time, event))
```

The wrapper detects Cox via the `ties` attribute on the base
estimator and threads `(time, event)` through the per-bootstrap fit.

## Plotting the stability path

The full `selection_probabilities_` matrix lets you plot the **stability
path** — `Π_j(λ)` against `λ` for each feature. Truly active features
show up as curves that hit 1.0 across most of the path; noise features
stay near 0:

```python
import matplotlib.pyplot as plt   # not a skein dep
fig, ax = plt.subplots()
for j in range(p):
    color = "C1" if j in active else "0.7"
    ax.plot(ss.lambdas_, ss.selection_probabilities_[:, j], color=color)
ax.axhline(ss.threshold, color="k", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("λ")
ax.set_ylabel("selection probability")
```

## Recap

| Question | Tool |
|---|---|
| What's the best λ for prediction? | `*PathCV` — minimize test loss |
| Which features are reliably active? | `StabilitySelection` — max-over-λ probability |
| Single-shot model fit | `MCPPathRegressor` + `select_by_ic` |

Stability selection is skein's headline differentiator — neither
glmnet, skglm, nor grpreg ships a clean implementation. The bootstrap
loop is embarrassingly parallel; pass `n_jobs=-1` to use every core.

## Next

→ [8. Adaptive estimators](08_adaptive_estimators.md) — the oracle property
via two-stage refitting.
