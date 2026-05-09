# Stability selection

Meinshausen–Bühlmann (2010) stability selection wraps any `*PathRegressor`
in a bootstrap-driven feature-selection meta-procedure. For each
bootstrap iteration:

1. Subsample the data (default half-sample without replacement).
2. Fit the underlying path estimator on the subsample.
3. Record the active set at each λ.

After `n_bootstraps` iterations, average the per-(feature, λ) activity
masks to get **selection probabilities** `Π_j(λ_k)`. The **stable set**
is `{j : max_k Π_j(λ_k) ≥ threshold}`.

## Why

Stability selection sidesteps the brittle "pick a single λ" decision.
The MB Theorem 1 result gives an explicit error-control bound on
expected false positives in the stable set, depending only on the
threshold and the typical active-set size — *not* on the chosen λ.

Skein's stability-selection module is the **headline M5.x
differentiator**: neither `glmnet`'s `cv.glmnet` nor `skglm` ships a
clean implementation, and `grpreg` has nothing comparable. The
bootstrap loop is embarrassingly parallel — pass `n_jobs=-1` to use
every core.

## Compatibility

`StabilitySelection` accepts any skein path estimator as `base_estimator`:

- **Scalar LS / GLM / Cox** (`MCPPathRegressor`,
  `LogisticMCPPathRegressor`, `CoxMCPPathRegressor`, etc.).
- **Group penalties** (`GroupLassoPathRegressor`,
  `GroupSCADPathRegressor`, `LogisticSparseGroupMCPPathRegressor`,
  etc.). For grouped variants, "selected" is evaluated at the group
  level — features in the same group share the same selection
  probability.
- **Multi-task / multinomial** (`MultiTaskLassoPathRegressor`,
  `MultinomialMCPPathClassifier`, etc.). The 3D `coefs_` shape
  `(n_lambdas, K, p)` is collapsed across the K axis with an
  "any-class active" rule.
- **Cox** is auto-detected via the `ties` attribute; pass outcomes
  as `y=(time, event)`.

## Tuning

- `n_bootstraps` (default 100): MB's bound improves asymptotically.
  50–200 is typical; smaller for quick exploratory runs, larger for
  publication.
- `sample_fraction` (default 0.5): MB recommend **half-sample
  subsampling without replacement**.
- `threshold` (default 0.6): must be > 0.5 for the MB error-control
  bound. Higher → fewer false positives but tighter recall.
- `n_jobs` (default `None` = serial): parallel bootstraps via
  `joblib`. `-1` uses all cores. For a fixed `random_state` the
  bootstrap indices are pre-generated, so `n_jobs > 1` produces
  identical selection probabilities.

## λ-grid choice

The shared λ-grid for all bootstraps is drawn from a single full-data
fit of the base estimator. **Avoid very small `lambda_min_ratio`**
(e.g. `1e-4`) — at near-OLS λs every bootstrap selects every feature,
which inflates `max_probabilities_` to 1 indiscriminately. Stick to
`lambda_min_ratio` ≥ `1e-2` and `n_lambdas` ≤ 50 for clean stability
plots.

## API

```{eval-rst}
.. autoclass:: skein_glm.StabilitySelection
   :members:
```

## Example

```python
import numpy as np
from skein_glm import MCPPathRegressor, StabilitySelection

rng = np.random.default_rng(0)
X = rng.standard_normal((300, 50))
y = X[:, [0, 5, 12]] @ np.array([1.5, -1.0, 0.8]) + 0.3 * rng.standard_normal(300)

ss = StabilitySelection(
    base_estimator=MCPPathRegressor(gamma=3.0, n_lambdas=20, lambda_min_ratio=1e-2),
    n_bootstraps=100,
    threshold=0.9,
    n_jobs=-1,
    random_state=0,
).fit(X, y)

print(ss.stable_features_)
# array([ 0,  5, 12])

X_stable = ss.transform(X)
# X_stable.shape == (300, 3)
```

## References

Meinshausen, N. and Bühlmann, P. (2010). *Stability selection*.
Journal of the Royal Statistical Society B 72(4), 417-473.
