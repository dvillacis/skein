# Weights

Three independent weight axes — per-sample, per-feature, and per-group
— wired through every solver. Most R packages support some;
none support all three. `skein`'s name is a deliberate metaphor:
each axis is a thread you can tighten independently of the others.

## The three axes

| Axis           | Length              | Where it lives                | Statistical role                                |
|----------------|---------------------|-------------------------------|-------------------------------------------------|
| Per-sample     | $n$ (samples)       | `Datafit::sample_weights`     | Frequency / propensity / survey / importance    |
| Per-feature    | $p$ (features)      | `Penalty::weights`            | Adaptive lasso, prior knowledge, unpenalized intercept |
| Per-group      | $G$ (groups)        | `GroupPenalty::weights`       | Structured penalties (size correction, prior)   |

Each axis affects the optimization differently. They compose: you
can supply all three at the same time on a group estimator with a
GLM datafit.

## Per-sample weights $w_i$

The datafit becomes:

$$
\mathcal{L}(\beta) \;=\; \frac{1}{n} \sum_{i=1}^n w_i \, \ell(y_i, x_i^\top \beta)
$$

Pass via the datafit's constructor. For `LeastSquares`, the
sklearn-style fit accepts `sample_weights` directly; for GLMs the
weights enter through the working-response calculation in
prox-Newton.

### When to use

- **Aggregated / frequency-weighted data**: row $i$ summarizes
  $w_i$ independent observations. Common in survey, public-health,
  and operations-research datasets.
- **Inverse-propensity weights** for ATE / ATT estimation.
- **Class-imbalance correction**: minority class rows get a higher
  weight so the loss isn't dominated by the majority.
- **Sampling-design corrections** for survey data.

### Behavior

Per-sample weights affect:

- The loss value (obviously).
- The gradient: $\nabla_\beta \mathcal{L}$ becomes weighted.
- The coordinate Lipschitz constant: `coord_lipschitz(j)` returns
  $(1/n) \sum_i w_i x_{ij}^2$, not the unweighted column norm.

This means the same set of true coefficients can give meaningfully
different fits depending on whether you supply weights, even on the
same data. There's no "weight ignored" failure mode in `skein`; if
you pass weights, every code path uses them.

## Per-feature weights $w_j$

The penalty becomes:

$$
P_\lambda(\beta) \;=\; \lambda \sum_{j=1}^p w_j \, \rho(\beta_j; \cdot)
$$

where $\rho$ is the per-coordinate penalty shape (lasso $|\beta_j|$,
MCP, SCAD, etc.). Pass via the estimator's `weights=` argument:

```python
import numpy as np
import skein_glm

# Don't penalize the first feature (e.g., a known-relevant baseline);
# penalize others at uniform weight 1.
weights = np.ones(p)
weights[0] = 0.0
model = skein_glm.MCPPathRegressor(gamma=3.0, weights=weights).fit(X, y)
```

### Use cases

- **Unpenalized features**: `w_j = 0` removes feature $j$ from the
  penalty. Most often used for the intercept (which `skein` handles
  internally via column augmentation, so you don't need to set
  `w_p = 0` yourself).
- **Adaptive lasso / adaptive MCP** (Zou 2006): fit a coarse
  preliminary model, set $w_j \propto 1/|\hat\beta_j|^\eta$, refit.
  The result has oracle-consistency properties under reasonable
  conditions. M5.x roadmap entry promotes this to a one-shot
  `AdaptiveLasso` estimator.
- **Prior knowledge**: features known to be relevant from prior
  studies get smaller $w_j$ (penalized less).
- **Group-size correction** for sparse-group lasso: see below.

### Behavior with standardization

If `standardize=True`, `skein` rescales per-feature weights internally
to match the standardized-coefficient space:
$\tilde w_j = w_j / s_j$ where $s_j$ is the column std. The
returned `coef_` is in original-feature scale, so the per-feature
weights you pass are interpreted in that same scale. You don't need
to think about this — it's handled automatically — but it's worth
knowing if you're debugging.

## Per-group weights $w_g$

The group penalty becomes:

$$
P_\lambda(\beta) \;=\; \lambda \sum_{g=1}^G w_g \, \|\beta_g\|_2
$$

(or analogously for group MCP / SCAD). Pass via the group estimator's
`weights=` argument.

### Use cases

- **Group-size correction**: the canonical recipe is $w_g = \sqrt{|g|}$
  where $|g|$ is the number of features in group $g$. Without this,
  larger groups are systematically more penalized (because $\|\beta_g\|_2$
  scales with $\sqrt{|g|}$ for non-zero coefficients of unit
  magnitude). This is the default in `grpreg::grpreg`.

- **Prior across groups**: groups corresponding to
  prior-knowledge-supported features get smaller $w_g$.

- **Hierarchical penalties**: combine with overlapping group lasso
  (M6 roadmap) to express tree-structured priors.

```python
groups = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
group_sizes = np.bincount(groups)
weights = np.sqrt(group_sizes)   # group-size correction
model = skein_glm.GroupLassoPathRegressor(
    groups=groups, weights=weights, n_lambdas=50,
).fit(X, y)
```

### Behavior with standardization

Per-group weights are **not rescaled** by `standardize=True`. The
group penalty is interpreted as applying in the standardized-coefficient
space (matching `glmnet`'s `cv.glmnet(family="multinomial", type.multinomial="grouped")`
convention). The intercept group, when added internally for column
augmentation, gets weight 0.

## Sparse-group: per-coordinate weights too

The sparse-group lasso / MCP penalties have a fourth weight axis —
**per-coordinate L1 weights** within the within-group L1 term. Pass
via `coord_weights=` on `SparseGroup*` estimators:

```python
skein_glm.SparseGroupMCPPathRegressor(
    groups=groups,
    weights=group_weights,         # per-group L2 penalty
    coord_weights=coord_weights,   # per-feature L1 penalty
    gamma=3.0, alpha=0.5,
)
```

This lets you express, e.g., "prior says feature 5 is unlikely to
matter on its own, but if its group is active, treat it normally" by
setting `coord_weights[5]` large and leaving its `groups[5]` as
normal.

## Composing all three axes

A logistic group MCP fit with all three weight types active:

```python
import numpy as np
import skein_glm

n, p, G = 1000, 100, 20
X = np.random.standard_normal((n, p))
y = (X[:, :3].sum(axis=1) > 0).astype(float)
groups = np.repeat(np.arange(G), p // G)

sample_weights = np.where(y == 1, 3.0, 1.0)         # upweight positives
feature_weights = np.ones(p)
feature_weights[:3] = 0.5                            # known-relevant features
group_weights = np.sqrt(np.bincount(groups))         # size correction

model = skein_glm.LogisticGroupMCPPathRegressor(
    groups=groups,
    weights=group_weights,
    gamma=3.0,
    n_lambdas=50,
).fit(X, y)
# Note: per-sample weights for GLM datafits are passed through the
# fit's `sample_weight=` arg in sklearn convention; per-feature
# weights via the constructor's `weights=` for scalar penalty
# estimators (this group estimator uses `weights=` for groups).
# v0.1 surface evolves; see API ref for the current arg names.
```

## Why three axes matter

In real applied workflows, you usually want at least two:

- **Genomics**: per-sample (batch / cohort weights), per-feature
  (adaptive from a prior study), per-group (gene set size).
- **Survey statistics**: per-sample (design weights), per-feature
  (some demographic fields known to matter).
- **NLP**: per-sample (importance weights from upstream curation),
  per-group (token / category structure).

R packages cover one axis cleanly (`glmnet` does per-feature;
`grpreg` does per-group; `survey` does per-sample) but combining
them requires hand-rolled code and breaks the optimizer's
performance. `skein` makes the combination first-class so you
can move all three knobs together without thinking about whether
the gradient is consistent.

## See also

- [Penalties](penalties.md) — where per-feature and per-group
  weights enter mathematically.
- [Datafits](datafits.md) — where per-sample weights enter.
