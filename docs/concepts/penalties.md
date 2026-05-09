# Penalties

The penalty term shapes the regularization landscape: which
coefficients can go to exactly zero, how heavily large coefficients
are shrunk, whether features are selected individually or as groups.
`skein` ships six penalties spanning convex / nonconvex × scalar /
group, all wired through every datafit.

## The shape of the optimization problem

For any estimator in `skein`, the objective is

$$
\min_{\beta} \; \mathcal{L}(\beta) + P_\lambda(\beta)
$$

where $\mathcal{L}$ is the [datafit](datafits.md) and $P_\lambda$ is
the penalty, parameterized by a regularization strength $\lambda > 0$.
The penalty is what this page is about.

## Scalar penalties

Penalize each coefficient independently. The coefficient vector
$\beta \in \mathbb{R}^p$ has no group structure.

### Lasso (L1)

The classical $\ell_1$ penalty:

$$
P_\lambda(\beta) \;=\; \lambda \sum_{j=1}^p w_j \, |\beta_j|
$$

`skein` doesn't ship `LassoRegressor` as a separate class because
**MCP at large $\gamma$ is numerically equivalent to lasso**:

```python
import skein_glm
# γ = 1e6 is "lasso for all practical purposes" — the MCP nonconvexity
# only kicks in when |β_j| > γ·λ, which is essentially never at this γ.
model = skein_glm.MCPRegressor(lambda_=0.1, gamma=1e6).fit(X, y)
```

For a true lasso fit, use either `MCPRegressor(gamma=1e6)` or
`ElasticNetRegressor(alpha=1.0)` (see below). Both reach the lasso
solution; `ElasticNet` does it without the residual `β²/(2γ)` shape
that `MCP` carries.

### Elastic net (Zou & Hastie 2005)

Convex combination of $\ell_1$ (lasso) and squared $\ell_2$ (ridge):

$$
\rho_{\text{EN}}(\beta_j; \lambda, \alpha) \;=\; \alpha \, \lambda \, |\beta_j| + (1 - \alpha) \, \frac{\lambda \, \beta_j^2}{2}
$$

The parameter $\alpha \in [0, 1]$ trades between the two:

- **$\alpha = 1$**: pure lasso. Hard active-set selection.
- **$\alpha = 0$**: pure ridge. No selection — all features get
  shrunk smoothly toward zero.
- **$\alpha \in (0, 1)$**: classical elastic net. Selects features
  but handles correlated groups gracefully (lasso alone tends to
  arbitrarily pick one feature from a correlated cluster; the
  ridge term distributes weight more evenly).

Convex penalty, so coordinate descent converges to the global
optimum at every λ. The prox is closed-form: soft-threshold the
L1 component, then divide by the ridge shrinkage factor
$1 + (1-\alpha)\lambda \cdot \text{step}$.

```python
skein_glm.ElasticNetRegressor(lambda_=0.1, alpha=0.5)        # single λ
skein_glm.ElasticNetPathRegressor(alpha=0.5, n_lambdas=50)   # full path
skein_glm.ElasticNetPathCV(alpha=0.5, cv=10)                  # CV-selected λ
```

Matches `glmnet`'s `glmnet(family = "gaussian", alpha = ...)` shape.
Per-feature weights apply to **both** the L1 and L2 components:
`w_j · [α λ |β_j| + (1-α) λ β_j² / 2]`.

### MCP (minimax concave penalty, Zhang 2010)

A nonconvex penalty that interpolates between $\ell_1$ for small
$|\beta_j|$ and zero penalty for $|\beta_j| > \gamma \lambda$:

$$
\rho_{\text{MCP}}(\beta_j; \lambda, \gamma) =
  \begin{cases}
    \lambda |\beta_j| - \dfrac{\beta_j^2}{2\gamma}, & |\beta_j| \leq \gamma \lambda \\
    \dfrac{\gamma \lambda^2}{2}, & |\beta_j| > \gamma \lambda
  \end{cases}
$$

The parameter $\gamma > 1$ controls the transition speed. **Smaller
$\gamma$** means more aggressive sparsification (the unbiased region
starts sooner) but also a more nonconvex landscape. **Larger
$\gamma$** behaves more like lasso. Common defaults: $\gamma = 3$
(`ncvreg`'s default) for moderate nonconvexity, $\gamma = 1.5$ for
aggressive sparsification.

```python
skein_glm.MCPRegressor(lambda_=0.1, gamma=3.0)               # single λ
skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=50)          # full path
```

Why MCP over lasso: lasso is asymptotically biased — it shrinks
large coefficients toward zero, even when the data strongly supports
them. MCP fixes this by flattening the penalty for large $|\beta_j|$,
giving **nearly unbiased** estimates of the truly active features.
The cost is a nonconvex objective; `skein` handles this via Local
Linear Approximation (see "How nonconvex is solved" below).

### SCAD (smoothly clipped absolute deviation, Fan & Li 2001)

Same motivation as MCP but with a piecewise-linear (rather than
quadratic) descent in the second region:

$$
\rho_{\text{SCAD}}(\beta_j; \lambda, a) =
  \begin{cases}
    \lambda |\beta_j|, & |\beta_j| \leq \lambda \\
    \dfrac{2a\lambda |\beta_j| - \beta_j^2 - \lambda^2}{2(a-1)}, & \lambda < |\beta_j| \leq a\lambda \\
    \dfrac{(a+1)\lambda^2}{2}, & |\beta_j| > a\lambda
  \end{cases}
$$

The parameter $a > 2$ controls where SCAD transitions from $\ell_1$
to flat. `ncvreg`'s default is $a = 3.7$. SCAD and MCP usually
recover similar active sets in practice; MCP tends to be slightly
more aggressive at sparsifying.

```python
skein_glm.SCADRegressor(lambda_=0.1, a=3.7)
skein_glm.SCADPathRegressor(a=3.7, n_lambdas=50)
```

## Group penalties

Penalize **groups of coefficients jointly**. Useful when features
have known structure — categorical variables (one group per level
set), basis expansions (one group per spline / wavelet), genomic
pathways (one group per gene set), etc. A group either turns on
entirely (every coefficient in the group is non-zero) or turns off
entirely.

Groups are specified as an integer label vector of length $p$:

```python
groups = np.repeat(np.arange(10), 5)   # 50 features in 10 groups of 5
```

Coefficients with the same label are in the same group. Groups can
be unequal sizes; group `g`'s features are
`np.where(groups == g)[0]`.

### Group lasso (Yuan & Lin 2006)

The convex group penalty: each group's coefficients have their L2
norm penalized.

$$
P_\lambda(\beta) \;=\; \lambda \sum_{g=1}^G w_g \, \|\beta_g\|_2
$$

Either every coefficient in group $g$ is zero, or all of them are
non-zero. Within an active group, coefficients aren't sparsified.

```python
skein_glm.GroupLassoRegressor(groups=groups, lambda_=0.1)
skein_glm.GroupLassoPathRegressor(groups=groups, n_lambdas=50)
```

### Group MCP / Group SCAD

Nonconvex group penalties. Apply the MCP / SCAD shape to each
group's $\|\beta_g\|_2$ instead of each scalar $|\beta_j|$:

$$
P_\lambda(\beta) \;=\; \sum_{g=1}^G \rho_{\text{MCP}}(\|\beta_g\|_2; \lambda, \gamma)
$$

Same advantage as scalar MCP — nearly unbiased estimates of the
**truly active groups** — at the cost of a nonconvex objective.

```python
skein_glm.GroupMCPPathRegressor(groups=groups, gamma=3.0, n_lambdas=50)
```

This is the headline algorithm: solved via Local Linear Approximation
(LLA) outer iterations around a group block-CD inner solver, with
chunks of work parallelized across Rayon threads at the group level.

### Group elastic net

Convex group analog of the scalar elastic net: each group's
coefficients have both their L2 norm and squared L2 norm penalized.

$$
P_\lambda(\beta) \;=\; \lambda \sum_{g=1}^G w_g \left[ \alpha \, \|\beta_g\|_2 + \frac{(1-\alpha)}{2} \, \|\beta_g\|_2^2 \right]
$$

The parameter $\alpha \in [0, 1]$ trades between **all-or-nothing
group selection** ($\alpha = 1$, recovers plain group lasso) and
**per-block ridge shrinkage** ($\alpha = 0$, no group sparsity at
all). In between, the ridge term smooths the L2-norm prox so that
correlated groups don't get arbitrarily picked one-out-of-many the
way pure group lasso can.

The block prox is closed-form (rotationally symmetric in $x$ along
the ray from the origin through $z$): block soft-threshold by the
L1 component, then divide by the per-block ridge shrinkage factor
$1 + (1-\alpha)\lambda \cdot \text{step}$.

```python
skein_glm.GroupElasticNetRegressor(groups=groups, lambda_=0.1, alpha=0.5)
skein_glm.GroupElasticNetPathRegressor(groups=groups, alpha=0.5, n_lambdas=50)
skein_glm.GroupElasticNetPathCV(groups=groups, alpha=0.5, cv=10)
```

Convex penalty, so block coordinate descent converges to the global
optimum at every λ. Per-group weights $w_g$ apply to **both** the
L2 and L2² components.

### Sparse-group lasso (Simon et al. 2013)

Both **across-group** and **within-group** sparsity. The penalty is
a convex combination of group lasso and scalar lasso:

$$
P_\lambda(\beta) = \alpha \lambda \sum_j |\beta_j| + (1 - \alpha) \lambda \sum_g w_g \, \|\beta_g\|_2
$$

A group can be partially active (some coefficients on, some off).
$\alpha$ controls the within-group sparsity: $\alpha = 0$ is pure
group lasso; $\alpha = 1$ is pure scalar lasso; $\alpha = 0.5$ is
balanced.

```python
skein_glm.SparseGroupLassoPathRegressor(groups=groups, alpha=0.5, n_lambdas=50)
```

### Sparse-group MCP

The nonconvex sparse-group: MCP shape applied to both the within-group
$|\beta_j|$ terms and the across-group $\|\beta_g\|_2$ terms.

```python
skein_glm.SparseGroupMCPPathRegressor(groups=groups, gamma=3.0, alpha=0.5, n_lambdas=50)
```

## How nonconvex is solved

MCP and SCAD are nonconvex, so vanilla coordinate descent doesn't
have a global optimality guarantee. `skein` uses **Local Linear
Approximation (LLA)** — pioneered by Zou & Li (2008) for SCAD,
adapted to MCP and group MCP/SCAD here.

The recipe:

1. Linearize the nonconvex penalty at the current iterate $\beta^{(t)}$.
2. The linearization is a **weighted convex penalty** with weights
   $w_j^{(t)}$ depending on $|\beta_j^{(t)}|$ (for scalar) or
   $\|\beta_g^{(t)}\|_2$ (for group). For MCP these weights are
   $\max(0, w_j^{\text{base}} - |\beta_j^{(t)}|/(\gamma\lambda))$.
3. Solve the weighted convex problem (lasso / group lasso / sparse-
   group lasso) via coordinate descent.
4. Update weights from the new iterate; repeat until convergence
   (typically 2-5 outer iterations).

This is what `skein` does internally for any `*MCP*` or `*SCAD*`
estimator. You don't see it; the path solver returns a single
$\beta$ per λ.

## How groups in the inner loop are solved

For convex group lasso (and the LLA-linearized weighted group lasso
that MCP/SCAD reduce to), the inner solver is **group block
coordinate descent** with three optimizations:

1. **Block prox-gradient updates per group**, with operator-norm
   Lipschitz constants computed via power iteration on $X_g^\top X_g$
   (cached once per fit).
2. **Working sets**: the Tibshirani sequential strong rule plus a
   KKT-verification step picks a small subset of groups likely to be
   active at each λ; only those groups get full updates.
3. **Rayon parallelism**: independent groups can be updated in
   parallel; opt in via `parallel=True` on group estimators when
   you have many small groups.

For Cox / logistic / Poisson, the outer prox-Newton loop linearizes
the GLM loss into a weighted-LS surrogate that the group block-CD
solver then handles unchanged.

## Choosing a penalty

A rough decision tree:

- **No group structure?**
    - Tolerance for nonconvex objective: **MCP** (default).
    - Want a known-convex objective: **MCP at γ=1e6** (≈ lasso).
- **Group structure, want all-or-nothing per group:**
    - Tolerance for nonconvex objective: **group MCP**.
    - Want convex: **group lasso**.
- **Group structure, want mixed across- and within-group sparsity:**
    - **Sparse-group MCP** (nonconvex, less biased) or
      **sparse-group lasso** (convex).

For most analyses where features are heterogeneous (some categorical
multi-level, some continuous, some grouped by domain knowledge),
**sparse-group MCP at $\gamma = 3$, $\alpha = 0.5$** is a good
starting point — it admits both within-group and across-group
selection, and the nonconvex shrinkage avoids lasso's bias.

## See also

- [Datafits](datafits.md) — the loss term that pairs with your
  penalty.
- [Weights](weights.md) — per-feature `weights=` and per-group
  `weights=` arguments map onto the $w_j$ and $w_g$ in the
  penalty formulas above.
