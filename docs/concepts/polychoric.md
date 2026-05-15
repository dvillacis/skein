# Polychoric and polyserial correlations

Network psychometrics fits graphical models to *correlation* matrices,
not raw data — and on ordinal Likert responses, the right correlation
is **not** Pearson. Naïve Pearson on ordinal data systematically
under-estimates the underlying continuous correlation that would have
produced the observed responses; the resulting precision matrix has
biased edge weights and, more importantly, the wrong sparsity
pattern.

`skein` provides three preprocessing helpers in
{mod}`skein_glm.preprocessing` that recover the latent-Gaussian
correlation underlying ordinal data:

| Function | Use when |
|---|---|
| {func}`~skein_glm.polychoric_correlation` | All columns are ordinal (Likert items). |
| {func}`~skein_glm.polyserial_correlation` | One side ordinal, one side continuous. |
| {func}`~skein_glm.polychoric_covariance_matrix` | Mixed data: auto-detects column type and dispatches to polychoric / polyserial / Pearson. |

Each returns a correlation matrix you can pass directly as the
`cov` argument to {class}`~skein_glm.GraphicalLasso`,
{class}`~skein_glm.GraphicalMCP`, or {class}`~skein_glm.GraphicalSCAD`.

## The latent-variable model

Polychoric assumes that each observed ordinal response `X_j ∈ {0, 1,
..., L_j − 1}` is the discretization of a latent continuous
`Z_j ∼ N(0, 1)` via fixed thresholds:

$$
X_j = a \iff \tau_{j,a-1} \le Z_j < \tau_{j,a}.
$$

The latent vector $\mathbf{Z}$ is multivariate normal with
correlation matrix $\mathbf{R}$. Polychoric estimates $\mathbf{R}$
from the observed $\mathbf{X}$ — the correlation among the latent
Gaussians, *not* among the discretized responses.

Polyserial generalizes this to one ordinal and one continuous
variable: the continuous side is its own latent value, the ordinal
side is the discretization of another latent $Z_j$ that's correlated
with it.

## Olsson's two-step ML estimator

`polychoric_correlation` implements the well-known two-step procedure
of Olsson (1979):

**Step 1 — thresholds.** For each ordinal column, compute the
cumulative marginal frequencies $\hat{F}_a = \hat{P}(X_j \le a)$
and pull back through the inverse standard-normal CDF:

$$
\hat{\tau}_{j,a} = \Phi^{-1}(\hat{F}_a).
$$

**Step 2 — pairwise ρ.** For each pair $(j, k)$, maximize the
bivariate-normal log-likelihood

$$
\ell(\rho) = \sum_{a,b} n_{ab} \cdot \log \pi_{ab}(\rho),
$$

where $n_{ab}$ is the count of jointly observed `(X_j = a, X_k = b)`
and $\pi_{ab}(\rho)$ is the bivariate-normal probability over the
rectangle $[\hat{\tau}_{j,a-1}, \hat{\tau}_{j,a}] \times [\hat{\tau}_{k,b-1}, \hat{\tau}_{k,b}]$.
The maximizer is the polychoric correlation $\hat{\rho}_{jk}$.

`skein` uses Brent's bounded scalar optimizer on $\rho \in (-1, 1)$,
and computes the rectangle probabilities via the four-corner formula
on the bivariate-normal CDF.

Polyserial (Olsson, Drasgow & Dorans 1982) uses the analogous
profile-likelihood ML in $\rho$ with the continuous-side mean and
standard deviation plugged in at sample moments.

## End-to-end psychometrics pipeline

```python
import numpy as np
from skein_glm import (
    polychoric_correlation,
    GraphicalMCPRegressor,
    GraphicalBootstrap,
)

# X is your respondent-by-item Likert matrix (n=250, p=12, 5-level).
X = load_likert_responses()

# Step 1: latent-Gaussian correlation matrix from ordinal data.
R = polychoric_correlation(X)

# Step 2: graphical MCP with the polychoric matrix as input.
model = GraphicalMCPRegressor(alpha=0.08, gamma=3.0).fit(R)

# Step 3: bootstrap edge stability for uncertainty quantification.
bootstrap = GraphicalBootstrap(
    base_estimator=GraphicalMCPRegressor(alpha=0.08, gamma=3.0),
    n_bootstraps=500,
    random_state=0,
).fit(X)

print("Stable edges (selection prob > 0.6):")
stable = bootstrap.edge_selection_probabilities_ > 0.6
for i, j in np.argwhere(np.triu(stable, k=1)):
    print(f"  item {i} — item {j}")
```

## Mixed data

If some columns are continuous and others are ordinal,
{func}`~skein_glm.polychoric_covariance_matrix` auto-detects each
column's type (by unique-value count) and uses the appropriate
estimator for each pair:

```python
from skein_glm import polychoric_covariance_matrix

# Auto-detects: columns with > 10 unique values are continuous,
# others are ordinal.
R = polychoric_covariance_matrix(X_mixed)

# Or specify explicitly:
R = polychoric_covariance_matrix(
    X_mixed,
    continuous_mask=[False, False, False, True, True],
)
```

## When *not* to use polychoric

- **More than ~7 ordinal levels.** The polychoric–Pearson gap shrinks
  rapidly as the number of levels grows; for 7-level Likert or
  higher, Pearson is usually within 0.01–0.02 of polychoric and is
  faster and simpler.
- **Highly skewed marginals with very small cell counts.** The 0.5
  continuity correction inside the threshold step handles small
  counts, but at the extreme (e.g. n × p with cells of 0 or 1) the
  ML estimator becomes high-variance. Listwise deletion + a larger
  sample is often better than trying to rescue a sparse table.
- **Time series or panel data.** Polychoric assumes iid observations;
  the bivariate-normal likelihood is the iid joint, not the
  conditional. Use a model that respects the temporal structure.

## References

- **Olsson, U.** (1979). "Maximum likelihood estimation of the
  polychoric correlation coefficient." *Psychometrika* 44(4): 443–460.
- **Olsson, U., Drasgow, F., & Dorans, N. J.** (1982). "The
  polyserial correlation coefficient." *Psychometrika* 47(3):
  337–347.
- **Drasgow, F.** (1986). "Polychoric and polyserial correlations."
  In *The Encyclopedia of Statistics*.
- **Epskamp, S., Borsboom, D., & Fried, E. I.** (2018). "Estimating
  psychological networks and their accuracy: A tutorial paper."
  *Behavior Research Methods* 50: 195–212. — Network psychometrics
  context.
