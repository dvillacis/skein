# 10. Graphical lasso end-to-end

Estimate a sparse precision matrix, tune by EBIC, visualise the
recovered graph. The workflow most social-science groups use today
in R via `qgraph`/`bootnet`, now reproducible in Python with
nonconvex penalties available.

For the conceptual background, see
[Graphical models](../concepts/graphical_models.md).

## Synthetic ground truth

Start with a known sparse precision matrix so we can check what gets
recovered.

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(0)
p = 12

# Build a sparse SPD precision matrix.
theta_true = np.zeros((p, p))
edges = [(0, 1), (0, 4), (1, 2), (3, 5), (5, 6), (7, 8), (8, 9), (9, 10)]
for i, j in edges:
    theta_true[i, j] = theta_true[j, i] = -0.4
np.fill_diagonal(theta_true, 1.5)  # diagonal-dominant ⇒ PD

# Sample n = 250 observations from N(0, Θ⁻¹).
sigma = np.linalg.inv(theta_true)
n = 250
X = rng.multivariate_normal(np.zeros(p), sigma, size=n)
```

## Fit at a single α

```python
est = skein_glm.GraphicalLasso(alpha=0.1).fit(X)

precision = est.precision_
covariance = est.covariance_
print(est.info_["converged"], est.info_["outer_iter"])
```

`fit` accepts either raw `X (n, p)` or a precomputed `(p, p)`
symmetric covariance — the estimator sniffs the input shape and
symmetry to decide. If you've computed polychoric correlations
externally (the standard pre-processing for Likert-scale data in
psychometrics), feed those in directly.

## Choose α by EBIC

The Extended BIC of Foygel & Drton is the field-standard tuning
rule for graphical lasso. `gamma = 0.5` is the default used by
`bootnet`/`qgraph`.

```python
lambdas = np.geomspace(0.5, 0.01, 30)
result = skein_glm.ebic_path(
    X, skein_glm.GraphicalLasso, lambdas, gamma=0.5,
)
print(f"selected α = {result.best_lambda:.4f}")
print(f"recovered {result.best_estimator.precision_[np.triu_indices(p, k=1)].astype(bool).sum()} edges")
```

`result.best_estimator` is the fitted `GraphicalLasso` at the
EBIC-selected α; `result.lambdas`, `result.ebic`, `result.n_edges`
give the full path so you can plot the EBIC curve.

## Switch to MCP for unbiased edge weights

L1 systematically shrinks every nonzero edge by `α`. If you care
about the magnitudes of the partial correlations (not just whether
they're nonzero), MCP is the better choice:

```python
mcp_est = skein_glm.GraphicalMCP(alpha=0.1, gamma=3.0).fit(X)

# Compare to truth.
iu = np.triu_indices(p, k=1)
print("max abs error (L1):", np.abs(est.precision_[iu] - theta_true[iu]).max())
print("max abs error (MCP):", np.abs(mcp_est.precision_[iu] - theta_true[iu]).max())
```

For supported (truly nonzero) edges, MCP recovers values near their
true magnitudes; L1 leaves them visibly shrunken.

## Visualise the recovered graph

If `networkx` is installed:

```python
import networkx as nx
import matplotlib.pyplot as plt

theta = result.best_estimator.precision_
G = nx.Graph()
for i in range(p):
    G.add_node(i)
for i in range(p):
    for j in range(i + 1, p):
        if abs(theta[i, j]) > 1e-6:
            G.add_edge(i, j, weight=abs(theta[i, j]))

pos = nx.spring_layout(G, seed=0)
weights = [G[i][j]["weight"] for i, j in G.edges()]
nx.draw(
    G, pos, with_labels=True, node_color="lightblue",
    width=[3 * w for w in weights],
)
plt.show()
```

For the psychometrics-flavoured analysis on a real dataset, see
the worked example: [Psychometrics](../examples/psychometrics.md).

## Per-edge weights for adaptive glasso

Set `edge_weights[i, j] = 1 / |Θ̂_init[i, j]|` from an initial fit to
get adaptive glasso for free (Zhou et al. 2011):

```python
init = skein_glm.GraphicalLasso(alpha=0.05).fit(X)
weights = 1.0 / (np.abs(init.precision_) + 1e-6)
np.fill_diagonal(weights, 0.0)  # diag is ignored anyway

adaptive = skein_glm.GraphicalLasso(alpha=0.1, edge_weights=weights).fit(X)
```

Same machinery, same `fit`/`predict` surface — the only difference
is the `(p, p)` weight matrix wired through to the per-coordinate
prox.
