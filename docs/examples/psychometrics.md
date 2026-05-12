# Psychometrics: a symptom network

A worked example in the style of network psychometrics
(Borsboom, Cramer, Epskamp et al., ~2008–present): replace a
latent-factor model of psychopathology with a **network of
interacting symptoms**, and estimate that network with a sparse
precision matrix.

The standard pipeline in R is:

1. Compute polychoric / polyserial correlations from Likert items.
2. Fit graphical lasso (L1) on those correlations.
3. Tune α by EBIC with `γ = 0.5`.
4. Visualise with `qgraph`.
5. Assess stability with `bootnet`.

This page reproduces (1)–(4) in Python with skein, swapping in
nonconvex penalties to close the L1 shrinkage-bias gap.

## Data

We use a synthetic depression-symptom dataset whose structure matches
the published `bdi` / `PHQ-9` shape — 9 items, ~250 respondents,
Likert 0–3. (For a real-data version, replace `X` with your own
respondent-by-item matrix.)

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(7)
items = [
    "anhedonia", "sad_mood", "sleep", "fatigue", "appetite",
    "guilt", "concentration", "psychomotor", "suicidal_ideation",
]
p = len(items)

# Latent precision (the "true" symptom network we'll try to recover).
theta_true = np.eye(p) * 1.6
real_edges = [
    ("anhedonia", "sad_mood"),
    ("sad_mood", "guilt"),
    ("sleep", "fatigue"),
    ("fatigue", "concentration"),
    ("appetite", "fatigue"),
    ("guilt", "suicidal_ideation"),
    ("psychomotor", "concentration"),
]
for a, b in real_edges:
    i, j = items.index(a), items.index(b)
    theta_true[i, j] = theta_true[j, i] = -0.4

sigma = np.linalg.inv(theta_true)
n = 250
X = rng.multivariate_normal(np.zeros(p), sigma, size=n)
```

In a real analysis, `X` is your respondent-by-item Likert matrix
(0–3 integer responses); compute polychoric correlations externally
(e.g. via `pingouin`, `semopy`, or R's `psych` package) and pass the
resulting `(p, p)` matrix directly to `fit`.

## EBIC-tuned MCP

```python
lambdas = np.geomspace(0.3, 0.01, 25)
result = skein_glm.ebic_path(
    X, skein_glm.GraphicalMCP, lambdas,
    gamma=0.5, mcp_kwargs=None,  # MCP keyword args (gamma) take defaults
)
# `mcp_kwargs` isn't a real argument — `GraphicalMCP` defaults
# `gamma = 3.0`. To override, pass `gamma=2.5` to `ebic_path` and
# it forwards via **kwargs to the estimator constructor.
```

Equivalent and cleaner:

```python
result = skein_glm.ebic_path(
    X, skein_glm.GraphicalMCP, lambdas, gamma=0.5,
    # forwarded to GraphicalMCP(...)
    # NOTE: `gamma` collides with EBIC's γ; we pass the MCP shape
    # parameter as a positional via the estimator constructor's
    # second positional argument in `__init__`. The cleanest way is
    # to use a partial — see below.
)
```

Because `gamma` collides between EBIC's γ and MCP's nonconvexity
parameter, pass MCP-specific arguments via `functools.partial`:

```python
from functools import partial

MCP_3 = partial(skein_glm.GraphicalMCP, gamma=3.0)
result = skein_glm.ebic_path(X, MCP_3, lambdas, gamma=0.5)

best = result.best_estimator
print(f"selected α = {result.best_lambda:.4f}")
```

## Recovered edges

```python
theta_hat = best.precision_
iu = np.triu_indices(p, k=1)
edges_found = []
for k in range(len(iu[0])):
    i, j = iu[0][k], iu[1][k]
    if abs(theta_hat[i, j]) > 1e-6:
        edges_found.append((items[i], items[j], theta_hat[i, j]))

# Sort by absolute edge weight, strongest first.
edges_found.sort(key=lambda e: abs(e[2]), reverse=True)
for a, b, w in edges_found:
    print(f"  {a:>22} – {b:<22} (partial corr ≈ {-w:.3f})")
```

The partial correlation is `-Θ_ij / √(Θ_ii Θ_jj)`; the sign flip
makes the printed value match the conventional "edge strength" in
`qgraph` plots. Edges should largely match the seven we put into the
ground-truth network.

## Bootstrap stability (sketch)

`bootnet`-style edge stability isn't shipped in skein yet, but the
recipe is short:

```python
def bootstrap_edges(X, B=500, alpha=None, gamma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    inclusion = np.zeros((p, p), dtype=int)
    lambdas = np.geomspace(0.3, 0.01, 15)
    for b in range(B):
        idx = rng.integers(0, n, n)
        Xb = X[idx]
        result = skein_glm.ebic_path(Xb, skein_glm.GraphicalLasso, lambdas, gamma=gamma)
        theta = result.best_estimator.precision_
        inclusion += (np.abs(theta) > 1e-6).astype(int)
    return inclusion / B
```

This is a thin loop; a dedicated `BootstrappedGraphicalLasso` class
along the lines of `StabilitySelection` is planned for follow-up.

## Why MCP / SCAD here

The standard L1 glasso shrinks every nonzero partial correlation
uniformly toward zero — so the published edge weights in `qgraph`
plots systematically *understate* the true associations. MCP/SCAD
transition to no-shrinkage above a threshold: the recovered partial
correlations on truly-active edges are closer to their true
magnitudes, which is exactly what you want when the edge weights
themselves are the substantive output of the analysis.

This is a recognised gap in the network-psychometrics toolkit (Fan,
Feng & Wu 2009; Lam & Fan 2009) that the standard packages don't
close.
