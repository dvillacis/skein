# Psychometrics: a symptom network from Likert data

A worked example in the style of network psychometrics
(Borsboom, Cramer, Epskamp et al., ~2008–present): replace a
latent-factor model of psychopathology with a **network of
interacting symptoms**, and estimate that network with a sparse
precision matrix.

The standard R pipeline (`qgraph` + `bootnet`) is:

1. Compute polychoric / polyserial correlations from Likert items.
2. Fit graphical lasso (L1) on those correlations.
3. Tune α by EBIC with `γ = 0.5`.
4. Visualise with `qgraph`.
5. Assess stability with `bootnet`.

This page reproduces (1)–(3) and (5) **entirely inside `skein_glm`**,
with two upgrades over the standard R pipeline:

- **Nonconvex penalty on edges** (MCP) closes the L1 shrinkage-bias
  gap that systematically under-estimates edge magnitudes in
  conventional `qgraph` plots.
- **FDR control on edges** (Benjamini–Hochberg over the bootstrap
  distribution) replaces the "stability threshold" heuristic with a
  per-edge expected-false-discovery guarantee.

## Data

We simulate Likert responses from a planted symptom network. In a
real analysis, ``X`` is your respondent-by-item matrix of integer
Likert responses (e.g. 0–3 on the PHQ-9).

```python
import numpy as np
import skein_glm
from functools import partial
from sklearn.base import BaseEstimator
```

```python
rng = np.random.default_rng(7)
items = [
    "anhedonia", "sad_mood", "sleep", "fatigue", "appetite",
    "guilt", "concentration", "psychomotor", "suicidal_ideation",
]
p = len(items)

# Latent precision (the "true" symptom network we want to recover).
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
    theta_true[i, j] = theta_true[j, i] = -0.5

# Draw continuous responses from the latent Gaussian, then
# discretise to a 4-level Likert scale (0–3) via fixed thresholds.
sigma = np.linalg.inv(theta_true)
n = 400
Z = rng.multivariate_normal(np.zeros(p), sigma, size=n)
thresholds = np.array([-1.0, 0.0, 1.0])
X = np.digitize(Z, thresholds)  # integer 0–3 per cell
print(f"Likert levels per item: {[len(np.unique(X[:, j])) for j in range(p)]}")
```

## Step 1: polychoric correlation

Naïve Pearson on ordinal Likert data systematically under-estimates
the latent Gaussian correlation. {func}`~skein_glm.polychoric_correlation`
inverts the discretization step using the Olsson (1979) two-step
maximum-likelihood estimator and recovers the latent ``R̂``.

```python
R = skein_glm.polychoric_correlation(X)

# Sanity check: polychoric should be substantially closer to the
# true latent correlation than Pearson on the discretised data.
true_R = sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
pearson = np.corrcoef(X, rowvar=False)
print(f"max |R_polychoric − R_true| = {np.max(np.abs(R - true_R)):.3f}")
print(f"max |R_pearson    − R_true| = {np.max(np.abs(pearson - true_R)):.3f}")
```

The polychoric matrix is approximately the *true* latent
correlation; the Pearson matrix attenuates it because of the
discretization. See the
[polychoric concept page](../concepts/polychoric.md) for the
theoretical background.

## Step 2: EBIC-tuned graphical MCP

The MCP penalty has two free parameters: the regularization
strength ``α`` (chosen by EBIC) and the nonconvexity shape ``γ``
(left at the default ``3.0``). EBIC also has its own ``γ`` parameter
(usually called ``γ_EBIC``, default ``0.5``); we resolve the
name collision via :func:`functools.partial`:

```python
MCP_3 = partial(skein_glm.GraphicalMCP, gamma=3.0)

lambdas = np.geomspace(0.5, 0.02, 25)
result = skein_glm.ebic_path(R, MCP_3, lambdas, gamma=0.5, n=n)

best = result.best_estimator
print(f"EBIC-selected α = {result.best_lambda:.4f}")
```

We pass ``R`` (the polychoric correlation) and ``n`` explicitly
because :func:`~skein_glm.ebic_path` detects pre-computed
correlation input by shape and needs the original sample size for
the BIC term.

## Step 3: recovered edges (point estimate)

```python
theta_hat = best.precision_
iu = np.triu_indices(p, k=1)
print("Recovered edges (partial correlation ≈ -Θ_ij / √(Θ_ii Θ_jj)):")
for k in range(len(iu[0])):
    i, j = iu[0][k], iu[1][k]
    if abs(theta_hat[i, j]) > 1e-6:
        pcor = -theta_hat[i, j] / np.sqrt(theta_hat[i, i] * theta_hat[j, j])
        marker = "*" if (items[i], items[j]) in [tuple(sorted(e)) for e in real_edges] else " "
        print(f" {marker} {items[i]:>20} – {items[j]:<20}  pcor ≈ {pcor:+.3f}")
```

Edges marked ``*`` are in the ground truth. With MCP the recovered
partial correlations are close to the true magnitudes — the L1
glasso would shrink them all toward zero.

## Step 4: edge inference via bootstrap + FDR

The point-estimate network tells us *which* edges the EBIC fit
selected; bootstrap-based FDR control tells us how many are likely
to be false discoveries.

To bootstrap a polychoric-input pipeline we need each bootstrap
sample to re-run polychoric on its own subsample of Likert
responses (Pearson on resampled Likert data would inherit the
same attenuation bias the polychoric is meant to fix). Wrap the
two-step into a small estimator that
:class:`~skein_glm.GraphicalBootstrap` can drive:

```python
class PolychoricGraphicalMCP(BaseEstimator):
    """Polychoric correlation → GraphicalMCP, in one fit step."""

    def __init__(self, alpha: float = 0.1, gamma: float = 3.0) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X):
        R = skein_glm.polychoric_correlation(X)
        self._inner = skein_glm.GraphicalMCP(
            alpha=self.alpha, gamma=self.gamma
        ).fit(R)
        self.precision_ = self._inner.precision_
        self.covariance_ = self._inner.covariance_
        return self


boot = skein_glm.GraphicalBootstrap(
    base_estimator=PolychoricGraphicalMCP(
        alpha=result.best_lambda, gamma=3.0
    ),
    n_bootstraps=300,
    random_state=0,
).fit(X)
```

Now apply Benjamini–Hochberg FDR to the per-edge bootstrap
distribution:

```python
fdr_out = boot.fdr_threshold(fdr=0.10)

print("Edges retained at FDR ≤ 10%:")
for k in range(len(iu[0])):
    i, j = iu[0][k], iu[1][k]
    if fdr_out["reject"][i, j]:
        pcor = -theta_hat[i, j] / np.sqrt(theta_hat[i, i] * theta_hat[j, j])
        adj_p = fdr_out["adjusted_pvalues"][i, j]
        marker = "*" if (items[i], items[j]) in [tuple(sorted(e)) for e in real_edges] else " "
        print(f" {marker} {items[i]:>20} – {items[j]:<20}  pcor ≈ {pcor:+.3f}  adj-p = {adj_p:.3g}")
```

For applications where even a single false edge is unacceptable
(e.g. clinical decision support), swap to family-wise error
control with Holm:

```python
fwer_out = boot.fwer_threshold(fwer=0.05, method="holm")
print(f"Edges at FWER ≤ 5%: {fwer_out['reject'].sum() // 2}")
```

See the [edge inference concept page](../concepts/graph_inference.md)
for the bootstrap p-value definition (it's a non-strict-inequality
two-sided p-value, designed to handle sparse-estimator zeros
correctly) and the trade-offs between FDR and FWER on graphs.

## Why MCP instead of L1?

The standard L1 glasso shrinks every nonzero partial correlation
uniformly toward zero — so the published edge weights in `qgraph`
plots systematically under-state the true associations. MCP and
SCAD transition to no-shrinkage above a threshold: recovered
partial correlations on truly-active edges land closer to their
true magnitudes, which matters when the edge weights themselves are
the substantive output of the analysis.

This is a recognised gap in the network-psychometrics toolkit (Fan,
Feng & Wu 2009; Lam & Fan 2009; Zhou et al. 2011) that the standard
R packages (`glasso`, `qgraph`, `bootnet`, `EstimateGroupNetwork`)
don't close.

## See also

- [Concepts: Polychoric correlations](../concepts/polychoric.md) —
  Olsson's two-step ML estimator.
- [Concepts: Graphical models](../concepts/graphical_models.md) —
  glasso, MCP/SCAD on edges, joint estimation.
- [Concepts: Edge-level inference](../concepts/graph_inference.md) —
  BH FDR, Holm/Bonferroni FWER, MB stability bound.
- [Tutorial: graphical lasso](../tutorials/10_graphical_lasso.md) —
  step-by-step introduction.
- [Worked example: joint networks across populations](../tutorials/11_joint_networks.md)
  — Danaher–Wang–Witten group form.
