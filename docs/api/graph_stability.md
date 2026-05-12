# Graphical edge stability

Bootstrap-based confidence in **edges** for graphical models.
Complements the EBIC tuner ([graph_selection](graph_selection.md)) by
quantifying *how often* an edge would be selected under resampling,
not just whether it is selected at one λ.

Two complementary wrappers around the M11 graphical estimators:

- {class}`~skein_glm.GraphicalStabilitySelection` — Meinshausen–
  Bühlmann (2010) stability selection lifted to **edges**. Sweep a
  user-supplied λ-grid; for each (bootstrap, λ) refit, record the
  off-diagonal nonzero pattern of `Θ̂`; aggregate to per-(λ, i, j)
  selection probability. The **stable edge set** is the edges whose
  max-over-λ probability crosses a threshold (default 0.6; the MB
  error-control bound requires > 0.5).
- {class}`~skein_glm.GraphicalBootstrap` — classic non-parametric
  (resample-with-replacement) bootstrap at a single λ. Returns the
  per-edge sampling distribution: mean, SD, `[α/2, 1−α/2]` quantile
  CIs, edge selection probability. This is the headline
  `bootnet::bootnet(type="nonparametric")` output used for edge
  error bars in network psychometrics.

Both work with single-population (`GraphicalLasso` / `GraphicalMCP` /
`GraphicalSCAD`) and joint (`JointGraphicalLasso` /
`JointGraphicalMCP`) base estimators. Family is auto-detected via the
estimator's `lambda_2` / `alpha` init param; no manual switching.

## Why

Field practice in network psychometrics is to fit a graphical lasso
at one λ chosen by EBIC, then **bootstrap** the edges to plot error
bars. R's `bootnet` is the canonical package for this; its output
(edge selection probability + CI on edge weight) is what
psychometric papers reproduce. Until now, the Python toolchain
(`sklearn.covariance.GraphicalLasso`) did not offer a clean
bootstrap utility — users would either drop to R or hand-roll a
resampling loop on top of sklearn fits.

`GraphicalBootstrap` matches `bootnet`'s output shape; users porting
from `bootnet` can plug skein in as a drop-in replacement that also
supports nonconvex edge penalties (MCP/SCAD) and joint estimation.

## Compatibility

Both classes require **raw observation data**, not a precomputed
covariance — bootstrap means we need to resample rows. Passing a
square symmetric `(p, p)` matrix is rejected with a clear error
message. For joint estimation, pass a list/tuple of per-population
`X^(k)` arrays.

The bootstrap loop is embarrassingly parallel via `joblib`; pass
`n_jobs=-1`. For fixed `random_state` the bootstrap indices are
pre-generated, so `n_jobs > 1` produces identical results to the
serial fit.

## API

```{eval-rst}
.. autoclass:: skein_glm.GraphicalStabilitySelection
   :members:

.. autoclass:: skein_glm.GraphicalBootstrap
   :members:
```

## Example

```python
import numpy as np
from skein_glm import GraphicalMCP, GraphicalStabilitySelection

rng = np.random.default_rng(0)
n, p = 200, 15
# Sparse precision: a chain Θ_{i, i+1} = -0.5, Θ_ii = 1.
Theta = np.eye(p)
for i in range(p - 1):
    Theta[i, i + 1] = Theta[i + 1, i] = -0.4
Sigma = np.linalg.inv(Theta)
L = np.linalg.cholesky(Sigma)
X = rng.standard_normal((n, p)) @ L.T

ss = GraphicalStabilitySelection(
    base_estimator=GraphicalMCP(gamma=3.0),
    lambdas=np.geomspace(0.5, 0.05, 8),
    n_bootstraps=100, threshold=0.6, n_jobs=-1, random_state=0,
).fit(X)
print(ss.stable_edges_)  # (n_stable, 2) — upper-triangular (i, j) pairs
```

## References

- Meinshausen, N. and Bühlmann, P. (2010). *Stability selection*.
  JRSS B 72(4): 417–473.
- van Borkulo, C. D. et al. (2017). *Network analysis of multivariate
  data in psychological science.* Nature Reviews Methods Primers
  2:60. (The methodology behind R's `bootnet`.)
- Friedman, J., Hastie, T., Tibshirani, R. (2008). *Sparse inverse
  covariance estimation with the graphical lasso.* Biostatistics 9(3):
  432–441.
