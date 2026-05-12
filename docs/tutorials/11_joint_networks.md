# 11. Joint networks across populations

Estimate `K` precision matrices jointly when you have data from
multiple related populations (treatment vs control, age cohorts,
time points). Each population gets its own `Θ̂^(k)`; a group
penalty couples the same edge across populations so shared structure
gets reinforced but population-specific edges can still appear.

The method is *joint graphical lasso* (Danaher, Wang & Witten 2014,
group form).

## Setup

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(42)
p = 8

# Two populations sharing most edges but differing on a few.
theta1 = np.eye(p) * 1.5
theta2 = np.eye(p) * 1.5
shared = [(0, 1), (1, 2), (3, 4), (5, 6)]
only_in_1 = [(2, 5)]
only_in_2 = [(4, 7)]

for i, j in shared:
    theta1[i, j] = theta1[j, i] = -0.3
    theta2[i, j] = theta2[j, i] = -0.3
for i, j in only_in_1:
    theta1[i, j] = theta1[j, i] = -0.3
for i, j in only_in_2:
    theta2[i, j] = theta2[j, i] = -0.3

sigma1, sigma2 = np.linalg.inv(theta1), np.linalg.inv(theta2)
X1 = rng.multivariate_normal(np.zeros(p), sigma1, size=200)
X2 = rng.multivariate_normal(np.zeros(p), sigma2, size=200)
```

## Fit

```python
joint = skein_glm.JointGraphicalLasso(lambda_2=0.05).fit([X1, X2])

print(joint.n_populations_)        # 2
print(joint.precisions_[0].shape)  # (p, p)
print(joint.info_["converged"], joint.info_["iter"])
```

Each `joint.precisions_[k]` is the estimated `Θ̂^(k)`. `lambda_2`
controls the coupling: `0` decouples to independent fits, large
values collapse populations.

## The two extremes

```python
# λ_2 = 0 (essentially): decoupled — equivalent to fitting K separate
# GraphicalLasso models with α = 0.
decoupled = skein_glm.JointGraphicalLasso(lambda_2=1e-6).fit([X1, X2])

# λ_2 large: populations collapse to a shared graph.
collapsed = skein_glm.JointGraphicalLasso(lambda_2=10.0).fit([X1, X2])

iu = np.triu_indices(p, k=1)
print("decoupled diff:", np.abs(decoupled.precisions_[0][iu] - decoupled.precisions_[1][iu]).mean())
print("collapsed diff:", np.abs(collapsed.precisions_[0][iu] - collapsed.precisions_[1][iu]).mean())
```

The middle ground is what's interesting — `λ_2 ≈ 0.05` to `0.2`
typically lets the shared edges stand out while preserving real
between-population differences.

## Tune λ_2 by EBIC

```python
lambdas = np.geomspace(0.5, 0.01, 20)
result = skein_glm.joint_ebic_path(
    [X1, X2], skein_glm.JointGraphicalLasso, lambdas, gamma=0.5,
)
best = result.best_estimator
print(f"selected λ_2 = {result.best_lambda_2:.4f}")
print(f"recovered {result.n_edges_union[np.argmin(result.ebic)]} edges (union)")
```

The joint EBIC sums per-population log-likelihoods and counts the
*union* of active edges across populations.

## Compare per-population structure

After fitting, the natural question is "which edges are shared and
which differ":

```python
theta1_hat, theta2_hat = best.precisions_
active1 = np.abs(theta1_hat) > 1e-6
active2 = np.abs(theta2_hat) > 1e-6

shared = active1 & active2
pop1_only = active1 & ~active2
pop2_only = active2 & ~active1

iu = np.triu_indices(p, k=1)
print(f"shared edges: {shared[iu].sum()}")
print(f"pop-1-only:   {pop1_only[iu].sum()}")
print(f"pop-2-only:   {pop2_only[iu].sum()}")
```

## When to reach for MCP

If you care about unbiased estimates of partial correlations across
populations (not just the support pattern), use
`JointGraphicalMCP`. The group penalty becomes group MCP, which
relaxes to no shrinkage once an edge's L2 norm across populations is
large enough.

```python
joint_mcp = skein_glm.JointGraphicalMCP(lambda_2=0.05, gamma=3.0).fit([X1, X2])
```

The fit signature, attributes, and EBIC pipeline are identical to
`JointGraphicalLasso`.
