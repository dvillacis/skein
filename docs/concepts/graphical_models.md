# Graphical models

The four axes (penalty, datafit, weights, backend) compose to fit
sparse regression. A small extension — exchanging the residual-shaped
datafit for the log-det barrier — lets the same machinery fit
**sparse precision matrices**, the workhorse of Gaussian graphical
models.

## Precision matrices and conditional independence

For a centred Gaussian `X ∼ N(0, Σ)` with covariance `Σ`, the
**precision matrix** is `Θ = Σ⁻¹`. Its off-diagonal entries encode
**conditional independence**: `Θ_{ij} = 0` if and only if `X_i ⫫ X_j |
X_{rest}` — variables `i` and `j` are independent given every other
variable. Drawing an undirected edge wherever `Θ_{ij} ≠ 0` gives the
**dependence graph**.

Raw maximum likelihood gives `Θ̂ = S⁻¹`, the inverse sample covariance.
With `n` samples and `p` variables, `Θ̂` is unstable (and undefined
when `n ≤ p`); every off-diagonal is nonzero by sampling noise, so the
recovered "graph" is fully connected.

## Graphical lasso

Friedman, Hastie & Tibshirani (2008) regularise the MLE with an L1
penalty on the off-diagonals:

$$
\min_{\Theta \succ 0} \; -\log\det\Theta + \mathrm{tr}(S\Theta) + \alpha \sum_{i \ne j} w_{ij}\,|\Theta_{ij}|.
$$

The zeros pattern of the solution is the recovered graph. Larger
`α` produces sparser graphs. Per-edge weights `w_{ij}` express prior
information (zero weight = "this edge is free", large weight = "this
edge should be hard to include").

The block-coordinate-descent algorithm peels off one column of `Θ` at
a time and reduces the column update to a **weighted lasso** on the
remaining `p − 1` coordinates — exactly the
`(LeastSquares, Lasso)` problem skein already solves elsewhere. The
graphical lasso solver in skein wraps that loop around the existing
[CD machinery](../api/estimators-ls.md), reusing penalties, weights,
and prox primitives unchanged.

## Why MCP and SCAD on edges

L1 estimates *every* nonzero off-diagonal with a shrinkage bias of
order `α`. For social-science applications where the partial
correlations themselves carry interpretive weight (the edge magnitude
is the effect size, not just its presence), this systematic
shrinkage matters. **MCP** and **SCAD** transition smoothly from L1
behaviour near zero to no-shrinkage above a threshold (`γ · α` for
MCP, `a · α` for SCAD): the support pattern is similar to L1's, but
the off-diagonals that survive are estimated near their true
magnitudes.

This is the same trade-off as in scalar MCP/SCAD versus lasso ([see
the penalty page](penalties.md)); applied to the inverse covariance,
it closes a well-known gap in tools like R's `glasso`, `qgraph`, and
`sklearn.covariance.GraphicalLasso`, none of which support nonconvex
penalties on edges.

## Per-edge weights as adaptive glasso

The `edge_weights` argument is the single hook for every weighted
variant. Setting `edge_weights[i, j] = 1 / |Θ̂_{ij}^{(0)}|` for an
initial estimate `Θ̂^{(0)}` (e.g. from a coarse L1 fit) gives
**adaptive graphical lasso** (Zhou et al. 2011) — the second
stage's penalty is data-driven and inherits the oracle property
under standard conditions. It's the same per-coordinate-weight
mechanism that powers
[`AdaptiveLassoPathRegressor`](../api/estimators-adaptive.md);
[see Weights](weights.md) for the broader picture.

## Joint estimation across populations

For `K` related populations sharing the same `p` variables (e.g. the
same psychometric instrument administered to treatment vs control
groups, or the same biomarker panel measured pre vs post
intervention), one can either fit `K` independent graphical lassos
(maximising flexibility, losing power for shared edges) or pool
everything into one fit (losing all between-population differences).
**Joint graphical lasso** (Danaher, Wang & Witten 2014) sits between
the two: it fits `K` precision matrices simultaneously, coupled by a
group penalty on each edge across populations:

$$
\min_{\{\Theta^{(k)}\succ 0\}} \;
\sum_{k=1}^K n_k\bigl[-\log\det\Theta^{(k)} + \mathrm{tr}(S^{(k)}\Theta^{(k)})\bigr]
+ \lambda \sum_{i \ne j} w_{ij}\,
\bigl\lVert (\Theta^{(1)}_{ij}, \ldots, \Theta^{(K)}_{ij}) \bigr\rVert_2.
$$

The coupling parameter `λ` controls how much populations align:
- `λ = 0` recovers `K` independent fits.
- `λ → ∞` collapses all populations to the same precision matrix.
- Intermediate `λ` lets the same edge be active in some populations
  and zero in others *individually*, but encourages overlap.

The group form (above) penalises the L2 norm of an edge across
populations; an active edge in any population pays a "shared cost".
This is the variant skein implements. The *fused* form (TV-penalty
across populations) is a follow-up.

The solver is an ADMM kernel; the Z-update is exactly an
edge-wise call to an existing
[`GroupLasso`](penalties.md) / `GroupMCP` prox.

## Model selection: EBIC

The standard tuning rule in network psychometrics is the **Extended
Bayesian Information Criterion** (Foygel & Drton 2010):

$$
\text{EBIC}(\alpha) = -2\,\hat\ell + |\hat E(\alpha)| \log n + 4 \gamma |\hat E(\alpha)| \log p,
$$

where `|Ê(α)|` is the number of nonzero off-diagonal entries and
`γ ∈ [0, 1]` is the EBIC strength (`γ = 0` is plain BIC; `γ = 0.5` is
the field default for graphical models). EBIC penalises edge count
more aggressively than BIC and produces stable graphs even at small
`n / p`.

`ebic_path` walks a λ-grid, fits a graphical estimator at each, and
returns the EBIC-selected model along with the full path. The joint
analogue (`joint_ebic_path`) sums per-population log-likelihoods and
counts the *union* of active edges across populations.

## Where to next

- **Tutorial**: [Graphical lasso end-to-end](../tutorials/10_graphical_lasso.md)
  — simulate sparse Θ, fit, tune by EBIC, visualise the graph.
- **Tutorial**: [Joint networks across populations](../tutorials/11_joint_networks.md)
  — joint glasso for multi-group studies.
- **Worked example**: [Psychometrics](../examples/psychometrics.md)
  — replicating a published symptom-network analysis.
- **API**: [Graphical estimators](../api/estimators-graphical.md)
  and [Graph model selection](../api/graph_selection.md).
