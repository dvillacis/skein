# Edge-level inference for graphical models

Mainstream graphical-model packages report point estimates of the
precision matrix `Θ̂` and — at best — a per-edge bootstrap selection
probability. That's enough for *visualization*, but it doesn't
answer the question working network-psychometrics or network-biology
analysts actually ask:

> *"Which of these edges are real, controlled at some error rate
> across the whole graph?"*

`skein` answers that question with three composable helpers in
{mod}`skein_glm.graph_inference`:

| Function | Controls | Returns |
|---|---|---|
| {func}`~skein_glm.edge_fdr_threshold` | Benjamini–Hochberg FDR | edge set + adjusted p-values |
| {func}`~skein_glm.edge_fwer_threshold` | Bonferroni or Holm FWER | edge set + adjusted p-values |
| {func}`~skein_glm.mb_stability_threshold` | Meinshausen–Bühlmann ``E[V]`` bound | stability threshold ``π_thr`` |

All three are bootstrap-based and require no distributional
assumption on the data — they exploit only the bootstrap's
exchangeability. The first two operate on a fitted
{class}`~skein_glm.GraphicalBootstrap`; the third operates on a fitted
{class}`~skein_glm.GraphicalStabilitySelection`. Convenience methods
(`bootstrap.fdr_threshold(0.05)`, `stability.mb_threshold(1.0)`)
forward to the functions.

## Bootstrap p-values for sparse estimators

For each edge `(i, j)`, the per-edge two-sided bootstrap p-value
tests `H0: Θ_ij = 0`:

$$
\hat{p}_{ij} = 2 \cdot \min\!\left(\hat{P}(\hat{\Theta}^*_{ij} \ge 0),\; \hat{P}(\hat{\Theta}^*_{ij} \le 0)\right).
$$

The inequalities are **non-strict** — zeros count on *both* sides.
That's essential for sparse estimators like graphical lasso, MCP,
and SCAD: a null edge's bootstrap distribution is exactly zero on
every replicate, both probabilities equal 1, the doubled minimum is
2, clipped to 1. A strict-inequality formulation would mistakenly
report the smallest possible p-value for every null edge.

The p-value is bounded below at `2 / B` (smallest representable
two-sided bootstrap p) and above at 1.

## False Discovery Rate (Benjamini–Hochberg)

For a family of `m = p(p − 1)/2` edge hypotheses (or
`K · p(p − 1)/2` for joint estimators — populations are pooled into
a single family), the BH step-up adjustment is

$$
\hat{p}^{(\text{BH})}_{(r)} = \min_{s \ge r} \frac{m \cdot \hat{p}_{(s)}}{s}.
$$

Reject all edges with `p^(BH) ≤ q` for nominal FDR `q`. Under
positive-regression-dependence-on-subset (which the bootstrap
distribution of `Θ̂` typically satisfies for sparse estimators) the
expected proportion of false discoveries is bounded by `q`.

```python
from skein_glm import GraphicalLasso, GraphicalBootstrap

bootstrap = GraphicalBootstrap(
    base_estimator=GraphicalLasso(alpha=0.05),
    n_bootstraps=500,
    random_state=0,
).fit(X)

fdr_out = bootstrap.fdr_threshold(fdr=0.05)
# fdr_out["reject"] is a (p, p) bool mask of edges below 5% FDR.
# fdr_out["adjusted_pvalues"] is the (p, p) BH-adjusted p-matrix.
print(f"{fdr_out['reject'].sum() // 2} edges retained at 5% FDR")
```

## Family-Wise Error Rate (Bonferroni, Holm)

When even a single false edge is unacceptable — small-`p` clinical
networks, regulatory filings — use FWER instead. Holm is uniformly
more powerful than Bonferroni at the same `α`:

$$
\hat{p}^{(\text{Holm})}_{(r)} = \max_{s \le r} (m - s + 1) \cdot \hat{p}_{(s)}.
$$

```python
fwer_out = bootstrap.fwer_threshold(fwer=0.05, method="holm")
```

## Meinshausen–Bühlmann stability bound

If you're using {class}`~skein_glm.GraphicalStabilitySelection`
instead of {class}`~skein_glm.GraphicalBootstrap`, the MB (2010)
closed-form bound

$$
\pi_{thr} = \frac{1}{2} + \frac{q_\Lambda^2}{2 \cdot p_{total} \cdot E[V]}
$$

inverts the stability-selection threshold to give an expected-false-
positive guarantee. `q_Λ` is the average number of edges activated
per `(λ, bootstrap)` cell; `p_total = p(p − 1)/2` is the edge family
size; `E[V]` is the desired upper bound on the number of falsely-
stable edges.

```python
from skein_glm import GraphicalStabilitySelection

stab = GraphicalStabilitySelection(
    base_estimator=GraphicalLasso(alpha=0.05),
    lambdas=[0.05, 0.10, 0.20, 0.40],
    n_bootstraps=100,
).fit(X)

# What threshold gives E[V] ≤ 1?
pi_required = stab.mb_threshold(expected_false_positives=1.0)
print(f"need π_thr ≥ {pi_required:.3f} to control E[V] ≤ 1")
```

If the requested error bound is infeasible at the observed `q_Λ`
(threshold would have to exceed 1), the function raises
{class}`ValueError` with a diagnostic — typically meaning either the
λ-grid is too loose (too many edges activate) or the desired `E[V]`
is unachievable on this graph at this sample size.

## Joint (multi-population) estimators

For joint graphical models — {class}`~skein_glm.JointGraphicalLasso`
and {class}`~skein_glm.JointGraphicalMCP` — the bootstrap produces a
`(B, K, p, p)` stack of `K` per-population precision matrices.
{func}`~skein_glm.edge_fdr_threshold` and
{func}`~skein_glm.edge_fwer_threshold` accept that shape transparently
and pool all `K · p(p − 1)/2` edge hypotheses into a *single*
multiple-testing family — the standard treatment for joint
estimators where the same edges are estimated across related
populations. The output `reject` mask has shape `(K, p, p)` so you
can read off which edges are retained per population at the global
error rate.

## When *not* to use FDR / FWER on edges

- **Exploratory analysis.** Edge bootstrap diagrams (the bootnet
  default) are still the right output for showing where the
  uncertainty lives. FDR / FWER answer a different, narrower
  question.
- **Very small graphs** (`p ≤ 5`). The family-size correction
  doesn't bite hard enough at low `p` to add much over the raw
  p-value; you'd report the raw bootstrap CIs.
- **Highly dependent edges.** BH controls FDR under PRDS, and
  bootstrap dependence within `Θ̂` is usually well-behaved, but
  pathological cases (e.g. block-equicorrelated underlying
  precisions) can violate it. Holm FWER is robust under arbitrary
  dependence and is the safer choice if you're worried.

## References

- **Benjamini, Y., & Hochberg, Y.** (1995). "Controlling the false
  discovery rate: a practical and powerful approach to multiple
  testing." *J. R. Stat. Soc. B* 57(1): 289–300.
- **Meinshausen, N., & Bühlmann, P.** (2010). "Stability selection."
  *J. R. Stat. Soc. B* 72(4): 417–473, Theorem 1.
- **Liu, W.** (2013). "Gaussian graphical model estimation with false
  discovery rate control." *Annals of Statistics* 41(6): 2948–2978.
- **van Borkulo, C. D. et al.** (2017). "Network analysis of
  multivariate data in psychological science." *Nature Reviews
  Methods Primers* 2:60.
