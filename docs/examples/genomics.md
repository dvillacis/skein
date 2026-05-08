# Worked example: genomics — sparse-group MCP on SNP data

A representative pattern: predict a continuous phenotype from
genome-wide SNP genotypes, where SNPs cluster into genes and we
want both gene-level and SNP-level sparsity. This is exactly what
**sparse-group MCP** is for — gene-level group structure plus
within-gene scalar selection — and the nonconvexity gives us
unbiased estimates of the genuinely active SNPs.

The example uses a synthetic problem (no real genomic data) so it
runs in seconds; the patterns scale directly to real GWAS-shaped
data when you swap in `MmapDesignF64` for the design.

## Setup

```python
import numpy as np
import skein_glm

rng = np.random.default_rng(42)

# 1000 individuals, 200 SNPs grouped into 20 genes of 10 SNPs each.
n = 1000
n_genes = 20
snps_per_gene = 10
p = n_genes * snps_per_gene
groups = np.repeat(np.arange(n_genes), snps_per_gene)

# Simulate genotypes: SNP frequencies in [0.1, 0.5], coded {0, 1, 2}.
mafs = rng.uniform(0.1, 0.5, size=p)
X = rng.binomial(2, mafs, size=(n, p)).astype(float)

# Simulate phenotype: only 3 genes are causal, with sparse within-gene
# effects (2-3 active SNPs per causal gene, others null).
true_beta = np.zeros(p)
causal_genes = [3, 7, 14]
for g in causal_genes:
    snp_indices = np.where(groups == g)[0]
    # Pick 2-3 SNPs in this gene to be active.
    n_active = rng.integers(2, 4)
    active = rng.choice(snp_indices, size=n_active, replace=False)
    true_beta[active] = rng.normal(0.5, 0.2, size=n_active)

y = X @ true_beta + 0.5 * rng.standard_normal(n)

print(f"n = {n}, p = {p}, n_genes = {n_genes}")
print(f"causal genes: {causal_genes}")
print(f"# truly active SNPs: {(true_beta != 0).sum()}")
```

## The fit

```python
# Group sizes are uniform here (10 SNPs each), but in real data
# you'd want the sqrt-size correction.
group_sizes = np.bincount(groups)
group_weights = np.sqrt(group_sizes.astype(float))

# Sparse-group MCP via LLA. α = 0.5 balances within-gene L1 and
# across-gene L2; γ = 3 is moderate nonconvexity.
fit = skein_glm.SparseGroupMCPPathCV(
    groups=groups,
    weights=group_weights,
    gamma=3.0,
    alpha=0.5,
    n_lambdas=40,
    lambda_min_ratio=1e-3,
    cv=10,
    standardize=True,
    random_state=0,
).fit(X, y)

print(f"Best λ: {fit.lambda_best_:.4f}")
print(f"# selected SNPs: {(fit.coef_ != 0).sum()}")
```

## Inspect what was selected

```python
# Which genes have any active SNP?
selected_genes = sorted(set(int(g) for g in groups[fit.coef_ != 0]))
print(f"Selected genes: {selected_genes}")
print(f"Causal genes:   {causal_genes}")

# Per-gene: # truly active vs # selected SNPs.
for g in causal_genes:
    snp_idx = np.where(groups == g)[0]
    n_true_active = (true_beta[snp_idx] != 0).sum()
    n_selected = (fit.coef_[snp_idx] != 0).sum()
    print(f"  Gene {g:2d}: {n_true_active} truly active, {n_selected} selected")
```

Expected output (depends on RNG, but qualitatively):

```text
Best λ: 0.0287
# selected SNPs: 7
Selected genes: [3, 7, 14]
Causal genes:   [3, 7, 14]
  Gene  3: 3 truly active, 3 selected
  Gene  7: 2 truly active, 2 selected
  Gene 14: 2 truly active, 2 selected
```

The sparse-group MCP recovers the right gene structure (every
causal gene has at least one selected SNP, no false-positive genes)
and within those genes selects close to the right SNP count.
Nonconvex shrinkage means the recovered coefficients aren't as
biased as a sparse-group lasso fit would be:

```python
# Compare to convex sparse-group lasso (no nonconvex shrinkage).
fit_lasso = skein_glm.SparseGroupLassoPathCV(
    groups=groups,
    weights=group_weights,
    alpha=0.5,
    n_lambdas=40,
    lambda_min_ratio=1e-3,
    cv=10,
    standardize=True,
    random_state=0,
).fit(X, y)

# MCP β tends to be closer to truth on active features.
def active_bias(beta_hat, beta_true):
    """Mean |β_hat - β_true| over truly active features."""
    active = beta_true != 0
    return float(np.abs(beta_hat[active] - beta_true[active]).mean())

print(f"Sparse-group MCP active-set bias:  {active_bias(fit.coef_, true_beta):.4f}")
print(f"Sparse-group lasso active-set bias: {active_bias(fit_lasso.coef_, true_beta):.4f}")
```

The MCP bias is meaningfully lower because the penalty flattens for
$|\beta_j| > \gamma\lambda$ — large coefficients aren't shrunk
toward zero.

## Scaling to GWAS-sized data

Real GWAS has `n` in the tens of thousands and `p` in the
hundreds of thousands or millions. The fit scales naturally:

1. **Write the genotype matrix to disk** in column-major f32
   (half the disk footprint vs f64; truncation error is irrelevant
   for the {0, 1, 2} encoding):

   ```python
   # gen.bin = column-major f32 dump of the (n, p) genotype matrix.
   buf = np.ascontiguousarray(X, dtype=np.float32).tobytes(order="F")
   with open("gen.bin", "wb") as f:
       f.write(buf)
   ```

2. **Open it as `MmapDesignF32`** and fit:

   ```python
   design = skein_glm.MmapDesignF32("gen.bin", n_rows=n, n_cols=p)
   fit = skein_glm.MCPPathCV(gamma=3.0, n_lambdas=50, cv=10).fit(design, y)
   ```

   v0.1 surface for genotype-sized fits: `MCPPathCV` and
   `LogisticMCPPathCV` (case/control phenotypes) are wired through
   for mmap; the group-MCP mmap path is mechanical to add and is
   the headline M4.x extension.

3. **For chromosome-by-chromosome chunking**, use
   `ChunkedDesignF64` / `ChunkedDesignF32` with one file per chrom.

## See also

- [Concepts: Penalties](../concepts/penalties.md) — sparse-group
  MCP details and the LLA outer loop.
- [Concepts: Backends](../concepts/backends.md) — mmap and chunked
  design for very large `n × p`.
- [Quick start](../quickstart.md) — the basic fitting recipes.
