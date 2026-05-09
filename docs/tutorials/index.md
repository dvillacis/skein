# Tutorials

A guided tour of skein in three tiers. Each tutorial is a focused
walkthrough — read in order, or skip to the tier that matches what
you already know.

## Basics

Start here if you've never used skein. Three short tutorials that
together get you to "fitting and interpreting a sparse model."

1. **[Your first fit](01_first_fit.md)** — `MCPRegressor` on a noisy
   regression. The mental model: `fit` / `coef_` / `predict` / `score`.
2. **[Picking λ](02_picking_lambda.md)** — paths, K-fold CV, and
   information criteria. Three principled options for the
   regularization-strength choice.
3. **[Logistic and Cox](03_logistic_and_cox.md)** — same workflow,
   different datafit. Demonstrates the `(datafit, penalty)`
   orthogonality that drives the rest of the library.

## Working with structure

Real data has structure: features cluster into groups, design
matrices are sparse, columns have different scales, counts come with
exposure. Three tutorials covering the most common practical
extensions.

4. **[Group penalties](04_group_penalties.md)** — when whole
   groups of features should be selected together (genes, dummies,
   bands). `GroupLasso`, `GroupMCP`, `SparseGroupMCP` —
   when to use which.
5. **[Sparse and standardize](05_sparse_and_standardize.md)** —
   scipy.sparse CSC input, per-column standardization, and the dense
   ↔ sparse equivalence. Plus per-feature weights for soft
   constraints.
6. **[Counts and rates](06_counts_and_rates.md)** — Poisson
   regression with log-exposure offsets. Rate ratios, predicting
   expected counts, and the ubiquitous epidemiology / click-through
   / insurance pattern.

## Advanced

The features that differentiate skein from glmnet / skglm / grpreg.
Each tutorial covers a method that closes a parity gap with R or
adds something none of the alternatives have.

7. **[Stability selection](07_stability_selection.md)** —
   bootstrap-based feature selection without picking a single λ. The
   M5.x headline differentiator (no clean equivalent in glmnet,
   skglm, or grpreg).
8. **[Adaptive estimators](08_adaptive_estimators.md)** — the oracle
   property via two-stage refitting. The headline use of skein's
   per-feature-weights axis. 30 adaptive classes covering scalar,
   group, and GLM datafits.
9. **[Multinomial and multi-task](09_multinomial_and_multitask.md)** —
   K-class softmax classification and multi-response regression. Both
   reduce to the same row-grouped problem on a virtual block-
   replicated design.

## What's next after the tutorials

- **[Worked examples](../examples/genomics.md)** — full analyses on
  realistic synthetic data (genomics SNP-style, NLP text
  classification, survival).
- **[Concepts](../concepts/index.md)** — the conceptual model behind
  the abstractions. The four orthogonal axes (penalty, datafit,
  weights, backend) and how they compose.
- **[Extending](../extending/penalty.md)** — building your own
  penalty, datafit, or design backend. Both Python ABCs and Rust
  trait surfaces.
- **[API reference](../api/index.md)** — the complete surface,
  organized by family.
