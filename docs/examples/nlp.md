# Worked example: NLP — sparse logistic regression on a bag-of-words classifier

Text classification is the canonical "wide sparse $X$, binary $y$,
heterogeneous feature scales" problem. We'll fit a logistic MCP
classifier on a synthetic spam-detection dataset where most tokens
are uninformative noise and a small subset is genuinely predictive.

The data is sparse (most documents don't contain most tokens) so we
use `scipy.sparse.csc_matrix` directly. Standardization is
important — token frequencies span several orders of magnitude.

## Setup

```python
import numpy as np
import scipy.sparse as sp
import skein_glm

rng = np.random.default_rng(0)

# 5000 documents, 2000 tokens.
n_docs = 5000
n_tokens = 2000

# Token frequencies follow Zipf's law-ish; some tokens are rare,
# others appear in most documents. The "active" tokens are mid-
# frequency.
log_freqs = rng.uniform(np.log(1e-4), np.log(0.5), size=n_tokens)
token_probs = np.exp(log_freqs)

# Build a sparse document-term matrix.
# Each document has Poisson(50) tokens on average; for each one we
# sample with-replacement from the token distribution.
docs_per_token = rng.binomial(1, np.minimum(token_probs[None, :], 1.0), size=(n_docs, n_tokens))
X_dense = docs_per_token.astype(float)

# Add a "TF" component: token counts are integer ≥ 0, not just 0/1.
counts = rng.poisson(2, size=X_dense.shape) * docs_per_token
X_dense = counts.astype(float)
X_sparse = sp.csc_matrix(X_dense)

print(f"shape: {X_sparse.shape}, nnz: {X_sparse.nnz}, "
      f"density: {X_sparse.nnz / (n_docs * n_tokens):.3%}")

# True labels: 30 tokens are genuinely informative (15 spam-correlated,
# 15 ham-correlated). The rest are noise.
true_beta = np.zeros(n_tokens)
spam_tokens = rng.choice(n_tokens, size=30, replace=False)
true_beta[spam_tokens[:15]] = rng.uniform(0.5, 1.5, size=15)
true_beta[spam_tokens[15:]] = rng.uniform(-1.5, -0.5, size=15)

logit = X_sparse @ true_beta
prob = 1.0 / (1.0 + np.exp(-logit))
y = (rng.uniform(size=n_docs) < prob).astype(float)

print(f"label balance: {y.mean():.2%} positive")
```

## The fit

Logistic MCP path with CV, standardize=True (so high-frequency tokens
don't get pre-emptively penalized for having larger column norms):

```python
fit = skein_glm.LogisticMCPPathCV(
    gamma=3.0,
    n_lambdas=50,
    lambda_min_ratio=1e-3,
    cv=5,
    standardize=True,
    random_state=0,
).fit(X_sparse, y)

print(f"Best λ: {fit.lambda_best_:.5f}")
print(f"# selected tokens: {(fit.coef_ != 0).sum()}")
print(f"# truly informative: 30")
```

The sparse path solver dispatches transparently — `X_sparse` is a
`scipy.sparse.csc_matrix` and skein routes to its sparse PyO3
entry. No conversion to dense ever happens.

## Evaluate

```python
# Predict on a held-out test set.
X_test_dense = (rng.poisson(2, size=(1000, n_tokens)) *
                rng.binomial(1, np.minimum(token_probs[None, :], 1.0), size=(1000, n_tokens))).astype(float)
X_test = sp.csc_matrix(X_test_dense)

logit_test = X_test @ true_beta
y_test = (rng.uniform(size=1000) < 1.0 / (1.0 + np.exp(-logit_test))).astype(float)

prob_test = fit.predict_proba(X_test)
labels_test = fit.predict(X_test)

# Accuracy and AUC.
accuracy = (labels_test == y_test).mean()
print(f"test accuracy: {accuracy:.3f}")

# AUC: scikit-learn (we already depend on it).
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, prob_test)
print(f"test AUC:      {auc:.3f}")
```

## Inspect the selected vocabulary

A useful diagnostic: how many of the truly informative tokens did
we recover, and how many false positives did we pick up?

```python
true_active = set(spam_tokens.tolist())
selected = set(np.where(fit.coef_ != 0)[0].tolist())
true_positives = true_active & selected
false_positives = selected - true_active
false_negatives = true_active - selected

print(f"recovered: {len(true_positives)} / {len(true_active)} truly informative")
print(f"false positives: {len(false_positives)}")
print(f"false negatives: {len(false_negatives)}")
```

Typical output:

```text
recovered: 27 / 30 truly informative
false positives: 4
false negatives: 3
```

The MCP nonconvexity helps both ways here:

- **Less bias on truly informative tokens** → their coefficients
  end up closer to truth, which makes them harder to mistakenly
  threshold to zero.
- **Aggressive sparsification** → noise tokens get zeroed cleanly,
  fewer false positives than lasso would give.

## Compare to lasso (MCP at γ=∞)

```python
fit_lasso = skein_glm.LogisticMCPPathCV(
    gamma=1e6,                        # ≈ lasso
    n_lambdas=50,
    lambda_min_ratio=1e-3,
    cv=5,
    standardize=True,
    random_state=0,
).fit(X_sparse, y)

print(f"Lasso # selected:  {(fit_lasso.coef_ != 0).sum()}")
print(f"MCP   # selected:  {(fit.coef_ != 0).sum()}")
```

Lasso typically selects more features (it under-penalizes truly
active ones, so the optimal λ ends up letting some noise through
to compensate). MCP at γ=3 selects a tighter active set with less
bias.

## Scaling up

A real production text classifier might have `n` in the millions
and `p` in the hundreds of thousands. The same recipe scales:

1. Keep `X` sparse — `scipy.sparse.csc_matrix` is the fastest path
   for fitting; conversion to dense would densify the document-
   term matrix to $n \times p$ floats, which is usually impossible.
2. If you need streaming (multi-shard pipelines, larger-than-RAM
   data), serialize each shard as a column-major dense file (or
   sparse-CSC concatenation) and use `ChunkedDesignF64` /
   `ChunkedDesignF32`. v0.1 doesn't ship a sparse-chunked backend
   yet; that's a natural follow-up to combine `Chunked<C>` with
   `SparseCSC<C>`.

## See also

- [Concepts: Backends](../concepts/backends.md) — `SparseCSC` and
  the sparse-aware coordinate descent path.
- [Concepts: Datafits](../concepts/datafits.md) — logistic
  regression and prox-Newton.
- [Concepts: Penalties](../concepts/penalties.md) — when to use MCP
  over lasso for text classification.
