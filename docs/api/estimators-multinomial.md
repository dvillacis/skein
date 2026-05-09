# Multinomial classifiers

K-class softmax logistic regression with **row-grouped feature
selection across classes**. The coefficient matrix is `B ∈ ℝ^(p × K)`,
and the penalty acts on `B[j, :]` — feature `j` is either active for
every class or inactive for every class. This is the natural shape for
genomics, document classification, and any task where the support is
expected to be shared across labels.

12 sklearn-compatible classes total, four penalty families:

- **Lasso (convex)**: `MultinomialLassoClassifier`,
  `MultinomialLassoPathClassifier`, `MultinomialLassoPathCV`.
- **MCP (LLA-wrapped non-convex)**: `MultinomialMCPClassifier`,
  `MultinomialMCPPathClassifier`, `MultinomialMCPPathCV`.
- **SCAD (LLA-wrapped non-convex)**: `MultinomialSCADClassifier`,
  `MultinomialSCADPathClassifier`, `MultinomialSCADPathCV`.
- **Elastic net (convex)**: `MultinomialElasticNetClassifier`,
  `MultinomialElasticNetPathClassifier`, `MultinomialElasticNetPathCV`.

Naming uses the `Classifier` suffix per sklearn convention (matches
`LogisticRegression`); the binary-logistic family in skein keeps its
existing `LogisticMCPRegressor` naming for backward compatibility.

## Inputs and outputs

`fit(X, y)` takes `X ∈ ℝ^(n, p)` (dense ndarray or scipy.sparse CSC)
and a 1D `y` of length `n` containing class labels. Labels can be any
hashable / sortable dtype — integers, strings, or anything `np.unique`
handles; the estimator stores the sorted unique labels on `classes_`
and decodes predictions back to the original dtype.

After `fit`:

- `coef_` has shape `(K, p)` — matches sklearn's
  `LogisticRegression(multi_class="multinomial").coef_`.
- `intercept_` has shape `(K,)`.
- `classes_` has shape `(K,)`, dtype matching the original labels.
- `decision_function(X) → (n, K)` — η values (logits).
- `predict_proba(X) → (n, K)` — softmax of η, rows sum to 1.
- `predict(X) → (n,)` — argmax class labels in the original dtype.

Path classifiers (`*PathClassifier`) expose:

- `coefs_` of shape `(n_lambdas, K, p)`.
- `intercepts_` of shape `(n_lambdas, K)`.
- `lambdas_` of shape `(n_lambdas,)`.

Path-CV classifiers (`*PathCV`) refit on the full data at the
CV-best λ and expose the same `coef_` / `intercept_` / `classes_` as
the single-λ classifiers, plus `lambda_best_`, `cv_scores_`,
`cv_mean_scores_`, `cv_std_scores_`, `lambdas_`. Default splitter is
`StratifiedKFold` to keep heavy class imbalance from producing
class-empty train folds.

## Symmetric (no reference class) parameterization

skein's multinomial follows glmnet's symmetric parameterization: all
`K` columns of `B` are estimated with no reference class pegged to
zero. With penalization, the redundancy that softmax has under adding
a constant to every column of `B` is broken (the penalty shrinks all
classes toward zero). With unpenalized intercepts, the per-class
intercept is independently fit and predictions are invariant to the
softmax's symmetric degree of freedom.

This is what `glmnet(family="multinomial", type.multinomial="grouped")`
does; sklearn's default OvR multinomial differs.

## Optimization

Each prox-Newton outer iteration uses Böhning's diagonal majorization
of the softmax Hessian — `diag(p_i) − p_i p_iᵀ ⪯ (1/2) (I − 11ᵀ/K)` —
which simplifies to a constant per-(sample, class) Hessian diagonal of
`1/2`. The local quadratic surrogate is then a multi-task LS problem
on the same `MultiTaskDesign<X>` wrapper used by `multitask`, with
working response `z_{i,k} = η_{i,k} − 2 (p_{i,k} − Y_{i,k})` and
uniform weight `1/2`. The M2 block-CD machinery handles the inner
solve, and the row-grouped LLA scheme handles the non-convex MCP/SCAD
penalties — both completely unchanged from their single-output forms.

For the algebraic details and reduction proof, see
[Concepts → Multinomial](../concepts/multinomial.md).

## Sparse and standardize

Both work the same way they do for multi-task: dispatch on
`scipy.sparse.issparse(X)`, intercept handled by lazy
`Augmented<SparseCSC>` for sparse and physical column augmentation for
dense. `standardize=True` is supported for both backends — dense uses
glmnet-style scale-only standardization, sparse uses
`Standardized<Augmented<SparseCSC>>` lazily so column scaling never
densifies. Per-feature L1 weights are rescaled by `1/s_j` exactly as
the LS sparse-group standardize convention dictates.

The pytest suite has dense ↔ sparse equivalence tests on shared
λ-grids, both with and without standardize.

## Lasso — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multinomial.MultinomialLassoClassifier
   :members:

.. autoclass:: skein_glm.multinomial.MultinomialLassoPathClassifier
   :members:
```

## MCP — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multinomial.MultinomialMCPClassifier
   :members:

.. autoclass:: skein_glm.multinomial.MultinomialMCPPathClassifier
   :members:
```

## SCAD — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multinomial.MultinomialSCADClassifier
   :members:

.. autoclass:: skein_glm.multinomial.MultinomialSCADPathClassifier
   :members:
```

## Elastic net — single λ + path

```{eval-rst}
.. autoclass:: skein_glm.multinomial.MultinomialElasticNetClassifier
   :members:

.. autoclass:: skein_glm.multinomial.MultinomialElasticNetPathClassifier
   :members:
```
