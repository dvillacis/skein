# Multinomial / softmax classification

K-class softmax logistic regression with row-grouped feature selection
across classes. The natural shape for genomics, document
classification, and any task where the support is expected to be
shared across labels.

## The model

For `K` classes with coefficient matrix `B ∈ ℝ^(p × K)`:

$$
\eta_{i, k} = X[i, :] \cdot B[:, k], \qquad
p_{i, k} = \frac{\exp(\eta_{i, k})}{\sum_{l=1}^K \exp(\eta_{i, l})}
$$

The penalized objective for the lasso case is

$$
\min_{B}\;\;
\frac{1}{n}\sum_{i=1}^{n}\!\bigl(\operatorname{logsumexp}(\eta_i) - \sum_{k=1}^K Y_{i, k}\,\eta_{i, k}\bigr)
\;+\;\lambda \sum_{j=1}^p w_j \, \|B[j, :]\|_2 .
$$

Reading off the penalty: `B[j, :]` is the row of feature `j` across
the K classes, and we penalize its L2 norm. If the row is exactly
zero, the feature is dropped from every class's logit
simultaneously — joint feature selection.

## Symmetric parameterization (no reference class)

skein follows glmnet's symmetric / "grouped" multinomial, in which all
K columns of B are estimated freely. The softmax has a redundant
degree of freedom (adding a constant to every column of `B` leaves
`p_{i,k}` unchanged), but the L2-row penalty breaks that redundancy by
shrinking all K coefficients toward zero. The unpenalized per-class
intercepts pick up the residual class frequencies independently.

Predictions are invariant to the softmax redundancy, so the symmetric
parameterization gives the same `predict_proba` as a reference-class
formulation would — just with cleaner row-grouped penalty geometry.
sklearn's `LogisticRegression(multi_class="multinomial")` uses a
similar overparameterized form by default; `glmnet(family="multinomial",
type.multinomial="grouped")` is the closest analog.

## How it works: stack-and-reuse

Multinomial logit reduces — through Böhning's diagonal Hessian
majorization — to a sequence of multi-task LS problems on the same
`MultiTaskDesign<X>` virtual design used by `multitask`. No new solver
code in skein.

### Step 1: Böhning bound

The exact softmax Hessian per sample is `diag(p_i) − p_i p_iᵀ`.
Böhning (1992) showed it's bounded above by a constant matrix:

$$
\operatorname{diag}(p_i) - p_i p_i^\top \;\preceq\; \tfrac{1}{2}\!\left(I - \tfrac{1}{K}\,\mathbf{1}\mathbf{1}^\top\right) .
$$

Equivalently, every per-(sample, class) Hessian diagonal is at most
`1/2`, regardless of the current `p_{i,k}`. This is the constant
glmnet uses (Friedman/Hastie/Tibshirani 2010, §4.4) — looser than the
true Hessian but cheap, robust, and decouples the K classes into
independent weighted-LS subproblems sharing the same X.

### Step 2: working response

With the constant `1/2` Hessian and gradient
`g_{i,k} = p_{i,k} − Y_{i,k}`, the prox-Newton step yields a per-class
working response

$$
z_{i, k} \;=\; \eta_{i, k} \;-\; 2\,(p_{i, k} - Y_{i, k}) .
$$

Stacking task-outer (`z̃[k·n + i] = z_{i,k}`), the local quadratic
surrogate becomes plain weighted multi-task LS on
`MultiTaskDesign<X>`:

$$
\frac{1}{2nK}\sum_{i,k}\;\tfrac{1}{2}\,\bigl(\tilde{X}\beta - \tilde{z}\bigr)_{ik}^2
\;+\;\lambda \sum_{j=1}^p w_j \, \|B[j, :]\|_2 .
$$

### Step 3: reuse M7's machinery

The wrapper that materializes the virtual `(nK × pK)` design lazily is
`MultiTaskDesign<D>`, the same wrapper M7's multi-task LS uses. Its
contiguous-block grouping `{jK, jK+1, …, jK+K-1}` per feature is
exactly the row-grouped penalty structure. The M2 block-CD inner
solver and the M2.3 LLA scheme for non-convex MCP/SCAD penalties
handle everything from there — no multinomial-specific code in the
solver.

Effectively: multinomial `=` multi-task LS surrogate with Böhning
weights, wrapped in a prox-Newton outer loop. Penalty zoo and design
backends (sparse, standardized, augmented) compose for free.

## Per-class intercepts

The intercept is handled by appending a single `1`s column to `X` at
position `j = p` and giving the resulting row-group weight `0` (so it
stays unpenalized). The `MultiTaskDesign` wrapper then replicates that
column into K virtual intercept columns (one per class), each
independently fit. Each per-class intercept comes out of `bvec[p·K + k]`
after the solve, the same convention multi-task uses for its per-task
intercepts.

## Label encoding

`fit(X, y)` accepts arbitrary 1D label arrays — integers, strings, or
anything `np.unique` can sort. The estimator runs `np.unique(y)` to
get `classes_` and converts to integer codes in `[0, K)` for the Rust
core. `predict(X)` decodes back to the original dtype, so:

```python
labels = np.array(["cat", "dog", "fish", "cat", ...])
clf = skein_glm.MultinomialLassoClassifier().fit(X, labels)
clf.classes_     # array(["cat", "dog", "fish"])
clf.predict(X)   # array(["cat", "fish", "dog", ...])
```

## Convention vs sklearn

skein uses the natural `(1/n) · Σ_i (logsumexp − Σ_k Y_{ik} η_{ik})`
data-fidelity factor — the per-sample multinomial NLL. sklearn's
`LogisticRegression` uses an `1/C` regularization parameterization
(C = 1/regularization), so a direct equivalence isn't a simple
constant rescaling — convert your `λ` ↔ `C` based on whichever side
you're starting from. The CV / IC paths in skein make `λ` selection
straightforward without needing the conversion.

The MCP shape `gamma` and SCAD shape `a` mean the same thing as in
the row-grouped MCP/SCAD families on the per-row L2 norm scale.

## See also

- [Multinomial estimators](../api/estimators-multinomial.md) —
  full API reference, all 12 classes.
- [Multi-task LS](multitask.md) — sister concept; same
  stack-and-reuse trick on a different datafit.
- [Penalties](penalties.md) — row-L2, row-MCP, row-SCAD,
  row-elastic-net penalty shapes.
