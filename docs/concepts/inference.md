# Inference on penalized fits

Penalized estimators (lasso, MCP, SCAD, group / sparse-group variants)
are biased by construction — that's the whole point. The bias buys
sparsity and variance reduction in finite samples, but it leaves the
analyst with a *point* estimate of `β̂` and no straightforward way
to attach a confidence interval, a p-value, or a hypothesis test to
any individual coefficient.

`skein` ships two complementary frameworks for closing that gap:

| Framework | When to use | Module |
|---|---|---|
| **Debiased lasso (VBR-style)** | Wald CIs + p-values per coefficient, asymptotic theory | {mod}`skein_glm.debiased` |
| **Stability selection (MB)** | Error-controlled support recovery, no distributional assumption | {mod}`skein_glm.stability` |

Both compose cleanly with every (datafit × penalty) combination on
the regression side; the graphical-models side has its own edge-level
analogs documented in [graph_inference](graph_inference).

## Debiased lasso for LS

The Van de Geer–Bühlmann–Ritov (2014) "desparsified" lasso adds a
single-step correction to ``β̂`` that removes the regularization bias
at the optimal rate:

$$
\hat{\beta}^{(d)} = \hat{\beta} + \frac{1}{n} \hat{\Theta} X^\top (y - X \hat{\beta}),
$$

where ``Θ̂`` is a nodewise-lasso construction of an approximate
inverse Gram ``≈ (XᵀX/n)⁻¹``. Under the standard high-dimensional
regularity conditions, ``√n · (β̂^(d) − β) ⇝ N(0, σ² · Θ̂ Σ̂ Θ̂ᵀ)``.

```python
from skein_glm import DebiasedLassoRegressor

deb = DebiasedLassoRegressor(alpha=0.05).fit(X, y)
print(deb.coef_)        # point estimate
print(deb.se_)          # standard errors
print(deb.ci_lower_)    # 95% Wald lower bound
print(deb.pvalues_)     # two-sided Wald p-values
```

## Debiased lasso for GLMs (logistic, Poisson)

The same construction extends to canonical-link GLMs through the
prox-Newton **weighted-LS surrogate**. At ``β̂``, the GLM's
per-sample Fisher information is the diagonal matrix
``W = diag(p_i(1 − p_i))`` for logistic or
``W = diag(μ_i)`` for Poisson; the construction proceeds on the
weighted design ``X̃ = W^{1/2} X``.

```python
from skein_glm import (
    DebiasedLogisticLassoRegressor,
    DebiasedPoissonLassoRegressor,
)

deb_logit = DebiasedLogisticLassoRegressor().fit(X, y_binary)
deb_pois  = DebiasedPoissonLassoRegressor().fit(X, y_counts)
```

The GLM variance formula is ``diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n²`` — no σ²
factor, since the GLM Fisher information already absorbs the noise
scale through ``W``.

## Debiased Cox PH

The Cox partial likelihood is not a regular GLM (it has no
multinomial-style score residual centered on a fitted mean), but the
*partial-likelihood Fisher information* still factors as
``X̃ᵀ X̃`` with ``W^{1/2}`` derived from the at-risk-set hazard
contributions. {func}`~skein_glm.debiased_cox_lasso` uses the same
surrogate-based construction as the GLM case:

1. Fit a penalized Cox lasso to obtain ``β̂``.
2. Extract the per-sample Cox Fisher diagonal
   ``w_i = exp(η̂_i)·Λ̂_0(t_i) − exp(2η̂_i)·Λ̂_0^{(2)}(t_i)``
   from the prox-Newton surrogate (via the Rust
   ``CoxPH::surrogate_at`` exposed as
   ``_core.cox_surrogate_weights_at``).
3. Build ``X̃ = W^{1/2} X``; nodewise-lasso ``Θ̂`` exactly as in
   logistic / Poisson.
4. Debias against the Cox score residual
   ``δ_i − exp(η̂_i)·Λ̂_0(t_i)``, where ``δ`` is the event indicator.
5. Variance: ``diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n²`` — the partial likelihood is
   self-normalizing, no σ² nuisance.

```python
from skein_glm import DebiasedCoxLassoRegressor

deb = DebiasedCoxLassoRegressor(alpha=0.05, ties="breslow").fit(X, time, event)
print(deb.coef_)        # debiased prognostic-feature weights
print(deb.pvalues_)     # one-sided H0: β_j = 0 → Wald p-value
print(deb.risk_score_)  # η̂ = X β̂ on the training rows
```

Cox has no intercept (the baseline hazard absorbs constants); the
``intercept_`` field on :class:`~skein_glm.DebiasedCoxLassoRegressor`
is a destandardization correction so that
``X @ coef_ + intercept_`` reproduces the prognostic index on the
original feature scale. The partial likelihood is invariant to that
shift.

## When *not* to use debiased lasso

- **`p > n`** with very weak signals across the board. The
  underlying penalized fit needs at least *some* recoverable
  support; the debiasing step only reaches its asymptotic regime
  once ``Θ̂`` is well-conditioned, which fails when no feature
  survives the penalty.
- **Highly correlated predictors.** The nodewise lassos work
  feature-by-feature; very high pairwise correlations inflate
  ``Θ̂`` entries and the resulting standard errors are
  conservative. Stability selection is more robust under
  collinearity.
- **Group-structured tests.** Debiased lasso gives per-coordinate
  inference. For group-level hypotheses (is the *gene* relevant?
  is the *region* relevant?), use the group-wise stability selection
  output, or a debiased *group* lasso construction — not on the
  roadmap yet.

## Stability selection

Stability selection complements debiased lasso. Instead of attaching
asymptotic CIs to coefficients, it subsamples the data
(Meinshausen–Bühlmann 2010), refits the penalized estimator on each
subsample across a λ-grid, and records the per-(feature, λ)
selection probability. The "stable set" — features whose maximum
selection probability across the λ-grid exceeds a threshold — has a
closed-form bound on its expected number of falsely-selected
features. It needs no distributional assumption and composes with
any penalized estimator.

See :class:`~skein_glm.StabilitySelection` and the
[stability selection
tutorial](../tutorials/07_stability_selection) for the regression
flavor; :class:`~skein_glm.GraphicalStabilitySelection` and
[graph_inference](graph_inference) for the edge-level analog.

## References

- **van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R.** (2014).
  "On asymptotically optimal confidence regions and tests for high-
  dimensional models." *Annals of Statistics* 42(3): 1166–1202.
  — The foundational VBR debiased construction.
- **Cai, T., & Wang, J.** (2017). "High-dimensional inference for
  covariance and precision matrices and applications to survival
  analysis." *J. Am. Stat. Assoc.* — Cox-PH extension.
- **Meinshausen, N., & Bühlmann, P.** (2010). "Stability selection."
  *J. R. Stat. Soc. B* 72(4): 417–473. — Stability selection.
