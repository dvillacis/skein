# Datafits

The datafit term is the loss function: how the model is graded
against the training data. `skein` ships four datafit families
covering the canonical generalized linear models — Gaussian
(least squares), binomial (logistic), Poisson, and Cox proportional
hazards — plus a clean trait surface for adding more.

## The `Datafit` trait

In Rust:

```rust
pub trait Datafit: Sync + Send {
    fn n_samples(&self) -> usize;
    fn value(&self, design: &dyn DesignMatrix, beta: ArrayView1<f64>) -> f64;
    fn coord_grad(&self, design: &dyn DesignMatrix, j: usize, residual: ArrayView1<f64>) -> f64;
    fn coord_lipschitz(&self, design: &dyn DesignMatrix, j: usize) -> f64;
    fn full_grad(&self, design: &dyn DesignMatrix, residual: ArrayView1<f64>) -> Array1<f64>;
    fn sample_weights(&self) -> Option<ArrayView1<f64>>;
    fn residual_at(&self, design: &dyn DesignMatrix, beta: ArrayView1<f64>) -> Array1<f64>;
}
```

In Python (`skein_glm.datafits.Datafit` ABC, mirrors the Rust trait for
prototyping new datafits without porting to Rust first):

```python
from skein_glm.datafits import Datafit
class MyDatafit(Datafit):
    def value(self, design, beta): ...
    def coord_grad(self, design, j, residual): ...
    # etc.
```

## Least squares (Gaussian)

The simplest datafit:

$$
\mathcal{L}(\beta) \;=\; \frac{1}{2n} \sum_{i=1}^n w_i \, (y_i - x_i^\top \beta)^2
$$

with optional per-sample weights $w_i \geq 0$. This is what
`MCPRegressor`, `SCADRegressor`, `GroupLassoPathRegressor`, etc. use.

Coordinate descent on LS is fast: each coordinate update is closed-form
(prox of the soft-threshold operator on $\langle x_j, r \rangle / n$,
where $r = y - X\beta$ is the residual). The residual is patched
in place after each update — this is the core of the M1 production
CD solver.

```python
import skein_glm
model = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, y)
```

## Generalized linear models via prox-Newton

Logistic, Poisson, and Cox don't have a closed-form coordinate
update — they're not quadratic in $\beta$. `skein` handles them
with a **proximal-Newton** outer loop:

1. At iterate $\beta^{(t)}$, compute the linear predictor $\eta = X\beta^{(t)}$.
2. Build a **weighted least-squares surrogate** by linearizing the
   GLM gradient and approximating the Hessian as diagonal:
   - working response $z_i^{(t)} = \eta_i + (y_i - \mu_i^{(t)}) / w_i^{(t)}$
   - per-sample weights $w_i^{(t)} = \mu_i^{(t)}(1 - \mu_i^{(t)})$ (logistic),
     $w_i^{(t)} = \mu_i^{(t)}$ (Poisson), etc.
3. Solve the weighted-LS subproblem using the M1 CD solver (no code
   changes — the trait dispatch handles it).
4. Update $\eta$ from the new $\beta$, repeat. Typically converges
   in 5-15 outer iterations.

This means **every penalty available for LS works for every GLM**
via the same prox-Newton scaffolding, without per-GLM
re-implementation.

### Binomial logistic

$$
\mathcal{L}(\beta) \;=\; \frac{1}{n} \sum_i \big[ \log(1 + e^{\eta_i}) - y_i \eta_i \big],
\quad \eta_i = x_i^\top \beta + \alpha
$$

with $y_i \in \{0, 1\}$. Numerically stable cross-entropy via
`softplus(η)`. Per-sample weights honored throughout.

```python
clf = skein_glm.LogisticMCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, y_bin)
clf.predict(X)         # class labels {0, 1}, shape (n, n_lambdas)
clf.predict_proba(X)   # P(y=1), shape (n, n_lambdas)
clf.decision_function(X)  # η = Xβ + α, shape (n, n_lambdas)
```

### Poisson (log link)

$$
\mathcal{L}(\beta) \;=\; \frac{1}{n} \sum_i \big[ e^{\eta_i} - y_i \eta_i \big]
$$

with $y_i \geq 0$ (integer counts in practice, but the loss is
defined for any non-negative real). The solver clamps $\eta$ to
$[-30, 30]$ before exponentiation for numerical stability.

```python
model = skein_glm.PoissonMCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, y_count)
model.predict(X)          # μ = exp(η), the conditional mean
model.decision_function(X)  # η = log-rate
```

### Cox proportional hazards

The partial likelihood:

$$
\mathcal{L}(\beta) \;=\; -\frac{1}{n} \sum_{i: \delta_i = 1} \Big[ \eta_i - \log \sum_{j: t_j \geq t_i} e^{\eta_j} \Big]
$$

with $(t_i, \delta_i)$ being right-censored survival data:
$t_i$ is the observed time, $\delta_i \in \{0, 1\}$ is the event
indicator. **No intercept** — the baseline hazard absorbs it. Breslow
ties handling (events at the same observed time share the risk-set
sum); Efron is on the M3.7 roadmap.

The fit signature is `fit(X, time, event)` — three arguments instead
of two:

```python
cox = skein_glm.CoxMCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, time, event)
cox.predict(X)            # prognostic index η = Xβ; higher → shorter survival
cox.decision_function(X)  # same as predict() for Cox
```

## Standardization vs centering

For LS, the dense path uses glmnet's centering+scaling: $X$ is
centered to have zero column means, $y$ is centered to have zero
mean, and the intercept is recovered from $\alpha = \bar y - \bar x^\top \beta$.

For sparse and mmap LS, **centering would densify** $X$, so the
backend uses **scaling-only** plus a column-augmented intercept (a
1s column with penalty weight 0). The two parameterizations are
mathematically equivalent at convergence; the sparse path's β
matches the dense path's β at every λ to ~1e-7.

For GLMs (logistic, Poisson, Cox), centering $X$ is unnecessary
because the prox-Newton inner LS already absorbs the working
response's mean shift. All GLM paths use scaling-only +
column-augmented intercept, on every backend.

This matters when you pass `standardize=True` — `skein` picks the
right strategy automatically.

## Per-sample weights

Every datafit takes optional per-sample weights $w_i \geq 0$
through its constructor. Use cases:

- **Frequency weights**: aggregated data where row $i$ represents
  $w_i$ independent observations.
- **Inverse-propensity weights** for causal inference.
- **Importance weights** for class imbalance.
- **Survey weights** — the sampling-design correction.

The weights enter every solver-side dispatch (`coord_grad`,
`coord_lipschitz`, `full_grad`) — there's no path that silently
ignores them. See [Weights](weights.md) for the three-axis story.

## Choosing a datafit

A rough decision tree:

| Outcome type            | Datafit             |
|-------------------------|---------------------|
| Continuous              | `LeastSquares`      |
| Binary                  | `BinomialLogit`     |
| Count (rate or counts)  | `PoissonLog`        |
| Right-censored survival | `CoxPH` (Breslow)   |

For multinomial / softmax (more than 2 classes), see the M3.6
roadmap entry — currently deferred until multi-task infrastructure
(M7) lands. For Cox with Efron ties, weights, or offsets, see
M3.7.

## See also

- [Penalties](penalties.md) — the regularization term.
- [Weights](weights.md) — per-sample, per-feature, per-group axes.
- [Backends](backends.md) — the `DesignMatrix` types each datafit
  composes with.
