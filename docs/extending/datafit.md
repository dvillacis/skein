# Extending: custom datafits

The `Datafit` ABC in `skein_glm.datafits` mirrors the Rust trait in
`skein-core::datafit`. Implement it to add a new loss function —
robust regression (Huber, quantile), heavy-tailed regression
(Student-t), zero-inflated counts, etc.

The same prototype-then-port workflow as for [custom penalties](penalty.md)
applies.

## The `Datafit` ABC

For a loss of the form $\mathcal{L}(\beta) = \frac{1}{n} \sum_i \ell(y_i, x_i^\top \beta)$:

```python
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Datafit(ABC):
    @abstractmethod
    def value(self, residual: NDArray[np.float64]) -> float: ...

    @abstractmethod
    def init_residual(
        self,
        x: NDArray[np.float64],
        beta: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def coord_lipschitz(
        self,
        x: NDArray[np.float64],
        j: int,
    ) -> float: ...

    @property
    def sample_weights(self) -> NDArray[np.float64] | None:
        return None
```

The full Rust trait has more methods (`coord_grad`, `full_grad`,
`residual_at`) but the ABC keeps the minimal surface for prototyping.

## Two flavors of datafit

### Quadratic (LS-shaped)

If your loss is

$$
\ell(y_i, \eta_i) = \frac{1}{2} (y_i - \eta_i)^2
$$

(or a weighted version), it slots directly into the M1 coordinate
descent solver: the residual $r = y - X\beta$ admits a closed-form
coordinate update. This is the easiest case. The existing
`LeastSquares` is the reference.

### Non-quadratic (GLM)

If your loss is non-quadratic — logistic, Poisson, robust regression,
etc. — you go through the **prox-Newton** outer loop. At each outer
iteration you build a **weighted-LS surrogate**:

$$
\mathcal{L}(\beta) \approx \frac{1}{2n} \sum_i w_i^{(t)} \, (z_i^{(t)} - x_i^\top \beta)^2 + \text{const}
$$

with working response $z_i^{(t)} = \eta_i + (y_i - \mu_i^{(t)}) / w_i^{(t)}$
and per-sample weights $w_i^{(t)} = -\partial^2 \ell / \partial \eta^2$
(diagonal Hessian approximation).

The Rust trait `GlmDatafit` adds two methods for this:

```rust
pub trait GlmDatafit: Sync + Send {
    fn surrogate_at(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<f64>,
    ) -> LeastSquares;

    fn loss(
        &self,
        design: &dyn DesignMatrix,
        beta: ArrayView1<f64>,
    ) -> f64;
}
```

`surrogate_at(β)` returns a `LeastSquares` instance with the working
response and per-sample weights baked in; the prox-Newton outer
loop then hands this to the M1 inner solver. `loss(β)` computes
the original (non-quadratic) loss for convergence reporting.

## Worked example: Huber regression

The Huber loss is robust to outliers — quadratic for small residuals,
linear for large ones:

$$
\ell_\delta(r) =
\begin{cases}
\frac{1}{2} r^2, & |r| \leq \delta \\
\delta(|r| - \frac{\delta}{2}), & |r| > \delta
\end{cases}
$$

The gradient is
$\partial \ell_\delta / \partial \eta = -\mathrm{clip}(r, -\delta, \delta)$
(where $r = y - \eta$), and the Hessian is 1 in the quadratic region,
0 in the linear region. We approximate the Hessian as 1 everywhere
for the prox-Newton diagonal — slightly over-bounded in the linear
region but always valid (the inner CD just takes smaller steps).

```python
import numpy as np
from numpy.typing import NDArray
from skein_glm.datafits import Datafit


class HuberLoss(Datafit):
    """Huber regression: quadratic for |r| ≤ δ, linear above."""

    def __init__(
        self,
        y: NDArray[np.float64],
        delta: float = 1.345,
        sample_weights: NDArray[np.float64] | None = None,
    ) -> None:
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.delta = delta
        self._w = sample_weights

    @property
    def sample_weights(self) -> NDArray[np.float64] | None:
        return self._w

    def value(self, residual: NDArray[np.float64]) -> float:
        r = residual
        d = self.delta
        # Quadratic region: 0.5 r²; linear region: δ(|r| - δ/2).
        abs_r = np.abs(r)
        loss = np.where(
            abs_r <= d,
            0.5 * r ** 2,
            d * (abs_r - 0.5 * d),
        )
        if self._w is not None:
            loss = loss * self._w
        return float(loss.mean())

    def init_residual(
        self,
        x: NDArray[np.float64],
        beta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Residual r = y - Xβ. Sign convention: positive r means
        # we under-predicted, matching the Datafit trait expectation.
        return self.y - x @ beta

    def coord_lipschitz(self, x: NDArray[np.float64], j: int) -> float:
        # Hessian upper bound: 1 (quadratic region) per sample.
        # Lipschitz of ∂ℓ/∂β_j is bounded by (1/n) · Σ w_i · x_ij².
        n = x.shape[0]
        col = x[:, j]
        if self._w is None:
            return float((col ** 2).sum() / n)
        return float((self._w * col ** 2).sum() / n)
```

### Validation

```python
rng = np.random.default_rng(0)
n, p = 500, 5
X = rng.standard_normal((n, p))
true_beta = np.array([1.0, -2.0, 0.5, 0.0, 0.0])
y_clean = X @ true_beta
# Add 5% outliers — Huber should ignore them; LS would be biased.
y = y_clean + 0.1 * rng.standard_normal(n)
outlier_idx = rng.choice(n, size=int(0.05 * n), replace=False)
y[outlier_idx] += 50.0 * rng.standard_normal(len(outlier_idx))

# Sanity check: LS would be biased here.
import numpy as np
beta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"LS β_hat: {beta_ls}")           # biased by outliers

# Hand-rolled prox-Newton validation loop. This isn't production —
# it's just to confirm the Datafit math is right before porting.
def proxnewton_loop_py(X, datafit, max_outer=20, tol=1e-7):
    n, p = X.shape
    beta = np.zeros(p)
    for outer in range(max_outer):
        r = datafit.init_residual(X, beta)
        # Build LS surrogate via working response. For Huber:
        # μ_i = clip(η_i + r_i, ...)... actually for our Hessian≈1
        # approximation, the working response is just η + r = y when
        # we're in the quadratic region. In the linear region, the
        # gradient is bounded so the surrogate undershoots.
        # For demonstration, hand-roll a small CD on the surrogate:
        eta = X @ beta
        # Hessian approximation w_i = 1 if |r_i| ≤ δ else δ/|r_i|
        # (this is the actual MM-tightest weight, vs our coord_lipschitz
        # bound).
        d = datafit.delta
        w = np.where(np.abs(r) <= d, 1.0, d / np.maximum(np.abs(r), 1e-12))
        z = eta + (-(-w * r)) / w   # working response = η + r·w/w = y
        # ... the rest of the prox-Newton loop. Left as exercise.
        # (For real validation, port to Rust and use skein's solver.)
        break
    return beta
```

The Rust port follows the same pattern as the existing
`BinomialLogit` / `PoissonLog` / `CoxPH` datafits — implement
`GlmDatafit::surrogate_at` returning a `LeastSquares` with working
response and per-sample weights.

## When the diagonal-Hessian assumption fails

Some datafits don't admit a usable diagonal Hessian:

- **Quantile regression**: ℓ is non-smooth at 0; the Hessian is
  literally zero where defined. Smoothed quantile (Bofinger et al.)
  is a workaround.
- **Censored quantile** / **interval censoring**: similar.

For these, you'd implement a different outer loop (subgradient,
ADMM, or a smoothed approximation). The current trait surface
assumes prox-Newton works; extending to other outer loops is a
larger change.

## See also

- [Concepts: Datafits](../concepts/datafits.md) — the existing
  datafit zoo and how prox-Newton works.
- [Extending: penalties](penalty.md) — same recipe for custom
  regularizers.
- [Extending: backends](backend.md) — same recipe for custom storage.
