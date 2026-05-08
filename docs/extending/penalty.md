# Extending: custom penalties

The `Penalty` and `GroupPenalty` ABCs in `skein.penalties` mirror
the Rust traits in `skein-core::penalty`. The intended workflow:

1. **Prototype** your penalty in Python by implementing the ABC.
2. **Validate** the math against a reference (synthetic problem
   where the optimum is known, comparison against an existing
   implementation, etc.).
3. **Port to Rust** for production use, if performance demands.

This page walks through the prototype side. Porting to Rust is a
trait-implementation exercise once the math is validated.

## The `Penalty` ABC

For separable scalar penalties of the form

$$
P(\beta) = \sum_{j=1}^p w_j \, \psi(\beta_j)
$$

implement three methods:

```python
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Penalty(ABC):
    @abstractmethod
    def value(self, beta: NDArray[np.float64]) -> float: ...

    @abstractmethod
    def prox_coord(self, j: int, z: float, step: float) -> float: ...

    @property
    @abstractmethod
    def weights(self) -> NDArray[np.float64]: ...
```

`prox_coord(j, z, step)` solves

$$
\mathrm{prox}_{j}(z, \tau) \;=\; \arg\min_x \, \frac{1}{2\tau} (x - z)^2 + w_j \, \psi(x).
$$

This is what the coordinate descent inner loop calls.

## Worked example: hard-thresholded L1 (capped lasso)

The "capped" or "hard-thresholded" L1 penalty caps the absolute
value at a maximum, behaving like lasso below the cap and like
ridge above:

$$
\psi_{\text{capped}}(\beta_j; \tau) \;=\; \min(|\beta_j|, \tau)
$$

The proximal operator is well-known:

$$
\mathrm{prox}(z, \mathrm{step}) =
\begin{cases}
\mathrm{sign}(z) \cdot \max(|z| - \mathrm{step} \cdot w_j, 0), & |z| \leq \tau + \mathrm{step} \cdot w_j \\
z, & \text{otherwise}
\end{cases}
$$

```python
import numpy as np
from numpy.typing import NDArray
from skein.penalties import Penalty


class CappedL1(Penalty):
    """Hard-thresholded L1: ψ(β) = min(|β|, τ)."""

    def __init__(
        self,
        lambda_: float,
        tau: float,
        n_features: int,
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        self.lambda_ = lambda_
        self.tau = tau
        self._weights = (
            weights if weights is not None else np.ones(n_features)
        )

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights

    def value(self, beta: NDArray[np.float64]) -> float:
        return float(self.lambda_ * np.sum(
            self._weights * np.minimum(np.abs(beta), self.tau)
        ))

    def prox_coord(self, j: int, z: float, step: float) -> float:
        thresh = step * self.lambda_ * self._weights[j]
        if abs(z) <= self.tau + thresh:
            # Lasso-like region: soft threshold.
            return float(np.sign(z) * max(abs(z) - thresh, 0.0))
        # Above the cap: penalty is constant, so prox is the identity.
        return z
```

## Validating against synthetic data

Before porting to Rust, sanity-check on a problem where you know
the answer. The Python ABC isn't currently wired into the inner
solver (that's a trait-objects-via-FFI problem we deferred), so
validation goes through a hand-rolled small CD loop:

```python
def cd_loop_py(X, y, penalty, tol=1e-7, max_iter=10_000):
    """Minimal coordinate descent for testing a Python Penalty.

    For production use, port the penalty to Rust and use skein's
    real solver. This is for *validation* only.
    """
    n, p = X.shape
    beta = np.zeros(p)
    r = y.copy()                                # residual
    col_sq_norms = (X ** 2).sum(axis=0)
    for it in range(max_iter):
        max_change = 0.0
        for j in range(p):
            old = beta[j]
            # Patch residual back as if β_j = 0:
            r += old * X[:, j]
            z = X[:, j] @ r / col_sq_norms[j]
            step = n / col_sq_norms[j]
            new = penalty.prox_coord(j, z, step)
            beta[j] = new
            r -= new * X[:, j]
            max_change = max(max_change, abs(new - old))
        if max_change < tol:
            return beta, it + 1
    return beta, max_iter


# Test: capped L1 should recover sparse signal but not over-shrink it.
rng = np.random.default_rng(0)
n, p = 200, 50
X = rng.standard_normal((n, p))
true_beta = np.zeros(p)
true_beta[:3] = [3.0, -2.5, 1.5]            # large coefs above the cap
y = X @ true_beta + 0.1 * rng.standard_normal(n)

penalty = CappedL1(lambda_=0.1, tau=1.0, n_features=p)
beta_hat, iters = cd_loop_py(X, y, penalty)
print(f"converged in {iters} iters")
print(f"large coefs unbiased: {beta_hat[:3]}")   # should be near [3.0, -2.5, 1.5]
print(f"noise zeroed: {(np.abs(beta_hat[3:]) < 1e-3).sum()} / 47")
```

If this matches expectations, the math is right and the port to
Rust is mechanical.

## Porting to Rust

Once you've validated, the production version goes in Rust. The
trait surface to implement is in `skein-core::penalty`:

```rust
pub trait Penalty: Sync + Send {
    fn n_features(&self) -> usize;
    fn value(&self, beta: ArrayView1<f64>) -> f64;
    fn prox_coord(&self, j: usize, z: f64, step: f64) -> f64;
    fn weights(&self) -> ArrayView1<f64>;
}
```

A direct port of `CappedL1`:

```rust
use ndarray::{Array1, ArrayView1};
use skein_core::Penalty;

pub struct CappedL1 {
    lambda: f64,
    tau: f64,
    weights: Array1<f64>,
}

impl Penalty for CappedL1 {
    fn n_features(&self) -> usize {
        self.weights.len()
    }

    fn value(&self, beta: ArrayView1<f64>) -> f64 {
        self.lambda * beta
            .iter()
            .zip(self.weights.iter())
            .map(|(&b, &w)| w * b.abs().min(self.tau))
            .sum::<f64>()
    }

    fn prox_coord(&self, j: usize, z: f64, step: f64) -> f64 {
        let thresh = step * self.lambda * self.weights[j];
        if z.abs() <= self.tau + thresh {
            z.signum() * (z.abs() - thresh).max(0.0)
        } else {
            z
        }
    }

    fn weights(&self) -> ArrayView1<f64> {
        self.weights.view()
    }
}
```

Drop this in `crates/skein-core/src/penalty/capped_l1.rs`,
register in `penalty/mod.rs`, write a few tests against a
hand-computed reference. Then PyO3-wrap it for Python access.

Once it's in Rust, every existing solver path (LS scalar, LS group,
prox-Newton GLM scalar, etc.) works with your penalty without any
solver-side code changes. That's the trait-abstraction payoff.

## Group penalties

The `GroupPenalty` ABC is similar but with block-prox semantics:

```python
class GroupPenalty(ABC):
    @abstractmethod
    def value(
        self,
        beta: NDArray[np.float64],
        groups: list[NDArray[np.intp]],
    ) -> float: ...

    @abstractmethod
    def prox_group(
        self,
        g: int,
        block: NDArray[np.float64],
        step: float,
    ) -> NDArray[np.float64]: ...

    @property
    @abstractmethod
    def weights(self) -> NDArray[np.float64]: ...
```

`prox_group(g, block, step)` returns the solution to

$$
\arg\min_{u} \frac{1}{2\,\mathrm{step}} \|u - \mathrm{block}\|_2^2 + w_g \, \psi_g(u).
$$

For group lasso this is the well-known "group soft-threshold":

```python
def prox_group(self, g, block, step):
    norm = np.linalg.norm(block)
    thresh = step * self.lambda_ * self._weights[g]
    if norm <= thresh:
        return np.zeros_like(block)
    return block * (1.0 - thresh / norm)
```

For group MCP / SCAD, see `crates/skein-core/src/solver/lla.rs` for
the LLA outer-loop convention: you don't implement the nonconvex
group penalty's prox directly. Instead, you implement the convex
inner penalty (group lasso, weighted) and a `surrogate_weights_*`
helper that derives per-group weights from the current iterate.
The LLA path solver iterates this until convergence.

## When to port to Rust

| Situation                                             | Stay in Python | Port to Rust |
|-------------------------------------------------------|----------------|--------------|
| Prototyping a new penalty for a paper                 | ✅              |              |
| Solving 100s of small problems, one at a time         | ✅              |              |
| Solving one large problem you'll fit many times       |                | ✅            |
| The penalty is in your team's permanent toolkit       |                | ✅            |
| You want it in skein's main release                   |                | ✅            |

The Python ABC is not wired into the inner solver in v0.1 (the
trait-object FFI bridge is M5.x or later). For now, validation
loops like the one above let you iterate on math; production fits
go through Rust.

## See also

- [Concepts: Penalties](../concepts/penalties.md) — the existing
  penalty zoo and how LLA reduces nonconvex to convex.
- [Extending: datafits](datafit.md) — same recipe for custom losses.
- [Extending: backends](backend.md) — same recipe for custom storage.
