"""Bootstrap-based edge stability for graphical models.

Two pure-Python wrappers around the M11 graphical estimators
(``GraphicalLasso``/``MCP``/``SCAD`` and ``JointGraphicalLasso``/
``MCP``):

* :class:`GraphicalStabilitySelection` — Meinshausen–Bühlmann (2010)
  stability selection lifted to edges. Subsample-bootstrap over a
  user-supplied λ-grid; record per-(λ, edge) selection probability;
  declare an edge **stable** if its max-over-λ probability crosses a
  threshold. Mirrors :class:`StabilitySelection` for coefficients.

* :class:`GraphicalBootstrap` — non-parametric (resample-with-
  replacement) bootstrap at a *single* λ. Returns the per-edge
  sampling distribution: mean, SD, quantile CIs, selection
  probability. This is what ``bootnet::bootnet(type="nonparametric")``
  in R returns and the headline output network-psychometrics
  practitioners use to draw edge-weight error bars.

Both wrappers work for single-population (single ``X``) and joint
(list of ``X^(k)`` across populations) graphical estimators; they
auto-detect by inspecting the base estimator for a ``lambda_2``
attribute (joint) vs. ``alpha`` (single).

Bootstrap requires *raw* data, not precomputed covariance: we have
no way to resample observations from a `(p, p)` covariance input.
Passing a square symmetric `X` raises ``ValueError``.

Reference
---------
van Borkulo et al. (2017). "Network analysis of multivariate data
in psychological science". *Nature Reviews Methods Primers* 2:60.
Implements the methodology behind R's `bootnet` package.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state


def _is_joint(est: Any) -> bool:
    """Joint estimators expose a ``lambda_2`` init param; single-pop
    expose ``alpha``. The two surfaces don't overlap."""
    return hasattr(est, "lambda_2")


def _looks_like_covariance(x: NDArray[np.float64]) -> bool:
    return x.ndim == 2 and x.shape[0] == x.shape[1] and bool(
        np.allclose(x, x.T, atol=1e-8)
    )


def _validate_raw_X(x: NDArray[np.float64], label: str) -> NDArray[np.float64]:
    xa = np.ascontiguousarray(x, dtype=np.float64)
    if xa.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {xa.shape}")
    if _looks_like_covariance(xa):
        raise ValueError(
            f"{label} looks like a precomputed covariance matrix "
            f"(square + symmetric, shape {xa.shape}); bootstrap "
            "requires raw observations. Pass the original (n, p) "
            "data array instead."
        )
    return xa


def _edge_mask(precision: NDArray[np.float64], eps: float) -> NDArray[np.bool_]:
    """Symmetric `(p, p)` boolean — `True` where `|Θ_ij| > eps`, off-
    diagonal only."""
    m = np.abs(precision) > eps
    np.fill_diagonal(m, False)
    return m


def _stable_pairs(max_prob: NDArray[np.float64], threshold: float) -> NDArray[np.int64]:
    """Convert a symmetric `(p, p)` probability matrix into an `(n, 2)`
    array of upper-triangular `(i, j)` pairs above ``threshold``."""
    p = max_prob.shape[0]
    iu_i, iu_j = np.triu_indices(p, k=1)
    sel = max_prob[iu_i, iu_j] >= threshold
    return np.stack([iu_i[sel], iu_j[sel]], axis=1).astype(np.int64)


# --- shared bootstrap workers -----------------------------------------


def _fit_stability_one(
    base_estimator: Any,
    X_or_Xs: Any,
    indices: Any,
    lambdas: NDArray[np.float64],
    alpha_name: str,
    is_joint: bool,
    active_eps: float,
) -> NDArray[np.bool_]:
    """One stability-selection bootstrap pass. Returns the boolean edge
    mask stack across λ:

    - single: `(n_lambdas, p, p)`
    - joint: `(n_lambdas, K, p, p)`
    """
    X_sub: Any
    if is_joint:
        X_sub = [
            np.ascontiguousarray(np.asarray(x)[idx], dtype=np.float64)
            for x, idx in zip(X_or_Xs, indices)
        ]
    else:
        X_sub = np.ascontiguousarray(np.asarray(X_or_Xs)[indices], dtype=np.float64)

    masks = []
    for lam in lambdas:
        est = clone(base_estimator).set_params(**{alpha_name: float(lam)})
        est.fit(X_sub)
        if is_joint:
            masks.append(
                np.stack([_edge_mask(t, active_eps) for t in est.precisions_], axis=0)
            )
        else:
            masks.append(_edge_mask(est.precision_, active_eps))
    return np.stack(masks, axis=0)


def _fit_bootstrap_one(
    base_estimator: Any,
    X_or_Xs: Any,
    indices: Any,
    is_joint: bool,
) -> NDArray[np.float64]:
    """One non-parametric bootstrap pass. Fits the estimator on the
    resampled data and returns the precision matrix (single) or a stack
    of precisions (joint)."""
    X_sub: Any
    if is_joint:
        X_sub = [
            np.ascontiguousarray(np.asarray(x)[idx], dtype=np.float64)
            for x, idx in zip(X_or_Xs, indices)
        ]
    else:
        X_sub = np.ascontiguousarray(np.asarray(X_or_Xs)[indices], dtype=np.float64)
    est = clone(base_estimator).fit(X_sub)
    if is_joint:
        return np.stack(est.precisions_, axis=0)
    return np.asarray(est.precision_, dtype=np.float64)


# --- stability selection ---------------------------------------------


class GraphicalStabilitySelection(BaseEstimator):
    """Meinshausen–Bühlmann stability selection for graphical models.

    Wraps a graphical estimator (single or joint) and runs
    subsample-bootstrap over a user-supplied λ-grid. Each bootstrap
    iteration draws a fraction of rows without replacement, refits at
    every λ in the grid, and records which off-diagonal entries of
    `Θ̂` are nonzero. The fraction of bootstraps in which edge
    ``(i, j)`` was selected at λ_k is its **selection probability**;
    an edge is **stable** if its max-over-λ probability crosses
    ``threshold``.

    The MB (2010) error-control bound on expected false positives
    applies to ``threshold > 0.5``.

    Parameters
    ----------
    base_estimator
        Any single-population (``GraphicalLasso``/``MCP``/``SCAD``) or
        joint (``JointGraphicalLasso``/``MCP``) graphical estimator.
        The init params (``gamma``/``a``/``edge_weights``/...) are
        passed through to every refit; only the regularisation
        parameter (``alpha`` or ``lambda_2``) is overridden per-λ.
    lambdas : array-like (n_lambdas,)
        Positive λ values to sweep. The order is preserved in the
        output but does not affect the result (each (bootstrap, λ)
        fit is independent).
    n_bootstraps : int, default 100
    sample_fraction : float, default 0.5
        Subsample fraction (without replacement). MB recommend 0.5.
    threshold : float, default 0.6
        Stable-edge probability cutoff; must be > 0.5 for the MB
        error-control bound.
    random_state : int or None
    n_jobs : int or None
        Forwarded to ``joblib.Parallel``. ``-1`` uses all cores.
    active_eps : float, default 1e-6
        Threshold for declaring `|Θ_ij|` nonzero. Looser than the
        sklearn default to absorb mild solver drift across subsample
        fits — graphical block-CD/ADMM converge to lower tolerance
        than coefficient solvers in practice.

    Attributes
    ----------
    selection_probabilities_ : ndarray
        Single-pop: shape `(n_lambdas, p, p)`. Joint: shape
        `(n_lambdas, K, p, p)`. Symmetric in the last two axes,
        zero on the diagonal.
    max_probabilities_ : ndarray
        Single-pop: shape `(p, p)`. Joint: shape `(K, p, p)`.
        Max over λ of the per-edge selection probability.
    stable_edges_ : ndarray of int64
        Single-pop: shape `(n_stable, 2)` of `(i, j)` pairs (upper
        triangular). Joint: list of `K` such arrays, one per
        population.
    lambdas_ : ndarray
    n_features_in_ : int
    n_populations_ : int (joint only)

    Notes
    -----
    Bootstrap requires raw observation data. Passing a precomputed
    covariance matrix is rejected with a clear error message.

    Examples
    --------
    >>> from skein_glm import (
    ...     GraphicalMCP, GraphicalStabilitySelection,
    ... )
    >>> import numpy as np
    >>> ss = GraphicalStabilitySelection(
    ...     base_estimator=GraphicalMCP(gamma=3.0),
    ...     lambdas=np.geomspace(0.5, 0.05, 8),
    ...     n_bootstraps=100, threshold=0.6, n_jobs=-1, random_state=0,
    ... )
    >>> ss.fit(X)
    >>> ss.stable_edges_      # shape (n_stable, 2)
    """

    selection_probabilities_: NDArray[np.float64]
    max_probabilities_: NDArray[np.float64]
    stable_edges_: Any  # ndarray for single, list[ndarray] for joint
    lambdas_: NDArray[np.float64]
    n_features_in_: int

    def __init__(
        self,
        base_estimator: Any,
        lambdas: NDArray[np.float64],
        *,
        n_bootstraps: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        random_state: int | None = None,
        n_jobs: int | None = None,
        active_eps: float = 1e-6,
    ) -> None:
        self.base_estimator = base_estimator
        self.lambdas = lambdas
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.active_eps = active_eps

    def _validate_params(self) -> None:
        if not 0.5 < self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in (0.5, 1]; got {self.threshold} "
                "(MB error-control requires > 0.5)"
            )
        if not 0.0 < self.sample_fraction < 1.0:
            raise ValueError(
                f"sample_fraction must be in (0, 1); got {self.sample_fraction}"
            )
        if self.n_bootstraps < 1:
            raise ValueError(f"n_bootstraps must be ≥ 1; got {self.n_bootstraps}")
        lam = np.asarray(self.lambdas, dtype=np.float64)
        if lam.ndim != 1 or lam.size == 0:
            raise ValueError("lambdas must be a non-empty 1D array")
        if np.any(lam <= 0):
            raise ValueError("lambdas must be positive")

    def fit(self, X) -> "GraphicalStabilitySelection":
        from joblib import Parallel, delayed

        self._validate_params()
        is_joint = _is_joint(self.base_estimator)
        alpha_name = "lambda_2" if is_joint else "alpha"
        lambdas = np.ascontiguousarray(self.lambdas, dtype=np.float64)
        rng = check_random_state(self.random_state)

        if is_joint:
            if not isinstance(X, (list, tuple)) or len(X) == 0:
                raise ValueError(
                    "joint estimator: X must be a non-empty list/tuple "
                    "of per-population arrays"
                )
            Xs = [_validate_raw_X(x, f"X[{k}]") for k, x in enumerate(X)]
            ns = [x.shape[0] for x in Xs]
            p = Xs[0].shape[1]
            for k, x in enumerate(Xs):
                if x.shape[1] != p:
                    raise ValueError(
                        f"all populations must share n_features; "
                        f"X[0] has {p}, X[{k}] has {x.shape[1]}"
                    )
            sub_sizes = [max(1, int(round(self.sample_fraction * n))) for n in ns]
            boot_indices: list[Any] = [
                tuple(
                    np.sort(
                        rng.choice(ns[k], size=sub_sizes[k], replace=False)
                    ).astype(np.int64)
                    for k in range(len(Xs))
                )
                for _ in range(self.n_bootstraps)
            ]
            X_or_Xs: Any = Xs
            self.n_populations_ = len(Xs)
        else:
            X_arr = _validate_raw_X(X, "X")
            n, p = X_arr.shape
            sub_size = max(1, int(round(self.sample_fraction * n)))
            boot_indices = [
                np.sort(
                    rng.choice(n, size=sub_size, replace=False)
                ).astype(np.int64)
                for _ in range(self.n_bootstraps)
            ]
            X_or_Xs = X_arr

        self.n_features_in_ = p

        masks = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_stability_one)(
                self.base_estimator,
                X_or_Xs,
                idx,
                lambdas,
                alpha_name,
                is_joint,
                self.active_eps,
            )
            for idx in boot_indices
        )

        stacked = np.stack(masks, axis=0)  # (B, n_lambdas, [K,] p, p)
        sel_prob = stacked.mean(axis=0)
        self.selection_probabilities_ = sel_prob
        self.lambdas_ = lambdas

        if is_joint:
            # sel_prob shape: (n_lambdas, K, p, p)
            self.max_probabilities_ = sel_prob.max(axis=0)  # (K, p, p)
            self.stable_edges_ = [
                _stable_pairs(self.max_probabilities_[k], self.threshold)
                for k in range(self.n_populations_)
            ]
        else:
            self.max_probabilities_ = sel_prob.max(axis=0)  # (p, p)
            self.stable_edges_ = _stable_pairs(
                self.max_probabilities_, self.threshold
            )

        return self


# --- non-parametric bootstrap ----------------------------------------


class GraphicalBootstrap(BaseEstimator):
    """Non-parametric bootstrap for graphical edge weights.

    Resamples observations with replacement (same `n`), refits the
    graphical estimator at the given regularisation parameter, and
    records the bootstrap sampling distribution of every edge in
    `Θ̂`. Output statistics:

    * **`mean_`, `std_`** — per-edge bootstrap mean and standard
      deviation. The SD doubles as a Wald-style standard error.
    * **`ci_lower_`, `ci_upper_`** — per-edge quantile confidence
      bands at `alpha/2` and `1 − alpha/2`. The default
      ``alpha=0.05`` gives the 2.5% / 97.5% percentile envelope
      typically plotted on bootnet stability diagrams.
    * **`edge_selection_probabilities_`** — fraction of bootstrap
      replicates in which the edge was active. This is what
      ``bootnet`` calls the "edge stability" output (distinct from
      the MB-style :class:`GraphicalStabilitySelection`, which
      sweeps λ instead of resampling at a fixed λ).

    Parameters
    ----------
    base_estimator
        A fully-configured graphical estimator. Its regularisation
        parameter (``alpha`` for single, ``lambda_2`` for joint) is
        used as-is at every bootstrap replicate; this class does not
        sweep λ.
    n_bootstraps : int, default 200
        Number of bootstrap replicates.
    alpha : float, default 0.05
        CI level — bands are `[alpha/2, 1 - alpha/2]` quantiles.
    random_state : int or None
    n_jobs : int or None
    active_eps : float, default 1e-6

    Attributes
    ----------
    precisions_ : ndarray
        Single-pop: shape `(B, p, p)`. Joint: shape `(B, K, p, p)`.
        Full bootstrap stack of fitted precisions.
    mean_, std_, ci_lower_, ci_upper_ : ndarray
        Single-pop: shape `(p, p)`. Joint: shape `(K, p, p)`.
    edge_selection_probabilities_ : ndarray
        Same shape as ``mean_``. Off-diagonal selection probability;
        diagonal is zero.
    n_features_in_ : int
    n_populations_ : int (joint only)
    """

    precisions_: NDArray[np.float64]
    mean_: NDArray[np.float64]
    std_: NDArray[np.float64]
    ci_lower_: NDArray[np.float64]
    ci_upper_: NDArray[np.float64]
    edge_selection_probabilities_: NDArray[np.float64]
    n_features_in_: int

    def __init__(
        self,
        base_estimator: Any,
        *,
        n_bootstraps: int = 200,
        alpha: float = 0.05,
        random_state: int | None = None,
        n_jobs: int | None = None,
        active_eps: float = 1e-6,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_bootstraps = n_bootstraps
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.active_eps = active_eps

    def _validate_params(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if self.n_bootstraps < 2:
            raise ValueError(
                f"n_bootstraps must be ≥ 2 for a non-degenerate CI; "
                f"got {self.n_bootstraps}"
            )

    def fit(self, X) -> "GraphicalBootstrap":
        from joblib import Parallel, delayed

        self._validate_params()
        is_joint = _is_joint(self.base_estimator)
        rng = check_random_state(self.random_state)

        if is_joint:
            if not isinstance(X, (list, tuple)) or len(X) == 0:
                raise ValueError(
                    "joint estimator: X must be a non-empty list/tuple "
                    "of per-population arrays"
                )
            Xs = [_validate_raw_X(x, f"X[{k}]") for k, x in enumerate(X)]
            ns = [x.shape[0] for x in Xs]
            p = Xs[0].shape[1]
            for k, x in enumerate(Xs):
                if x.shape[1] != p:
                    raise ValueError(
                        f"all populations must share n_features; "
                        f"X[0] has {p}, X[{k}] has {x.shape[1]}"
                    )
            boot_indices: list[Any] = [
                tuple(
                    rng.choice(ns[k], size=ns[k], replace=True).astype(np.int64)
                    for k in range(len(Xs))
                )
                for _ in range(self.n_bootstraps)
            ]
            X_or_Xs: Any = Xs
            self.n_populations_ = len(Xs)
        else:
            X_arr = _validate_raw_X(X, "X")
            n, p = X_arr.shape
            boot_indices = [
                rng.choice(n, size=n, replace=True).astype(np.int64)
                for _ in range(self.n_bootstraps)
            ]
            X_or_Xs = X_arr

        self.n_features_in_ = p

        precs = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_bootstrap_one)(
                self.base_estimator, X_or_Xs, idx, is_joint
            )
            for idx in boot_indices
        )

        stacked = np.stack(precs, axis=0)  # (B, [K,] p, p)
        self.precisions_ = stacked
        self.mean_ = stacked.mean(axis=0)
        self.std_ = stacked.std(axis=0, ddof=1)
        q_lo = self.alpha / 2.0
        q_hi = 1.0 - q_lo
        self.ci_lower_ = np.quantile(stacked, q_lo, axis=0)
        self.ci_upper_ = np.quantile(stacked, q_hi, axis=0)

        # Edge selection probability — off-diagonal only, diagonal zeroed.
        active = np.abs(stacked) > self.active_eps
        sel = active.mean(axis=0)
        if is_joint:
            for k in range(sel.shape[0]):
                np.fill_diagonal(sel[k], 0.0)
        else:
            np.fill_diagonal(sel, 0.0)
        self.edge_selection_probabilities_ = sel

        return self


__all__ = ["GraphicalStabilitySelection", "GraphicalBootstrap"]
