"""Meinshausen-Bühlmann (2010) stability selection.

A variable-selection meta-procedure that wraps any skein `*PathRegressor`.
For each bootstrap iteration, subsample the data (default half-sample
without replacement), fit the underlying path estimator on the subsample,
and record the active set at each λ. Aggregate over bootstraps to get
per-(feature, λ) selection probabilities; the **stable set** is the
features whose max-over-λ selection probability exceeds a threshold.

Why this matters: stability selection sidesteps the brittle "pick a single
λ" decision. The MB result (Theorem 1, *J. R. Stat. Soc. B* 2010) gives
an explicit error-control bound on expected false positives in the
stable set, depending only on the threshold π_thr and the typical
active-set size — not on λ.

Stability selection is the headline M5.x differentiator: neither
``glmnet``'s ``cv.glmnet`` nor ``skglm`` ships a clean implementation.
``grpreg`` has nothing comparable. The bootstrap loop is embarrassingly
parallel — pass ``n_jobs=-1`` to use every core.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import check_random_state


def _active_mask_per_lambda(
    coefs: NDArray[np.float64],
    eps: float,
) -> NDArray[np.bool_]:
    """`(n_lambdas, p)` boolean mask: `True` where `|β| > eps`."""
    return np.abs(coefs) > eps


def _group_active_mask_per_lambda(
    coefs: NDArray[np.float64],
    groups: NDArray[np.int64],
    eps: float,
) -> NDArray[np.bool_]:
    """For grouped estimators: a feature `j` is "selected" at λ_k if any
    feature in its group has nonzero coefficient. Returns the same
    `(n_lambdas, p)` shape (so the rest of the pipeline doesn't care
    about the grouping)."""
    n_lambdas, p = coefs.shape
    groups = np.asarray(groups, dtype=np.int64)
    if groups.shape[0] != p:
        raise ValueError(
            f"groups length {groups.shape[0]} does not match n_features {p}"
        )
    n_groups = int(groups.max()) + 1
    out = np.zeros((n_lambdas, p), dtype=np.bool_)
    for g in range(n_groups):
        mask = groups == g
        # Per-λ: group is active if any feature in the group is active.
        group_active = np.asarray(
            (np.abs(coefs[:, mask]) > eps).any(axis=1)
        )
        out[:, mask] = group_active[:, None]
    return out


def _fit_one_bootstrap(
    base_estimator,
    x,
    y_or_outcomes,
    indices: NDArray[np.int64],
    lambdas: NDArray[np.float64],
    is_cox: bool,
    is_grouped: bool,
    active_eps: float,
):
    """Fit the underlying path estimator on the subsample at the master
    λ-grid and return its `(n_lambdas, p)` active-mask.

    Sparse `x` is row-indexed via tocsr/back; dense via fancy indexing.
    """
    # Row-index x.
    try:
        from scipy import sparse  # type: ignore[import-untyped]
        if sparse.issparse(x):
            x_sub = (x.tocsr() if not sparse.isspmatrix_csr(x) else x)[indices]
            x_sub = x_sub.tocsc()
        else:
            x_sub = np.ascontiguousarray(np.asarray(x)[indices], dtype=np.float64)
    except ImportError:
        x_sub = np.ascontiguousarray(np.asarray(x)[indices], dtype=np.float64)

    est = clone(base_estimator)
    # Override the λ-grid to share with all other bootstraps.
    est.lambdas = lambdas

    if is_cox:
        time, event = y_or_outcomes
        est.fit(x_sub, time[indices], event[indices])
    else:
        y = np.asarray(y_or_outcomes, dtype=np.float64)
        est.fit(x_sub, y[indices])

    # `coefs_` may be (n_lambdas, p) for scalar/group LS, GLMs, Cox;
    # multi-task / multinomial have (n_lambdas, K, p) — flatten the
    # K axis with an "any-class active" rule.
    coefs = np.asarray(est.coefs_)
    if coefs.ndim == 3:
        # (n_lambdas, K, p) — collapse: feature is active if any class
        # has a nonzero coefficient.
        per_lambda = (np.abs(coefs) > active_eps).any(axis=1)  # (n_lambdas, p)
        return per_lambda
    if is_grouped:
        return _group_active_mask_per_lambda(coefs, est.groups, active_eps)
    return _active_mask_per_lambda(coefs, active_eps)


class StabilitySelection(BaseEstimator, TransformerMixin):
    """Meinshausen-Bühlmann (2010) stability selection.

    Wraps any skein `*PathRegressor` (LS, group, GLM, multi-task,
    multinomial, Cox) and turns it into a feature-selection procedure
    via subsample-bootstrap.

    Parameters
    ----------
    base_estimator
        A skein path estimator with a ``fit`` method, ``lambdas_`` and
        ``coefs_`` attributes after fitting, and (optionally) a
        ``groups`` attribute for grouped variants. Examples:
        ``MCPPathRegressor()``, ``GroupLassoPathRegressor(groups=...)``,
        ``LogisticMCPPathRegressor()``, ``CoxMCPPathRegressor()``.
    n_bootstraps : int, default 100
        Number of subsample-bootstrap iterations. The MB error-control
        bound improves with B (asymptotically); 50–200 is typical.
    sample_fraction : float, default 0.5
        Fraction of `n` to subsample without replacement at each
        iteration. MB recommend 0.5 (half-sample subsampling).
    threshold : float, default 0.6
        Selection-probability cutoff for the stable set. Must be > 0.5
        for the MB bound to apply. Higher → fewer false positives.
    random_state : int or None, default None
        Seed for the bootstrap subsampling RNG.
    n_jobs : int or None, default None
        Number of parallel workers via ``joblib``. ``-1`` uses all
        cores. ``None`` is serial.
    active_eps : float, default 1e-8
        Threshold for declaring `|β_j|` "nonzero". Looser than the
        ``select_by_ic`` default (1e-12) to absorb mild solver drift
        across bootstrap subsamples.

    Attributes
    ----------
    selection_probabilities_ : ndarray of shape (n_lambdas, n_features)
        ``Π_j(λ_k)`` — fraction of bootstraps in which feature ``j``
        was active at ``lambdas_[k]``.
    max_probabilities_ : ndarray of shape (n_features,)
        ``max_k Π_j(λ_k)``. The MB selection criterion.
    stable_features_ : ndarray of int64
        Indices into ``[0, n_features)`` of features in the stable set
        (``max_probabilities_ >= threshold``).
    lambdas_ : ndarray of shape (n_lambdas,)
        The shared λ-grid used across all bootstraps. Drawn once from
        a full-data fit of ``base_estimator``.
    n_features_in_ : int

    Notes
    -----
    For grouped path estimators (``GroupLassoPathRegressor`` etc.),
    "selected" is evaluated at the group level: a feature ``j`` is
    counted as selected at ``λ_k`` if any coefficient in its group is
    nonzero. The output ``selection_probabilities_`` is still
    ``(n_lambdas, n_features)`` — features in the same group share the
    same probability.

    For multi-task / multinomial path estimators with ``coefs_`` shape
    ``(n_lambdas, K, p)``, "selected" means any of the K class/task
    coefficients is nonzero (the row-grouped active-feature rule).

    Cox (``fit(x, time, event)``) is detected by whether the base
    estimator has a ``ties`` attribute; ``fit`` then expects
    ``stability.fit(x, (time, event))`` — pass the outcomes as a tuple.

    Examples
    --------
    >>> from skein_glm import MCPPathRegressor, StabilitySelection
    >>> ss = StabilitySelection(
    ...     base_estimator=MCPPathRegressor(gamma=3.0, n_lambdas=50),
    ...     n_bootstraps=100, threshold=0.6, n_jobs=-1, random_state=0,
    ... )
    >>> ss.fit(X, y)
    >>> stable_X = ss.transform(X)
    >>> ss.stable_features_
    array([0, 2, 4, 5])

    References
    ----------
    Meinshausen, N. and Bühlmann, P. (2010). "Stability selection".
    *Journal of the Royal Statistical Society B* 72(4), 417-473.
    """

    selection_probabilities_: NDArray[np.float64]
    max_probabilities_: NDArray[np.float64]
    stable_features_: NDArray[np.int64]
    lambdas_: NDArray[np.float64]
    n_features_in_: int

    def __init__(
        self,
        base_estimator,
        *,
        n_bootstraps: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        random_state: int | None = None,
        n_jobs: int | None = None,
        active_eps: float = 1e-8,
    ) -> None:
        self.base_estimator = base_estimator
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

    def _detect_cox(self) -> bool:
        return hasattr(self.base_estimator, "ties")

    def _detect_grouped(self) -> bool:
        # Grouped LS / GLM path estimators have a `groups` attribute set
        # in __init__. Multi-task / multinomial classes don't (they
        # derive groups internally) — those are handled via the 3D
        # coefs_ branch in `_fit_one_bootstrap`.
        return hasattr(self.base_estimator, "groups") and self.base_estimator.groups is not None

    def fit(self, x, y) -> "StabilitySelection":
        """Run the bootstrap loop and store selection probabilities.

        For Cox base estimators, pass outcomes as ``y=(time, event)``.
        """
        from joblib import Parallel, delayed

        self._validate_params()

        is_cox = self._detect_cox()
        is_grouped = self._detect_grouped()

        # Resolve n_samples, normalize y / outcomes.
        if is_cox:
            time, event = y
            time = np.ascontiguousarray(time, dtype=np.float64)
            event = np.ascontiguousarray(event, dtype=np.float64)
            n_samples = time.shape[0]
            outcomes_full: Any = (time, event)
        else:
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            n_samples = y_arr.shape[0]
            outcomes_full = y_arr

        # Full-data fit drives the shared λ-grid.
        full = clone(self.base_estimator)
        if is_cox:
            full.fit(x, outcomes_full[0], outcomes_full[1])
        else:
            full.fit(x, outcomes_full)
        lambdas = np.ascontiguousarray(full.lambdas_, dtype=np.float64)

        # Resolve n_features from the full-data fit. coefs_ may be 2D
        # ((n_lambdas, p)) or 3D ((n_lambdas, K, p) for multi-task /
        # multinomial); take the last axis.
        coefs_full = np.asarray(full.coefs_)
        n_features = coefs_full.shape[-1]
        self.n_features_in_ = n_features

        # Generate bootstrap indices up-front; each child sees a
        # deterministic subsample for the given seed.
        rng = check_random_state(self.random_state)
        sub_size = max(1, int(round(self.sample_fraction * n_samples)))
        boot_indices = [
            np.sort(
                rng.choice(n_samples, size=sub_size, replace=False)
            ).astype(np.int64)
            for _ in range(self.n_bootstraps)
        ]

        # Parallelize the bootstrap loop. Each worker returns a
        # (n_lambdas, n_features) boolean mask.
        masks = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_bootstrap)(
                self.base_estimator,
                x,
                outcomes_full,
                idx,
                lambdas,
                is_cox,
                is_grouped,
                self.active_eps,
            )
            for idx in boot_indices
        )

        # Aggregate: average boolean masks per (λ, feature).
        stacked = np.stack(masks, axis=0)  # (B, n_lambdas, p)
        self.selection_probabilities_ = stacked.mean(axis=0)
        self.max_probabilities_ = self.selection_probabilities_.max(axis=0)
        self.stable_features_ = np.flatnonzero(
            self.max_probabilities_ >= self.threshold
        ).astype(np.int64)
        self.lambdas_ = lambdas

        return self

    def transform(self, x):
        """Return ``x`` restricted to the stable feature set."""
        try:
            from scipy import sparse  # type: ignore[import-untyped]
            if sparse.issparse(x):
                return x[:, self.stable_features_]
        except ImportError:
            pass
        return np.asarray(x)[:, self.stable_features_]

    def fit_transform(self, x, y, **fit_params):
        return self.fit(x, y, **fit_params).transform(x)


__all__ = ["StabilitySelection"]
