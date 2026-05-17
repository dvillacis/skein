"""Correlation preprocessing for ordinal / mixed-type data.

Network psychometrics fits graphical models to *correlation* matrices,
not raw data — and the right correlation depends on the variable type:

- Pearson for continuous–continuous pairs
- **Polychoric** for ordinal–ordinal pairs (Olsson 1979)
- **Polyserial** for ordinal–continuous pairs (Olsson, Drasgow & Dorans
  1982)

Naïve Pearson on ordinal Likert data systematically *under-estimates*
the underlying continuous correlation; the resulting network has
biased edge weights and the wrong dependence structure. Polychoric
inverts the discretization and recovers the latent Gaussian
correlation that would have produced the observed ordinal frequencies.

This module implements the **two-step ML estimator** of Olsson (1979):

1. **Step 1 — thresholds.** For each ordinal column, take the
   cumulative marginal frequencies and pull back through the inverse
   standard-normal CDF: ``τ_a = Φ⁻¹(P̂[X ≤ a])``.

2. **Step 2 — ρ.** For each pair, maximize the bivariate-normal
   log-likelihood
   ``ℓ(ρ) = Σ_ab n_ab · log π_ab(ρ)``
   over ``ρ ∈ (-1+ε, 1-ε)`` via Brent's bounded optimizer, where
   ``π_ab(ρ) = Φ_2(τ_a, τ_b; ρ) - Φ_2(τ_{a-1}, τ_b; ρ) ...`` is the
   bivariate-normal probability over the rectangle defined by the
   thresholds.

The output is directly usable as the ``cov`` input to
:class:`~skein_glm.GraphicalLasso` and friends.

References
----------
Olsson, U. (1979). "Maximum likelihood estimation of the polychoric
correlation coefficient." *Psychometrika* 44(4): 443–460.

Olsson, U., Drasgow, F., & Dorans, N. J. (1982). "The polyserial
correlation coefficient." *Psychometrika* 47(3): 337–347.

Drasgow, F. (1986). "Polychoric and polyserial correlations." In *The
Encyclopedia of Statistics*.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize_scalar

# Numerical guards. ρ is bounded inside ±(1-_RHO_EDGE) so the
# bivariate normal stays non-degenerate; bivariate cell probabilities
# are floored at _PI_FLOOR before taking logs to avoid -inf.
_RHO_EDGE = 1e-6
_PI_FLOOR = 1e-12
_DEFAULT_ORDINAL_LEVELS = 10  # ≤ this many unique values → treated as ordinal
# Threshold sentinels in place of ±inf. `scipy.stats.multivariate_normal.cdf`
# in scipy < 1.16 returns NaN when any coordinate is `-np.inf` (the +inf
# arm works), which silently breaks `_bivariate_rect_prob` → every cell
# probability gets clipped to _PI_FLOOR → log-likelihood is constant in ρ
# → Brent's optimizer terminates at its golden-section initial point
# (≈ ±0.236). Fixed in scipy 1.16+ but Python 3.10 pip-installs ≤ 1.15.
# Φ(±8) is 1 − 6.2e-16 / 6.2e-16 in double precision — effectively ±∞ for
# the rectangle-difference arithmetic, and the interior thresholds are
# bounded in `[stats.norm.ppf(1e-15), stats.norm.ppf(1-1e-15)] ≈ [-7.94,
# 7.94]` by the `cum` clipping below, so ±8 is safely outside the
# interior-threshold range.
_INF_SENTINEL = 8.0


def _normalize_ordinal_column(col: NDArray) -> NDArray[np.int64]:
    """Map a column to consecutive integer levels 0..L-1.

    Accepts integer-coded or float-coded ordinal data. NaN-bearing
    rows are flagged with a sentinel ``-1``; callers handle pairwise
    deletion themselves.
    """
    arr = np.asarray(col)
    mask = np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.zeros(
        arr.shape, dtype=bool
    )
    finite = arr[~mask]
    levels, inv = np.unique(finite, return_inverse=True)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[~mask] = inv
    return out


def _thresholds_from_counts(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    """Olsson step 1: invert the cumulative marginal to get thresholds.

    Returns ``(L+1,)`` thresholds with ``-inf`` and ``+inf`` at the
    endpoints. With ``L`` observed levels, there are ``L-1`` finite
    interior thresholds and 2 sentinel infinities.
    """
    n = counts.sum()
    if n == 0:
        return np.array([-_INF_SENTINEL, _INF_SENTINEL])
    # Olsson 0.5 continuity correction for zero-count cells avoids
    # τ = ±inf for the lowest / highest observed level.
    safe = np.maximum(counts, 0.5)
    cum = np.cumsum(safe) / safe.sum()
    # Clip away from 0 / 1 so Φ⁻¹ doesn't return ±inf.
    cum = np.clip(cum, 1e-15, 1.0 - 1e-15)
    interior = stats.norm.ppf(cum[:-1])
    # ±_INF_SENTINEL instead of ±np.inf at endpoints — see the constant's
    # docstring for the scipy 1.15 `multivariate_normal.cdf` NaN-at-(-inf)
    # bug that motivates this. The interior values from `stats.norm.ppf`
    # are bounded by the `cum` clipping above (|x| ≤ 7.94), so ±8 stays
    # strictly outside the interior range and the rectangle-difference
    # arithmetic in `_bivariate_rect_prob` still recovers `≈1` for the
    # full-domain integral.
    return np.concatenate(([-_INF_SENTINEL], interior, [_INF_SENTINEL]))


def _bivariate_rect_prob(
    tau_j: NDArray[np.float64],
    tau_k: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """Per-cell bivariate-normal rectangle probabilities ``π_ab(ρ)``.

    Uses the 4-corner formula:
    ``P(a < Z_j < b, c < Z_k < d) = Φ_2(b, d) - Φ_2(a, d) - Φ_2(b, c) + Φ_2(a, c)``

    Returns a ``(L_j, L_k)`` array of probabilities summing to ≈ 1.
    """
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    # Vectorize over the (L_j + 1) × (L_k + 1) corner grid.
    grid_j, grid_k = np.meshgrid(tau_j, tau_k, indexing="ij")
    corners = np.stack([grid_j.ravel(), grid_k.ravel()], axis=1)
    cdf = mvn.cdf(corners).reshape(grid_j.shape)
    # Rectangle prob over cell [a, a+1] × [b, b+1]:
    pi = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
    return np.clip(pi, _PI_FLOOR, 1.0)


def _polychoric_pair(
    x: NDArray[np.int64],
    y: NDArray[np.int64],
    tau_x: NDArray[np.float64],
    tau_y: NDArray[np.float64],
) -> float:
    """Maximize the polychoric log-likelihood for one pair via Brent.

    Inputs ``x``/``y`` are integer-coded ordinal columns *with sentinel
    -1 for missing*; missing rows are pairwise-deleted.
    """
    keep = (x >= 0) & (y >= 0)
    if keep.sum() < 2:
        return float("nan")
    xc, yc = x[keep], y[keep]
    Lx, Ly = len(tau_x) - 1, len(tau_y) - 1
    # Cell counts n_ab.
    counts = np.zeros((Lx, Ly), dtype=np.float64)
    np.add.at(counts, (xc, yc), 1.0)

    def neg_loglik(rho: float) -> float:
        pi = _bivariate_rect_prob(tau_x, tau_y, rho)
        return -float(np.sum(counts * np.log(pi)))

    result = minimize_scalar(
        neg_loglik,
        bounds=(-1.0 + _RHO_EDGE, 1.0 - _RHO_EDGE),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(result.x)


def polychoric_correlation(
    X: NDArray,
    *,
    missing: Literal["pairwise", "listwise"] = "pairwise",
) -> NDArray[np.float64]:
    """Polychoric correlation matrix for an ordinal data matrix.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Ordinal observations. Each column should contain a small
        number of distinct levels (≤ ~10); levels can be integer or
        float-coded. ``NaN`` entries are treated as missing.

    missing : {"pairwise", "listwise"}, default "pairwise"
        How to handle missing values. ``"pairwise"`` deletes only
        rows missing for the specific pair being estimated;
        ``"listwise"`` drops any row with a missing value before
        estimating any pair.

    Returns
    -------
    R : ndarray of shape (p, p)
        Polychoric correlation matrix. Diagonal is 1.0; the matrix is
        symmetric; entries are in ``[-1, 1]``. Entries for which a pair
        had < 2 jointly observed rows are ``NaN``.

    Notes
    -----
    Implementation follows Olsson (1979): two-step ML, with the per-
    pair ρ optimization in Brent's bounded scalar minimizer.
    Bivariate-normal rectangle probabilities are computed via the
    four-corner formula using ``scipy.stats.multivariate_normal``.

    The 0.5 continuity correction is applied to zero-count cells so
    that thresholds for sparsely-sampled categories don't collapse to
    ±∞.

    References
    ----------
    Olsson, U. (1979). "Maximum likelihood estimation of the
    polychoric correlation coefficient." *Psychometrika* 44(4):
    443–460.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    _, p = X.shape
    if p < 2:
        raise ValueError(f"X must have at least 2 columns, got {p}")

    if missing == "listwise":
        if np.issubdtype(X.dtype, np.floating):
            keep = ~np.any(np.isnan(X), axis=1)
            X = X[keep]
    elif missing != "pairwise":
        raise ValueError(f"missing must be 'pairwise' or 'listwise', got {missing!r}")

    # Encode each column to consecutive integer levels.
    cols = [_normalize_ordinal_column(X[:, j]) for j in range(p)]
    # Per-column marginal counts and thresholds.
    thresholds = []
    for j, col in enumerate(cols):
        finite = col[col >= 0]
        if finite.size == 0:
            raise ValueError(f"column {j} has no non-missing entries")
        L = int(finite.max()) + 1
        counts = np.bincount(finite, minlength=L)
        thresholds.append(_thresholds_from_counts(counts))

    R = np.eye(p, dtype=np.float64)
    for j in range(p):
        for k in range(j + 1, p):
            rho = _polychoric_pair(cols[j], cols[k], thresholds[j], thresholds[k])
            R[j, k] = rho
            R[k, j] = rho
    return R


def _polyserial_pair(
    x_ord: NDArray[np.int64],
    y_cont: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> float:
    """ML polyserial ρ for one ordinal × one continuous pair.

    Maximizes the joint log-likelihood
    ``ℓ(ρ, μ, σ) = Σ_i [log φ(z_i)/σ + log {Φ((τ_a - ρ z_i)/√(1-ρ²))
                                          - Φ((τ_{a-1} - ρ z_i)/√(1-ρ²))}]``
    profile-likelihood in ρ with μ, σ at sample moments
    (Olsson-Drasgow-Dorans 1982, ML version).
    """
    keep = (x_ord >= 0) & np.isfinite(y_cont)
    if keep.sum() < 3:
        return float("nan")
    xo, yc = x_ord[keep], y_cont[keep]
    mu = yc.mean()
    sigma = yc.std(ddof=1)
    if sigma <= 0:
        return float("nan")
    z = (yc - mu) / sigma  # standardized continuous score
    # Boundaries for each observation's ordinal cell.
    tau_lo = tau[xo]  # τ_{a-1}
    tau_hi = tau[xo + 1]  # τ_a

    def neg_loglik(rho: float) -> float:
        denom = np.sqrt(max(1.0 - rho * rho, _RHO_EDGE))
        upper = stats.norm.cdf((tau_hi - rho * z) / denom)
        lower = stats.norm.cdf((tau_lo - rho * z) / denom)
        cond = np.clip(upper - lower, _PI_FLOOR, 1.0)
        # The marginal log φ(z)/σ term doesn't depend on ρ, so we drop
        # it from the objective being minimized — it cancels.
        return -float(np.sum(np.log(cond)))

    result = minimize_scalar(
        neg_loglik,
        bounds=(-1.0 + _RHO_EDGE, 1.0 - _RHO_EDGE),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(result.x)


def polyserial_correlation(
    X_ord: NDArray,
    Y_cont: NDArray,
) -> float | NDArray[np.float64]:
    """Polyserial correlation(s) for ordinal × continuous variable(s).

    Parameters
    ----------
    X_ord : array-like, shape (n,) or (n, p_ord)
        Ordinal observations. ``NaN`` rows are dropped pairwise.

    Y_cont : array-like, shape (n,) or (n, p_cont)
        Continuous observations.

    Returns
    -------
    R : ndarray
        Polyserial correlations. Scalar if both inputs are 1D; shape
        ``(p_ord,)`` if only ``X_ord`` is 2D; shape ``(p_cont,)`` if
        only ``Y_cont`` is 2D; shape ``(p_ord, p_cont)`` if both are 2D.

    References
    ----------
    Olsson, U., Drasgow, F., & Dorans, N. J. (1982). "The polyserial
    correlation coefficient." *Psychometrika* 47(3): 337–347.
    """
    X_ord = np.asarray(X_ord)
    Y_cont = np.asarray(Y_cont, dtype=np.float64)
    x_scalar = X_ord.ndim == 1
    y_scalar = Y_cont.ndim == 1
    if x_scalar:
        X_ord = X_ord[:, None]
    if y_scalar:
        Y_cont = Y_cont[:, None]
    n, p_ord = X_ord.shape
    if Y_cont.shape[0] != n:
        raise ValueError(
            f"X_ord and Y_cont must have the same n; got {n} vs {Y_cont.shape[0]}"
        )
    p_cont = Y_cont.shape[1]

    # Encode + threshold each ordinal column once.
    cols = [_normalize_ordinal_column(X_ord[:, j]) for j in range(p_ord)]
    thresholds = []
    for j, col in enumerate(cols):
        finite = col[col >= 0]
        if finite.size == 0:
            raise ValueError(f"X_ord[:, {j}] has no non-missing entries")
        L = int(finite.max()) + 1
        counts = np.bincount(finite, minlength=L)
        thresholds.append(_thresholds_from_counts(counts))

    R = np.full((p_ord, p_cont), np.nan, dtype=np.float64)
    for j in range(p_ord):
        for k in range(p_cont):
            R[j, k] = _polyserial_pair(cols[j], Y_cont[:, k], thresholds[j])

    if x_scalar and y_scalar:
        return float(R[0, 0])
    if x_scalar:
        return R[0]
    if y_scalar:
        return R[:, 0]
    return R


def polychoric_covariance_matrix(
    X: NDArray,
    *,
    continuous_mask: Sequence[bool] | None = None,
    ordinal_levels_cutoff: int = _DEFAULT_ORDINAL_LEVELS,
    missing: Literal["pairwise", "listwise"] = "pairwise",
) -> NDArray[np.float64]:
    """Mixed polychoric / polyserial / Pearson correlation matrix.

    Detects ordinal vs continuous columns (by unique-value count if
    ``continuous_mask`` is not given), then dispatches each pairwise
    correlation to the appropriate estimator:

    - ordinal × ordinal → polychoric (Olsson 1979)
    - ordinal × continuous → polyserial (Olsson-Drasgow-Dorans 1982)
    - continuous × continuous → Pearson

    The resulting ``(p, p)`` matrix is suitable as the ``cov`` input
    to :class:`~skein_glm.GraphicalLasso`, :class:`~skein_glm.GraphicalMCP`,
    or :class:`~skein_glm.GraphicalSCAD`.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Mixed-type data. Ordinal columns can be integer- or float-
        coded with a small number of distinct values.

    continuous_mask : sequence of bool, length p, optional
        If provided, ``continuous_mask[j] = True`` forces column ``j``
        to be treated as continuous. If omitted, columns with
        ``> ordinal_levels_cutoff`` unique values are auto-detected
        as continuous.

    ordinal_levels_cutoff : int, default 10
        Columns with at most this many distinct non-missing values
        are treated as ordinal. Ignored if ``continuous_mask`` is
        given.

    missing : {"pairwise", "listwise"}, default "pairwise"
        Missing-value handling.

    Returns
    -------
    R : ndarray of shape (p, p)
        Mixed correlation matrix; symmetric, diag 1.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    _, p = X.shape

    if missing == "listwise":
        keep = ~np.any(np.isnan(X), axis=1)
        X = X[keep]
    elif missing != "pairwise":
        raise ValueError(f"missing must be 'pairwise' or 'listwise', got {missing!r}")

    continuous: NDArray[np.bool_]
    if continuous_mask is None:
        continuous = np.zeros(p, dtype=bool)
        for j in range(p):
            finite = X[~np.isnan(X[:, j]), j]
            if np.unique(finite).size > ordinal_levels_cutoff:
                continuous[j] = True
    else:
        continuous = np.asarray(continuous_mask, dtype=bool)
        if continuous.shape != (p,):
            raise ValueError(
                f"continuous_mask must have length {p}, got {continuous.shape}"
            )

    ordinal_cols = [j for j in range(p) if not continuous[j]]
    cont_cols = [j for j in range(p) if continuous[j]]

    R = np.eye(p, dtype=np.float64)

    # Pearson block (continuous × continuous). Use pairwise-complete
    # observations when missing="pairwise".
    if len(cont_cols) >= 2:
        if missing == "pairwise":
            for ji, j in enumerate(cont_cols):
                for k in cont_cols[ji + 1 :]:
                    mask = ~(np.isnan(X[:, j]) | np.isnan(X[:, k]))
                    if mask.sum() >= 2:
                        c = np.corrcoef(X[mask, j], X[mask, k])[0, 1]
                    else:
                        c = float("nan")
                    R[j, k] = R[k, j] = c
        else:
            sub = X[:, cont_cols]
            if sub.shape[0] >= 2:
                C = np.corrcoef(sub, rowvar=False)
                for ji, j in enumerate(cont_cols):
                    for ki, k in enumerate(cont_cols):
                        R[j, k] = C[ji, ki]

    # Polychoric block (ordinal × ordinal).
    if len(ordinal_cols) >= 2:
        sub_R = polychoric_correlation(X[:, ordinal_cols], missing=missing)
        for ji, j in enumerate(ordinal_cols):
            for ki, k in enumerate(ordinal_cols):
                R[j, k] = sub_R[ji, ki]

    # Polyserial block (ordinal × continuous).
    if ordinal_cols and cont_cols:
        # Encode ordinal cols once.
        cols_ord = [_normalize_ordinal_column(X[:, j]) for j in ordinal_cols]
        thresholds_ord = []
        for col in cols_ord:
            finite = col[col >= 0]
            L = int(finite.max()) + 1
            counts = np.bincount(finite, minlength=L)
            thresholds_ord.append(_thresholds_from_counts(counts))
        for ji, j in enumerate(ordinal_cols):
            for k in cont_cols:
                rho = _polyserial_pair(cols_ord[ji], X[:, k], thresholds_ord[ji])
                R[j, k] = R[k, j] = rho

    return R
