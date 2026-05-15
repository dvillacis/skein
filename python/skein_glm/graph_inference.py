"""Edge-level inference for graphical models.

Mainstream graphical-model packages (sklearn's
``GraphicalLasso``, R's ``glasso`` / ``qgraph`` / ``bootnet`` /
``EstimateGroupNetwork``) report point estimates of the precision
matrix and — at best — a per-edge bootstrap selection probability.
None of them provides:

- **Bootstrap-based per-edge p-values** for ``H0: Θ_ij = 0``;
- **Multiple-testing correction** across the ``p(p − 1)/2`` edges
  (BH for FDR; Bonferroni / Holm for FWER);
- The **Meinshausen–Bühlmann (2010) closed-form bound** that turns a
  stability-selection threshold into an expected-false-positive
  guarantee for a graph.

This module provides all three, on top of the existing
:class:`~skein_glm.GraphicalBootstrap` and
:class:`~skein_glm.GraphicalStabilitySelection` output. The
inference is bootstrap-based — exact under the bootstrap's
exchangeability assumption, with no distributional assumption on
the data. For applications where rigor matters (network
psychometrics, biology), this closes the gap between "stable edges"
(an empirical heuristic) and "selected edges at FDR ≤ q" (a
controlled-error procedure).

The functions are thin and composable: they take either a full
``GraphicalBootstrap`` / ``GraphicalStabilitySelection`` result, or
the underlying NumPy arrays. The same-named methods bolted onto the
classes are sugar around these.

References
----------
Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false
discovery rate: a practical and powerful approach to multiple
testing." *J. R. Stat. Soc. B* 57(1): 289–300.

Meinshausen, N., & Bühlmann, P. (2010). "Stability selection."
*J. R. Stat. Soc. B* 72(4): 417–473.

Liu, W. (2013). "Gaussian graphical model estimation with false
discovery rate control." *Annals of Statistics* 41(6): 2948–2978.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Two-sided bootstrap p-values are always ≥ this floor so log-/
# small-p arithmetic stays well-conditioned. With B = 200 bootstraps
# the smallest representable two-sided p is 2/200 = 0.01, so this
# floor only ever kicks in when an edge is *entirely* on one side of
# zero across the bootstrap — which is the strongest signal.
_PVALUE_FLOOR = 1e-12


def _validate_precision_stack(
    precisions: NDArray[np.float64],
) -> tuple[int, int, int]:
    """Return ``(B, K, p)`` for a bootstrap precision stack. ``K=1``
    for single-population stacks of shape ``(B, p, p)``; otherwise
    the second axis is treated as ``K``."""
    if precisions.ndim == 3:
        B, p1, p2 = precisions.shape
        if p1 != p2:
            raise ValueError(
                f"precisions must have square trailing axes; got {precisions.shape}"
            )
        return B, 1, p1
    if precisions.ndim == 4:
        B, K, p1, p2 = precisions.shape
        if p1 != p2:
            raise ValueError(
                f"precisions must have square trailing axes; got {precisions.shape}"
            )
        return B, K, p1
    raise ValueError(
        f"precisions must be 3D (single) or 4D (joint); got ndim={precisions.ndim}"
    )


def edge_pvalues(
    precisions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-edge two-sided bootstrap p-value for ``H0: Θ_ij = 0``.

    For each edge ``(i, j)``, computes
    ``p = 2 · min(P̂(Θ̂* ≥ 0), P̂(Θ̂* ≤ 0))``,
    with **non-strict** inequalities (zeros count on both sides).
    The non-strict choice is essential for sparse estimators
    (graphical lasso, MCP, SCAD) where the bootstrap distribution
    of a null edge is *exactly zero* on every replicate. Under that
    case both probabilities equal 1, the doubled minimum is 2,
    clipped to 1 — i.e. no evidence to reject H0. With strict
    inequalities the same case would yield zero on both counts and
    spuriously produce the smallest representable p-value.

    Lower-bounded at ``2/B``; upper-bounded at 1.

    Parameters
    ----------
    precisions : ndarray
        Bootstrap stack of fitted precisions, shape ``(B, p, p)``
        single-population or ``(B, K, p, p)`` joint.

    Returns
    -------
    pvals : ndarray
        Same trailing shape (``(p, p)`` or ``(K, p, p)``). Diagonal
        entries are 1.0 (the diagonal is not an edge).
    """
    B, K, p = _validate_precision_stack(precisions)
    if K == 1:
        stack = precisions  # (B, p, p)
    else:
        stack = precisions  # (B, K, p, p)

    nonneg = (stack >= 0.0).sum(axis=0)
    nonpos = (stack <= 0.0).sum(axis=0)
    two_min = np.minimum(nonneg, nonpos)
    pvals = (2.0 * two_min / B).astype(np.float64)
    pvals = np.clip(pvals, 2.0 / B, 1.0)
    # Diagonal isn't a tested hypothesis — set to 1 so BH/Holm
    # exclude it from the family.
    if K == 1:
        np.fill_diagonal(pvals, 1.0)
    else:
        for k in range(K):
            np.fill_diagonal(pvals[k], 1.0)
    return pvals


def _upper_triangular_pairs(p: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Indices ``(i, j)`` with ``i < j`` flattened in row-major
    order. The family of hypotheses tested over the p×p precision
    matrix is the upper-triangular off-diagonal."""
    iu = np.triu_indices(p, k=1)
    return iu[0].astype(np.int64), iu[1].astype(np.int64)


def _bh_adjust(pvals_flat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini–Hochberg adjusted (step-up) p-values.

    Input is a 1D vector of m raw p-values. Returns the BH-adjusted
    p-values such that ``reject_at_q[i] = (q_adj[i] <= q)``.
    """
    m = pvals_flat.size
    order = np.argsort(pvals_flat)
    ranks = np.arange(1, m + 1, dtype=np.float64)
    sorted_p = pvals_flat[order]
    raw = sorted_p * m / ranks
    # Enforce monotonicity from the largest p-value downward.
    cummin = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.empty_like(pvals_flat)
    adj[order] = np.clip(cummin, 0.0, 1.0)
    return adj


def _holm_adjust(pvals_flat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Holm–Bonferroni adjusted (step-down) p-values."""
    m = pvals_flat.size
    order = np.argsort(pvals_flat)
    sorted_p = pvals_flat[order]
    multipliers = (m - np.arange(m, dtype=np.float64))
    raw = sorted_p * multipliers
    # Enforce monotonicity from the smallest p-value upward.
    cummax = np.maximum.accumulate(raw)
    adj = np.empty_like(pvals_flat)
    adj[order] = np.clip(cummax, 0.0, 1.0)
    return adj


def edge_fdr_threshold(
    bootstrap_result: object,
    *,
    fdr: float = 0.1,
    method: Literal["bh"] = "bh",
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Benjamini–Hochberg FDR on edges from a bootstrap result.

    Computes per-edge two-sided bootstrap p-values, then applies BH
    step-up across the ``p(p − 1)/2`` upper-triangular edges (or
    ``K · p(p − 1)/2`` for joint estimators — all populations are
    pooled into a single FDR family).

    Parameters
    ----------
    bootstrap_result : GraphicalBootstrap or ndarray
        Either a fitted :class:`~skein_glm.GraphicalBootstrap` object
        (uses its ``.precisions_`` attribute) or the bootstrap stack
        directly.

    fdr : float, default 0.1
        Target false discovery rate ``q`` in ``(0, 1)``.

    method : {"bh"}, default "bh"
        Only Benjamini–Hochberg is supported; the parameter is kept
        for forward compatibility.

    Returns
    -------
    out : dict
        Keys:

        - ``"pvalues"`` — raw two-sided bootstrap p-values, same
          shape as ``mean_`` (``(p, p)`` single, ``(K, p, p)`` joint).
        - ``"adjusted_pvalues"`` — BH-adjusted p-values, same shape.
        - ``"reject"`` — boolean mask of edges where
          ``adjusted_pvalues ≤ fdr``. Diagonal is False; symmetric.

    Notes
    -----
    The test is two-sided. Diagonal entries are not tested and not
    counted in the BH family; symmetry of ``Θ`` is respected by
    treating only the upper-triangle as the family and mirroring the
    rejection mask.
    """
    if not 0.0 < fdr < 1.0:
        raise ValueError(f"fdr must be in (0, 1); got {fdr}")
    if method != "bh":
        raise ValueError(f"method must be 'bh'; got {method!r}")

    precisions = _extract_precisions(bootstrap_result)
    B, K, p = _validate_precision_stack(precisions)
    pvals = edge_pvalues(precisions)

    iu_i, iu_j = _upper_triangular_pairs(p)
    if K == 1:
        flat = pvals[iu_i, iu_j]
        adj_flat = _bh_adjust(flat)
        adj = np.ones_like(pvals)
        adj[iu_i, iu_j] = adj_flat
        adj[iu_j, iu_i] = adj_flat
        np.fill_diagonal(adj, 1.0)
        reject = (adj <= fdr) & (np.eye(p) == 0)
    else:
        # Pool all (k, i, j) into one BH family.
        per_pop_flat = [pvals[k][iu_i, iu_j] for k in range(K)]
        all_flat = np.concatenate(per_pop_flat)
        all_adj = _bh_adjust(all_flat)
        adj = np.ones_like(pvals)
        for k in range(K):
            start = k * iu_i.size
            stop = start + iu_i.size
            adj[k, iu_i, iu_j] = all_adj[start:stop]
            adj[k, iu_j, iu_i] = all_adj[start:stop]
            np.fill_diagonal(adj[k], 1.0)
        reject = adj <= fdr
        for k in range(K):
            np.fill_diagonal(reject[k], False)

    return {
        "pvalues": pvals,
        "adjusted_pvalues": adj,
        "reject": reject,
    }


def edge_fwer_threshold(
    bootstrap_result: object,
    *,
    fwer: float = 0.05,
    method: Literal["bonferroni", "holm"] = "holm",
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Family-wise error control on edges via Bonferroni or Holm.

    Parameters
    ----------
    bootstrap_result : GraphicalBootstrap or ndarray
        See :func:`edge_fdr_threshold`.

    fwer : float, default 0.05
        Target family-wise error rate in ``(0, 1)``.

    method : {"bonferroni", "holm"}, default "holm"
        Bonferroni is the single-step correction (``p · m``); Holm
        is the uniformly-more-powerful step-down version.

    Returns
    -------
    out : dict
        Same keys as :func:`edge_fdr_threshold`. ``adjusted_pvalues``
        uses the chosen FWER method.
    """
    if not 0.0 < fwer < 1.0:
        raise ValueError(f"fwer must be in (0, 1); got {fwer}")
    if method not in ("bonferroni", "holm"):
        raise ValueError(f"method must be 'bonferroni' or 'holm'; got {method!r}")

    precisions = _extract_precisions(bootstrap_result)
    B, K, p = _validate_precision_stack(precisions)
    pvals = edge_pvalues(precisions)

    iu_i, iu_j = _upper_triangular_pairs(p)

    def _adjust(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        if method == "bonferroni":
            return np.clip(flat * flat.size, 0.0, 1.0)
        return _holm_adjust(flat)

    if K == 1:
        flat = pvals[iu_i, iu_j]
        adj_flat = _adjust(flat)
        adj = np.ones_like(pvals)
        adj[iu_i, iu_j] = adj_flat
        adj[iu_j, iu_i] = adj_flat
        np.fill_diagonal(adj, 1.0)
        reject = (adj <= fwer) & (np.eye(p) == 0)
    else:
        per_pop_flat = [pvals[k][iu_i, iu_j] for k in range(K)]
        all_flat = np.concatenate(per_pop_flat)
        all_adj = _adjust(all_flat)
        adj = np.ones_like(pvals)
        for k in range(K):
            start = k * iu_i.size
            stop = start + iu_i.size
            adj[k, iu_i, iu_j] = all_adj[start:stop]
            adj[k, iu_j, iu_i] = all_adj[start:stop]
            np.fill_diagonal(adj[k], 1.0)
        reject = adj <= fwer
        for k in range(K):
            np.fill_diagonal(reject[k], False)

    return {
        "pvalues": pvals,
        "adjusted_pvalues": adj,
        "reject": reject,
    }


def mb_stability_threshold(
    p_total: int,
    q_lambda: float,
    expected_false_positives: float,
) -> float:
    """Meinshausen–Bühlmann (2010) closed-form stability threshold.

    Given:

    - ``p_total`` — total size of the tested family (number of
      *unique* edges for a graph: ``p(p − 1)/2``);
    - ``q_lambda`` — average number of edges selected by the base
      estimator across the λ-grid swept by stability selection;
    - ``expected_false_positives`` — the desired upper bound on
      ``E[V]``, the expected number of falsely-stable edges;

    the MB bound gives the minimum stability-selection threshold
    ``π_thr ∈ (0.5, 1]`` that controls ``E[V] ≤ expected_false_positives``:

    .. math::

        \\pi_{thr} = \\frac{1}{2} + \\frac{q_\\Lambda^2}{2 \\cdot p_{total} \\cdot E[V]}

    Returns
    -------
    threshold : float
        Required selection probability threshold. If the formula
        yields > 1, the requested error bound is infeasible at the
        given ``q_lambda`` and a ``ValueError`` is raised.

    Notes
    -----
    The MB result requires a *symmetric distribution* of the base
    estimator under sub-sampling (mild — satisfied by lasso /
    elastic-net / MCP with sub-sampling fraction 1/2). The family
    size for *edges* in a graphical model is ``p(p − 1)/2``, not
    ``p`` — counting unique unordered pairs, since ``Θ`` is symmetric
    and the test is on whether ``Θ_ij = 0``.

    References
    ----------
    Meinshausen, N., & Bühlmann, P. (2010). "Stability selection."
    *J. R. Stat. Soc. B* 72(4): 417–473, Theorem 1.
    """
    if p_total < 1:
        raise ValueError(f"p_total must be ≥ 1; got {p_total}")
    if q_lambda <= 0:
        raise ValueError(f"q_lambda must be > 0; got {q_lambda}")
    if not 0.0 < expected_false_positives:
        raise ValueError(
            f"expected_false_positives must be > 0; got {expected_false_positives}"
        )
    thr = 0.5 + (q_lambda * q_lambda) / (2.0 * p_total * expected_false_positives)
    if thr > 1.0:
        raise ValueError(
            f"Infeasible: MB threshold = {thr:.3f} > 1 for "
            f"p_total={p_total}, q_lambda={q_lambda}, "
            f"E[V]={expected_false_positives}. Either raise the "
            f"acceptable E[V] or reduce q_lambda (e.g. by tightening "
            f"the λ-grid)."
        )
    return float(thr)


def _extract_precisions(arg: object) -> NDArray[np.float64]:
    """Accept either a fitted ``GraphicalBootstrap`` or a raw stack."""
    if hasattr(arg, "precisions_"):
        precisions = getattr(arg, "precisions_")
    else:
        precisions = arg
    arr = np.asarray(precisions, dtype=np.float64)
    if arr.ndim not in (3, 4):
        raise TypeError(
            "expected a fitted GraphicalBootstrap or a 3D / 4D "
            f"precision-stack array; got ndim={arr.ndim}"
        )
    return arr


__all__ = [
    "edge_pvalues",
    "edge_fdr_threshold",
    "edge_fwer_threshold",
    "mb_stability_threshold",
]
