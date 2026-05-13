"""Per-λ cross-package agreement metrics.

Lifted from benches/correctness/_common.py so v2 cells can compute
agreement against any comparator without depending on the legacy
correctness suite.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _meaningful_mask(coef: NDArray[np.float64], rtol: float = 1e-6) -> NDArray[np.bool_]:
    """True where |coef| > rtol * max|coef|. Skips near-zero numerical noise."""
    scale = float(np.max(np.abs(coef)))
    return np.abs(coef) > rtol * scale if scale > 0 else np.zeros_like(coef, dtype=bool)


def jaccard(a: NDArray[np.float64], b: NDArray[np.float64], rtol: float = 1e-6) -> float:
    """Jaccard index between the meaningful supports of two coefficient vectors."""
    sa, sb = _meaningful_mask(a, rtol), _meaningful_mask(b, rtol)
    inter = int(np.sum(sa & sb))
    union = int(np.sum(sa | sb))
    return 1.0 if union == 0 else inter / union


def sign_agreement(a: NDArray[np.float64], b: NDArray[np.float64], rtol: float = 1e-6) -> float:
    """Fraction of meaningful indices where sign matches."""
    mask = _meaningful_mask(a, rtol) | _meaningful_mask(b, rtol)
    n = int(np.sum(mask))
    if n == 0:
        return 1.0
    return float(np.sum(np.sign(a[mask]) == np.sign(b[mask]))) / n


def rel_l2(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """‖a - b‖₂ / max(‖a‖₂, ‖b‖₂, ε)."""
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b)) / denom


def per_lambda(
    path_a: NDArray[np.float64],
    path_b: NDArray[np.float64],
    rtol: float = 1e-6,
) -> dict[str, list[float]]:
    """Compute the three agreement metrics for every λ along two coefficient paths.

    path_a, path_b: (n_lambdas, p) arrays.
    """
    if path_a.shape != path_b.shape:
        raise ValueError(f"path shapes differ: {path_a.shape} vs {path_b.shape}")
    return {
        "jaccard":  [jaccard(a, b, rtol) for a, b in zip(path_a, path_b)],
        "sign":     [sign_agreement(a, b, rtol) for a, b in zip(path_a, path_b)],
        "rel_l2":   [rel_l2(a, b) for a, b in zip(path_a, path_b)],
    }
