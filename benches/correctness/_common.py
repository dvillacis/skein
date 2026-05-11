"""Shared utilities for `benches/correctness/` cross-package checks.

The metrics here are intentionally tolerant — MCP / SCAD are nonconvex,
so different solvers can converge to different local minima even at the
same tolerance. We report agreement rather than assert it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class PairAgreement:
    """Pairwise agreement between two coefficient paths along a shared λ-grid.

    Per-λ arrays (`jaccard`, `sign_agreement`, `rel_l2`,
    `meaningful_mask`) have shape `(n_lambdas,)`; the headline summary
    fields are scalars.

    `meaningful_mask[k]` is `True` when `max(‖a_k‖, ‖b_k‖) >
    norm_threshold` — at near-zero solutions the *relative* L2 metric
    divides near-zero by near-zero and the value is uninformative
    (e.g. at λ_max where both packages output essentially noise). The
    headline `mean_rel_l2_meaningful` / `worst_rel_l2_meaningful`
    aggregate only over the meaningful subset, so the bench summary
    reflects real coefficient disagreement rather than metric
    artifacts.

    Active-set (`jaccard`) and `sign_agreement` are aggregated over
    the full path — they are well-defined even when both solutions are
    zero (we return 1.0 by convention).
    """

    a: str
    b: str
    n_lambdas: int
    jaccard: NDArray[np.float64]
    sign_agreement: NDArray[np.float64]
    rel_l2: NDArray[np.float64]
    meaningful_mask: NDArray[np.bool_]
    norm_threshold: float
    mean_jaccard: float
    mean_sign_agreement: float
    mean_rel_l2: float
    worst_rel_l2: float
    mean_rel_l2_meaningful: float
    worst_rel_l2_meaningful: float
    n_lambdas_meaningful: int
    n_lambdas_perfect_support: int

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "n_lambdas": self.n_lambdas,
            "norm_threshold": self.norm_threshold,
            "mean_jaccard": self.mean_jaccard,
            "mean_sign_agreement": self.mean_sign_agreement,
            "mean_rel_l2": self.mean_rel_l2,
            "worst_rel_l2": self.worst_rel_l2,
            "mean_rel_l2_meaningful": self.mean_rel_l2_meaningful,
            "worst_rel_l2_meaningful": self.worst_rel_l2_meaningful,
            "n_lambdas_meaningful": self.n_lambdas_meaningful,
            "n_lambdas_perfect_support": self.n_lambdas_perfect_support,
            "per_lambda": {
                "jaccard": self.jaccard.tolist(),
                "sign_agreement": self.sign_agreement.tolist(),
                "rel_l2": self.rel_l2.tolist(),
                "meaningful_mask": self.meaningful_mask.tolist(),
            },
        }


def _active(coef: NDArray[np.float64], atol: float) -> NDArray[np.bool_]:
    return np.abs(coef) > atol


def pair_agreement(
    a_name: str,
    a_path: NDArray[np.float64],
    b_name: str,
    b_path: NDArray[np.float64],
    atol: float = 1e-8,
    rel_norm_threshold: float = 1e-2,
) -> PairAgreement:
    """Compute per-λ agreement between two coefficient paths.

    Both paths must have shape `(n_lambdas, p)`. `atol` is the cutoff
    for treating a coefficient as nonzero when building the active
    set.

    `rel_norm_threshold` gates the relative-L2 *summary* numbers, on
    a path-relative scale: the per-λ "norm of interest" is
    `n_k = max(‖a_k‖, ‖b_k‖)`, and index `k` is flagged meaningful
    when `n_k > rel_norm_threshold · max_k n_k`. The default of 1 %
    masks the near-`λ_max` indices where the coefficient norm is a
    fraction of the path peak — there, relative-L2 divides one tiny
    number by another and the metric is uninformative. Indices in
    the interior and tail are kept. A relative threshold is
    scale-invariant; an absolute one would need to be re-tuned per
    problem.
    """
    if a_path.shape != b_path.shape:
        raise ValueError(f"shape mismatch: {a_name}={a_path.shape} {b_name}={b_path.shape}")
    n_lambdas, _p = a_path.shape

    jaccard = np.empty(n_lambdas)
    sign_agreement = np.empty(n_lambdas)
    rel_l2 = np.empty(n_lambdas)
    per_lambda_norm = np.empty(n_lambdas)
    perfect_support = 0

    for k in range(n_lambdas):
        a_k = a_path[k]
        b_k = b_path[k]
        act_a = _active(a_k, atol)
        act_b = _active(b_k, atol)
        union = int(np.count_nonzero(act_a | act_b))
        inter = int(np.count_nonzero(act_a & act_b))
        jaccard[k] = 1.0 if union == 0 else inter / union
        if union == 0 or np.array_equal(act_a, act_b):
            perfect_support += 1
        # Sign agreement uses np.sign (0 for zero coefs treated as its own class).
        sign_agreement[k] = float(np.mean(np.sign(a_k) == np.sign(b_k)))
        norm_a = float(np.linalg.norm(a_k))
        norm_b = float(np.linalg.norm(b_k))
        denom = max(norm_a, norm_b, 1e-12)
        rel_l2[k] = float(np.linalg.norm(a_k - b_k) / denom)
        per_lambda_norm[k] = max(norm_a, norm_b)

    peak_norm = float(np.max(per_lambda_norm))
    norm_threshold = peak_norm * rel_norm_threshold
    meaningful_mask = per_lambda_norm > norm_threshold

    n_meaningful = int(np.count_nonzero(meaningful_mask))
    if n_meaningful > 0:
        rel_l2_meaningful = rel_l2[meaningful_mask]
        mean_rel_l2_meaningful = float(np.mean(rel_l2_meaningful))
        worst_rel_l2_meaningful = float(np.max(rel_l2_meaningful))
    else:
        mean_rel_l2_meaningful = 0.0
        worst_rel_l2_meaningful = 0.0

    return PairAgreement(
        a=a_name,
        b=b_name,
        n_lambdas=n_lambdas,
        jaccard=jaccard,
        sign_agreement=sign_agreement,
        rel_l2=rel_l2,
        meaningful_mask=meaningful_mask,
        norm_threshold=norm_threshold,
        mean_jaccard=float(np.mean(jaccard)),
        mean_sign_agreement=float(np.mean(sign_agreement)),
        mean_rel_l2=float(np.mean(rel_l2)),
        worst_rel_l2=float(np.max(rel_l2)),
        mean_rel_l2_meaningful=mean_rel_l2_meaningful,
        worst_rel_l2_meaningful=worst_rel_l2_meaningful,
        n_lambdas_meaningful=n_meaningful,
        n_lambdas_perfect_support=perfect_support,
    )


def write_results(scenario: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{scenario}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def format_summary_table(pairs: list[PairAgreement]) -> str:
    """Pretty-print a one-line-per-pair summary table.

    The `mean_relL2*` / `worst_relL2*` columns aggregate only over
    indices where the solution norm exceeds `norm_threshold` (default
    `1e-2`) — that filters out the near-`λ_max` indices where the
    relative metric is uninformative noise. Use the raw `.rel_l2`
    array if you want the unfiltered values.
    """
    header = (
        f"{'pair':<24} {'mean_jacc':>10} {'mean_sign':>10} "
        f"{'mean_relL2*':>12} {'worst_relL2*':>13} "
        f"{'meaningful':>11} {'perfect_supp/K':>16}"
    )
    rows = [header, "-" * len(header)]
    for p in pairs:
        rows.append(
            f"{p.a + ' vs ' + p.b:<24} "
            f"{p.mean_jaccard:>10.4f} "
            f"{p.mean_sign_agreement:>10.4f} "
            f"{p.mean_rel_l2_meaningful:>12.4f} "
            f"{p.worst_rel_l2_meaningful:>13.4f} "
            f"{p.n_lambdas_meaningful:>5d}/{p.n_lambdas:<5d} "
            f"{p.n_lambdas_perfect_support:>7d}/{p.n_lambdas:<8d}"
        )
    return "\n".join(rows)
