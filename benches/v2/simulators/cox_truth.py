"""Cox proportional hazards simulator with controllable censoring.

Generates (time, event) under an exponential baseline hazard:
    T ~ Exp(λ₀ exp(x·β))
    C ~ Exp(rate_c)               (independent right-censoring)
    obs = min(T, C),  status = 1[T ≤ C]

The censoring rate knob `target_censoring ∈ [0, 1)` calibrates
`rate_c` so the empirical censoring rate matches the target.

Problem.y holds the observed time; Problem.meta["event"] holds the
binary status (1 = event observed, 0 = right-censored). The R runner
repacks these into Surv(time, status) on the R side.
"""
from __future__ import annotations

import math

import numpy as np

from benches.problems import Problem
from benches.v2.simulators._design import design, sparse_beta


def make(
    *,
    n: int, p: int,
    seed: int = 0,
    signal_scale: float = 0.5,
    sparsity_k: float = 1.0,
    target_censoring: float = 0.3,
    baseline_rate: float = 0.1,
    corr_kind: str = "iid",
    corr_rho: float = 0.0,
) -> Problem:
    if not (0.0 <= target_censoring < 1.0):
        raise ValueError(f"target_censoring must be in [0, 1); got {target_censoring}")
    rng = np.random.default_rng(seed)
    x = design(n, p, rng=rng, kind=corr_kind, rho=corr_rho)
    k_active = max(1, int(round(sparsity_k * math.sqrt(p))))
    beta = signal_scale * sparse_beta(p, rng=rng, k_active=k_active)

    eta = x @ beta
    # Cap |eta| so exp(eta) doesn't underflow rate to 0 (creates infinite times).
    eta = np.clip(eta, -8.0, 8.0)
    rate_event = baseline_rate * np.exp(eta)
    t_event = rng.exponential(1.0 / np.maximum(rate_event, 1e-12))

    # Pick rate_c so the empirical censoring rate matches target. With
    # independent exponentials, P(C < T) = rate_c / (rate_c + rate_event).
    # Solve for the rate that makes the mean match: simpler — just
    # bisect on the empirical rate.
    if target_censoring == 0.0:
        rate_c = 0.0
        t_censor = np.full(n, np.inf)
    else:
        lo, hi = 1e-8, 1e3 * float(np.max(rate_event))
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            rate_c = mid
            t_c = rng.exponential(1.0 / rate_c, size=n)
            cens_rate = float(np.mean(t_c < t_event))
            if abs(cens_rate - target_censoring) < 0.01:
                t_censor = t_c
                break
            if cens_rate < target_censoring:
                lo = mid
            else:
                hi = mid
        else:
            t_censor = rng.exponential(1.0 / mid, size=n)
            rate_c = mid

    obs_time = np.minimum(t_event, t_censor)
    status = (t_event <= t_censor).astype(np.int64)
    return Problem(
        x=x, y=obs_time, beta_true=beta, family="cox",
        meta={
            "simulator": "cox_truth",
            "event": status,
            "k_active": k_active, "signal_scale": signal_scale,
            "target_censoring": target_censoring,
            "empirical_censoring": float(1.0 - status.mean()),
            "baseline_rate": baseline_rate, "rate_c": float(rate_c),
            "corr_kind": corr_kind, "corr_rho": corr_rho,
        },
    )
