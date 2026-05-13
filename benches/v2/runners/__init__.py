"""Comparator runner adapters.

Phase A re-exports the existing benches/runners/* protocol; Phase B
ports them in and adds the feather-IPC R transport, plus lifelines
and statsmodels.
"""
from __future__ import annotations

# Re-export the existing protocol so v2 scenarios share the ABI.
from benches.runners import PenaltyName, RunResult, Runner  # noqa: F401
