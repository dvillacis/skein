"""Copy this file when adding a new scenario.

A scenario is responsible for:
  1. Building the Problem (delegates to a simulator)
  2. Picking the λ-grid for the requested regime
  3. Invoking the runner via the shared timing harness
  4. Computing optional accuracy metrics (agreement / recovery / deviance)
  5. Returning a dict that becomes one JSONL row

The cell driver (benches.v2.report._run_cell) takes care of:
  - dispatching to the right runner module
  - 1 warm-up + N timed trials
  - environment capture
  - JSONL serialization
"""
from __future__ import annotations

from typing import TypedDict


class ScenarioSpec(TypedDict):
    datafit: str          # "gaussian" | "logistic" | "poisson" | "cox" | "gaussian_inv_cov"
    penalty: str          # PenaltyName from runners
    family_module: str    # qualified name in benches.problems or benches.v2.simulators


# Each scenario module exposes:
#   SPEC: ScenarioSpec
#   def make_problem(size: str, seed: int) -> Problem: ...
