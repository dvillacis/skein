"""Cox-Lasso scenario. The simulator emits event status in problem.meta;
runners that need it (lifelines, glmnet via R) pull it from there.
"""
from __future__ import annotations

from benches.v2.simulators import cox_truth

SPEC = {
    "datafit": "cox",
    "penalty": "lasso",
    "family_module": "benches.v2.simulators.cox_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return cox_truth.make(n=n, p=p, seed=seed)
