"""Common runner ABI for the M9 bench suite.

Each runner exposes a `fit(problem, *, penalty, lambda_grid, tol, **kwargs)`
that returns a `RunResult`. The dispatch layer in `benches/run.py` does
not import any runner unconditionally — runners that fail to import (the
package isn't installed) are skipped with a logged warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


PenaltyName = Literal[
    "lasso",
    "elastic_net",
    "mcp",
    "scad",
    "group_lasso",
    "group_mcp",
    "group_scad",
    "sparse_group_lasso",
    "sparse_group_mcp",
    "sparse_group_scad",
    "glasso",
    "glasso_mcp",
]


@dataclass(frozen=True)
class RunResult:
    package: str
    version: str
    fit_time_s: float
    n_iter: int | None
    final_obj: float | None
    active_set_size: int
    coef_path: NDArray[np.float64] | None = None
    intercept_path: NDArray[np.float64] | None = None
    extra: dict[str, object] = field(default_factory=dict)


class Runner(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def fit(
        self,
        problem,  # benches.problems.Problem
        *,
        penalty: PenaltyName,
        lambda_grid: NDArray[np.float64],
        tol: float,
        **kwargs: object,
    ) -> RunResult: ...
