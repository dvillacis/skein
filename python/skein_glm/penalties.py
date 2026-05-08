"""Python-side penalty ABCs.

These mirror the Rust traits in `skein-core::penalty` so that per-paper uv
projects can prototype custom penalties in Python before (optionally)
porting them to Rust. The base `skein` library is the extension surface;
papers extend it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Penalty(ABC):
    """Separable scalar penalty p(β) = Σ_j w_j · ψ(β_j)."""

    @abstractmethod
    def value(self, beta: NDArray[np.float64]) -> float: ...

    @abstractmethod
    def prox_coord(self, j: int, z: float, step: float) -> float: ...

    @property
    @abstractmethod
    def weights(self) -> NDArray[np.float64]: ...


class GroupPenalty(ABC):
    """Block-separable penalty over a group partition."""

    @abstractmethod
    def value(self, beta: NDArray[np.float64], groups: list[NDArray[np.intp]]) -> float: ...

    @abstractmethod
    def prox_group(self, g: int, block: NDArray[np.float64], step: float) -> NDArray[np.float64]: ...

    @property
    @abstractmethod
    def weights(self) -> NDArray[np.float64]: ...
