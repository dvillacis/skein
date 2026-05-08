"""Python-side datafit ABC. Mirrors `skein-core::datafit::Datafit`."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Datafit(ABC):
    @abstractmethod
    def value(self, residual: NDArray[np.float64]) -> float: ...

    @abstractmethod
    def init_residual(
        self, x: NDArray[np.float64], beta: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def coord_lipschitz(self, x: NDArray[np.float64], j: int) -> float: ...

    @property
    def sample_weights(self) -> NDArray[np.float64] | None:
        return None
