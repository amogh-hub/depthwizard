from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometryPriorOutput:
    relative_height: np.ndarray
    confidence: np.ndarray | None
    model_id: str
    metadata: dict[str, str | float | int | bool]


class GeometryPrior(ABC):
    """Permanent model boundary for pretrained monocular geometry backbones."""

    @abstractmethod
    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        """Infer a relative surface-height convention suitable for positive-scale calibration."""
        raise NotImplementedError
