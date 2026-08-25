from __future__ import annotations

from enum import Enum


class ProcessingStage(str, Enum):
    INGEST = "ingest"
    PREPROCESS = "preprocess"
    GEOMETRY = "geometry"
    REFINEMENT = "refinement"
    CALIBRATION = "calibration"
    EXPORT = "export"
    VALIDATION = "validation"
    MESH = "mesh"
    COMPLETE = "complete"
