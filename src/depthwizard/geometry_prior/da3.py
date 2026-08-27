from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput

DeviceName = Literal["auto", "cuda", "mps", "cpu"]


def depth_to_relative_height(
    depth: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Convert camera depth into a stable dimensionless relative-height convention.

    In nadir/near-nadir optical imagery, smaller camera depth corresponds to a higher surface.
    The returned field is monotonic with height and deliberately has no metric units.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d)
    if not np.any(valid):
        raise ValueError("DA3 returned no finite depth values")
    lo, hi = np.percentile(d[valid], [low_percentile, high_percentile])
    scale = max(float(hi - lo), 1e-6)
    clipped = np.clip(d, lo, hi)
    relative_height = (hi - clipped) / scale
    relative_height[~valid] = np.nan
    return relative_height.astype(np.float32)


@dataclass
class DA3MonocularPrior(GeometryPrior):
    """Depth Anything 3 monocular geometry prior adapter.

    This adapter intentionally exposes relative height only. Absolute geospatial elevation remains
    the responsibility of DepthWizard's evidence-calibration subsystem.
    """

    model_source: str | Path = "depth-anything/DA3MONO-LARGE"
    device: DeviceName = "auto"
    _model: Any | None = None
    _resolved_device: str | None = None

    def _load(self) -> tuple[Any, Any, str]:
        try:
            torch: Any = importlib.import_module("torch")
            da3_api: Any = importlib.import_module("depth_anything_3.api")
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            raise RuntimeError(
                "Depth Anything 3 runtime import failed before model loading: "
                f"{type(exc).__name__}: {exc}; missing_module={missing!r}. "
                "The packaged runtime is incomplete or the pinned DA3 environment is unavailable."
            ) from exc

        try:
            depth_anything_3: Any = da3_api.DepthAnything3
        except AttributeError as exc:
            raise RuntimeError(
                "Depth Anything 3 runtime import completed, but depth_anything_3.api does not expose "
                "DepthAnything3. The pinned DA3 API contract is incompatible with this runtime."
            ) from exc

        if self._model is None:
            if self.device == "auto":
                if torch.cuda.is_available():
                    resolved = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved = "mps"
                else:
                    resolved = "cpu"
            else:
                resolved = self.device
            model = depth_anything_3.from_pretrained(str(self.model_source))
            model = model.to(device=torch.device(resolved))
            model.eval()
            self._model = model
            self._resolved_device = resolved
        return self._model, torch, self._resolved_device or "cpu"

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        rgb = np.asarray(rgb_normalized, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("rgb_normalized must have shape HxWx3")
        if float(np.nanmin(rgb)) < 0.0 or float(np.nanmax(rgb)) > 1.0:
            raise ValueError("rgb_normalized must be in [0, 1]")

        model, torch, device = self._load()
        image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        with torch.inference_mode():
            prediction = model.inference([image])

        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        conf = getattr(prediction, "conf", None)
        confidence = np.asarray(conf[0], dtype=np.float32) if conf is not None else None

        if depth.shape != rgb.shape[:2]:
            depth_tensor = torch.from_numpy(depth)[None, None].float()
            depth = torch.nn.functional.interpolate(
                depth_tensor,
                size=rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy().astype(np.float32)
            if confidence is not None:
                conf_tensor = torch.from_numpy(confidence)[None, None].float()
                confidence = torch.nn.functional.interpolate(
                    conf_tensor,
                    size=rgb.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu().numpy().astype(np.float32)

        relative_height = depth_to_relative_height(depth)
        return GeometryPriorOutput(
            relative_height=relative_height,
            confidence=confidence,
            model_id="DA3MONO-LARGE",
            metadata={
                "model_source": str(self.model_source),
                "device": device,
                "output_semantics": "dimensionless_relative_surface_height",
                "license": "Apache-2.0",
            },
        )
