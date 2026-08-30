from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput

DeviceName = Literal["auto", "cuda", "mps", "cpu"]

DA3_MODEL_ID = "DA3MONO-LARGE"
DA3_MODEL_SOURCE = "depth-anything/DA3MONO-LARGE"
DA3_HF_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
DA3_CHECKPOINT_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
DA3_UPSTREAM_SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


def depth_to_affine_height_evidence(depth: np.ndarray) -> np.ndarray:
    """Invert camera depth without applying tile-local normalization.

    The tiled production pipeline must preserve the model's affine evidence until neighboring
    predictions have been harmonized. Percentile-normalizing each inference tile independently
    destroys inter-tile low-frequency information and can imprint the tile grid into the final
    surface. ``-depth`` keeps the required height ordering (nearer surface = higher evidence) while
    leaving scale/offset available to the scene scaffold and later normalization.
    """
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d)
    if not np.any(valid):
        raise ValueError("DA3 returned no finite depth values")
    evidence = -d
    evidence = evidence.astype(np.float32, copy=False)
    evidence[~valid] = np.nan
    return evidence


def depth_to_relative_height(
    depth: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Convert a complete depth field into a stable dimensionless relative-height convention.

    This helper is appropriate when ``depth`` represents one complete scene. The tiled production
    adapter deliberately does *not* call it per tile; production first assembles affine-preserving
    evidence and performs one scene-global normalization downstream.
    """
    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
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
    the responsibility of DepthWizard's evidence-calibration subsystem. The production Hub source
    is revision-pinned so a moving ``main`` branch can never silently change scientific behavior.
    """

    model_source: str | Path = DA3_MODEL_SOURCE
    device: DeviceName = "auto"
    _model: Any | None = None
    _resolved_device: str | None = None
    _resolved_model_revision: str | None = None

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

            source = str(self.model_source)
            if source == DA3_MODEL_SOURCE:
                model = depth_anything_3.from_pretrained(source, revision=DA3_HF_REVISION)
            else:
                # Explicit local/custom sources remain supported for controlled tests and research.
                # They do not inherit the production Hub revision because that would be misleading.
                model = depth_anything_3.from_pretrained(source)

            loaded_revision = getattr(model, "_commit_hash", None)
            if source == DA3_MODEL_SOURCE and loaded_revision is not None:
                loaded_revision = str(loaded_revision)
                if loaded_revision != DA3_HF_REVISION:
                    raise RuntimeError(
                        "Depth Anything 3 resolved an unexpected model revision: "
                        f"{loaded_revision}; expected {DA3_HF_REVISION}"
                    )
                self._resolved_model_revision = loaded_revision
            elif source == DA3_MODEL_SOURCE:
                # The revision argument still pins Hub resolution even when a particular Hub mixin
                # version does not expose its private commit-hash field after loading.
                self._resolved_model_revision = DA3_HF_REVISION

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

        relative_height = depth_to_affine_height_evidence(depth)
        return GeometryPriorOutput(
            relative_height=relative_height,
            confidence=confidence,
            model_id=DA3_MODEL_ID,
            metadata={
                "model_source": str(self.model_source),
                "model_revision": self._resolved_model_revision or "custom_or_local_source",
                "checkpoint_sha256": (
                    DA3_CHECKPOINT_SHA256
                    if str(self.model_source) == DA3_MODEL_SOURCE
                    else "custom_or_local_source"
                ),
                "upstream_source_commit": DA3_UPSTREAM_SOURCE_COMMIT,
                "device": device,
                "output_semantics": "affine_relative_surface_height_evidence",
                "scene_normalize_relative_height": True,
                "scene_global_scaffold": True,
                "confidence_semantics": "model_native_not_probability_calibrated",
                "license": "Apache-2.0",
            },
        )
