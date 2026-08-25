"""RDAH-Net checkpoint loading and inference utilities.

The released RDAH-Net model predicts normalized surface height from an orthophoto plus a
Depth Anything v2 relative-depth prior. This adapter keeps the upstream checkpoint semantics
while adding deterministic device selection and validation suitable for DepthWizard benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from depthwizard.baselines.rdah_model import HeightPredTransformer

DeviceName = Literal["auto", "cuda", "mps", "cpu"]

_IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class RDAHInferenceResult:
    height_like: np.ndarray
    device: str
    checkpoint: Path


def resolve_device(device: DeviceName = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise ValueError("RDAH checkpoint does not contain a state dictionary")

    state: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        clean = key.removeprefix("module.")
        state[clean] = value
    if not state:
        raise ValueError("RDAH checkpoint state dictionary contains no tensors")
    return state


def load_rdah_model(
    checkpoint: str | Path,
    *,
    device: DeviceName = "auto",
) -> tuple[HeightPredTransformer, str]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    resolved = resolve_device(device)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = _checkpoint_state(payload)
    model = HeightPredTransformer()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "RDAH checkpoint is incompatible with the released architecture: "
            f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    model = model.to(torch.device(resolved)).eval()
    return model, resolved


def preprocess_rdah_inputs(
    rgb: np.ndarray,
    relative_depth: np.ndarray,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(rgb)
    depth = np.asarray(relative_depth, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise ValueError("relative_depth must be HxW and match rgb")
    if image.shape[0] % 128 or image.shape[1] % 128:
        raise ValueError("RDAH input height and width must be divisible by 128")
    if not np.all(np.isfinite(depth)):
        raise ValueError("RDAH relative-depth prior must be finite")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        finite = np.isfinite(image)
        image = np.where(finite, image, 0.0)
        if float(np.max(image)) <= 1.0:
            image = image * 255.0
        else:
            low = np.percentile(image, 1, axis=(0, 1), keepdims=True)
            high = np.percentile(image, 99, axis=(0, 1), keepdims=True)
            image = (image - low) / np.maximum(high - low, 1e-6) * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    normalized = image.astype(np.float32) / 255.0
    normalized = (normalized - _IMAGE_MEAN) / _IMAGE_STD
    image_tensor = torch.from_numpy(np.moveaxis(normalized, -1, 0).copy())[None]
    depth_tensor = torch.from_numpy(depth.copy())[None, None]
    target_device = torch.device(device)
    return depth_tensor.to(target_device), image_tensor.to(target_device)


def infer_rdah(
    model: HeightPredTransformer,
    rgb: np.ndarray,
    relative_depth: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    depth_tensor, image_tensor = preprocess_rdah_inputs(
        rgb,
        relative_depth,
        device=device,
    )
    with torch.inference_mode():
        prediction = model(depth_tensor, image_tensor)
    output = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
    if output.shape != relative_depth.shape:
        raise RuntimeError(
            f"RDAH output shape {output.shape} does not match input {relative_depth.shape}"
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("RDAH produced non-finite output")
    return output
