from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from depthwizard.height_model.losses import compute_height_losses
from depthwizard.height_model.model import DepthWizardHeightModel

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "smoke" / "height-model"


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(26175)
    device = resolve_device()
    model = DepthWizardHeightModel().to(device).train()

    rgb = torch.rand(1, 3, 256, 256, device=device)
    geometry = torch.rand(1, 1, 256, 256, device=device)
    target = torch.rand(1, 1, 256, 256, device=device)
    valid = torch.ones_like(target, dtype=torch.bool)
    semantic = torch.randint(0, 5, (1, 256, 256), device=device)
    gsd = torch.tensor([0.5], device=device)

    started = time.perf_counter()
    output = model(rgb, geometry, gsd_m=gsd)
    losses = compute_height_losses(output, target, valid, semantic_target=semantic)
    losses.total.backward()
    elapsed = time.perf_counter() - started

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    payload = {
        "status": "PASS",
        "device": str(device),
        "shape": [1, 3, 256, 256],
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "forward_backward_seconds": elapsed,
        "outputs": {
            "relative_height": list(output.relative_height.shape),
            "uncertainty": list(output.uncertainty.shape),
            "semantic_logits": list(output.semantic_logits.shape),
            "height_bin_logits": list(output.height_bin_logits.shape),
            "normals": list(output.normals.shape),
            "boundary_probability": list(output.boundary_probability.shape),
        },
        "loss": float(losses.total.detach().cpu()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "smoke_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("DepthWizard remote-sensing height model smoke: PASS")
    print(f"Device: {device}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Forward+backward: {elapsed:.2f} s")
    print(f"Loss: {payload['loss']:.5f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
