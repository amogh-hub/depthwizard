from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from depthwizard.geometry_prior.da3 import DA3MonocularPrior

ROOT = Path(__file__).resolve().parents[1]
DA3_DIR = ROOT / ".vendor" / "depth-anything-3"
OUT_DIR = ROOT / "artifacts" / "smoke" / "da3"
MODEL_SOURCE = "depth-anything/DA3MONO-LARGE"


def find_sample() -> Path:
    roots = [DA3_DIR / "assets" / "examples", DA3_DIR / "assets"]
    suffixes = {".png", ".jpg", ".jpeg"}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                return path
    raise SystemExit(
        "No DA3 example image found. Run `make da3-setup` first and confirm the pinned vendor checkout exists."
    )


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    source = find_sample()
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32) / 255.0

    prior = DA3MonocularPrior(model_source=MODEL_SOURCE, device="auto")

    started = time.perf_counter()
    output = prior.infer(rgb)
    elapsed = time.perf_counter() - started

    relative = np.asarray(output.relative_height, dtype=np.float32)
    finite = np.isfinite(relative)
    if not np.any(finite):
        raise SystemExit("DA3 smoke test failed: no finite relative-height values were produced")
    if relative.shape != rgb.shape[:2]:
        raise SystemExit(
            f"DA3 smoke test failed: output shape {relative.shape} does not match input {rgb.shape[:2]}"
        )

    finite_values = relative[finite]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if minimum < -1e-5 or maximum > 1.00001:
        raise SystemExit(
            f"DA3 smoke test failed: relative-height range [{minimum}, {maximum}] is outside [0, 1]"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "relative_height.npy", relative)

    if output.confidence is not None:
        np.save(OUT_DIR / "confidence.npy", np.asarray(output.confidence, dtype=np.float32))

    lo, hi = np.nanpercentile(relative, [1.0, 99.0])
    scale = max(float(hi - lo), 1e-6)
    visual = np.clip((relative - lo) / scale, 0.0, 1.0)
    visual_u8 = np.nan_to_num(visual, nan=0.0)
    visual_u8 = (visual_u8 * 255.0).astype(np.uint8)
    Image.fromarray(visual_u8, mode="L").save(OUT_DIR / "relative_height_vis.png")

    report = {
        "status": "PASS",
        "model_id": output.model_id,
        "model_source": MODEL_SOURCE,
        "device": output.metadata.get("device", "unknown"),
        "source_image": str(source.relative_to(ROOT)),
        "input_shape": list(rgb.shape),
        "output_shape": list(relative.shape),
        "finite_fraction": float(np.mean(finite)),
        "relative_height_min": minimum,
        "relative_height_max": maximum,
        "confidence_present": output.confidence is not None,
        "wall_time_seconds": elapsed,
    }
    (OUT_DIR / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard DA3 smoke inference: PASS")
    print("Model:", output.model_id)
    print("Device:", report["device"])
    print("Input:", source)
    print("Output shape:", tuple(relative.shape))
    print(f"Relative-height range: {minimum:.6f} .. {maximum:.6f}")
    print(f"Wall time: {elapsed:.2f} s")
    print("Artifacts:", OUT_DIR)


if __name__ == "__main__":
    main()
