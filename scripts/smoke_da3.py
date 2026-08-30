from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.pipeline.geometry import normalize_relative_height_scene

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

    evidence = np.asarray(output.relative_height, dtype=np.float32)
    finite = np.isfinite(evidence)
    if not np.any(finite):
        raise SystemExit("DA3 smoke test failed: no finite affine height evidence was produced")
    if evidence.shape != rgb.shape[:2]:
        raise SystemExit(
            f"DA3 smoke test failed: output shape {evidence.shape} does not match input {rgb.shape[:2]}"
        )
    if output.metadata.get("scene_normalize_relative_height") is not True:
        raise SystemExit(
            "DA3 smoke test failed: production adapter did not request scene-global normalization"
        )
    if output.metadata.get("output_semantics") != "affine_relative_surface_height_evidence":
        raise SystemExit("DA3 smoke test failed: unexpected affine-evidence semantics")

    relative = normalize_relative_height_scene(evidence)
    relative_finite = relative[np.isfinite(relative)]
    minimum = float(np.min(relative_finite))
    maximum = float(np.max(relative_finite))
    if minimum < -1e-5 or maximum > 1.00001:
        raise SystemExit(
            "DA3 smoke test failed: scene-global relative-height normalization is outside [0, 1]"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "affine_height_evidence.npy", evidence)
    np.save(OUT_DIR / "relative_height.npy", relative)

    if output.confidence is not None:
        np.save(OUT_DIR / "confidence.npy", np.asarray(output.confidence, dtype=np.float32))

    visual_u8 = np.nan_to_num(relative, nan=0.0)
    visual_u8 = (np.clip(visual_u8, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(visual_u8, mode="L").save(OUT_DIR / "relative_height_vis.png")

    evidence_values = evidence[finite]
    evidence_minimum = float(np.min(evidence_values))
    evidence_maximum = float(np.max(evidence_values))
    report: dict[str, object] = {
        "status": "PASS",
        "model_id": output.model_id,
        "model_source": MODEL_SOURCE,
        "device": output.metadata.get("device", "unknown"),
        "source_image": str(source.relative_to(ROOT)),
        "input_shape": list(rgb.shape),
        "output_shape": list(evidence.shape),
        "finite_fraction": float(np.mean(finite)),
        "affine_height_evidence_min": evidence_minimum,
        "affine_height_evidence_max": evidence_maximum,
        "relative_height_min": minimum,
        "relative_height_max": maximum,
        "scene_normalization_required": True,
        "confidence_present": output.confidence is not None,
        "wall_time_seconds": elapsed,
    }
    (OUT_DIR / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard DA3 smoke inference: PASS")
    print("Model:", output.model_id)
    print("Device:", output.metadata.get("device", "unknown"))
    print("Input:", source)
    print("Output shape:", tuple(evidence.shape))
    print(
        "Affine height evidence range:",
        f"{evidence_minimum:.6f} .. {evidence_maximum:.6f}",
    )
    print(f"Scene-global relative-height range: {minimum:.6f} .. {maximum:.6f}")
    print(f"Wall time: {elapsed:.2f} s")
    print("Artifacts:", OUT_DIR)


if __name__ == "__main__":
    main()
