from __future__ import annotations

import importlib

# Import names required by DepthWizard's pinned DA3 monocular runtime. This is intentionally narrower
# than upstream DA3's full exporter/application dependency set: setup_da3_macos.sh excludes native
# export stacks that are not used by the production monocular prior on macOS.
DA3_RUNTIME_DEPENDENCY_MODULES: tuple[str, ...] = (
    "addict",
    "cv2",
    "e3nn",
    "einops",
    "evo",
    "huggingface_hub.hub_mixin",
    "imageio",
    "omegaconf",
    "requests",
    "safetensors.torch",
    "torch",
    "torchvision",
    "tqdm",
)


def verify_da3_runtime_dependencies() -> tuple[str, ...]:
    """Import the frozen DA3 monocular dependency closure or fail with the exact missing module."""
    imported: list[str] = []
    for module_name in DA3_RUNTIME_DEPENDENCY_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            raise RuntimeError(
                "DepthWizard DA3 runtime dependency closure is incomplete: "
                f"required={module_name!r}; {type(exc).__name__}: {exc}; "
                f"missing_module={missing!r}"
            ) from exc
        imported.append(module_name)
    return tuple(imported)
