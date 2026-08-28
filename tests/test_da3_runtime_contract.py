from __future__ import annotations

import importlib

import pytest

from depthwizard.geometry_prior.da3_runtime_contract import (
    DA3_RUNTIME_DEPENDENCY_MODULES,
    verify_da3_runtime_dependencies,
)


def test_da3_runtime_contract_includes_lazy_checkpoint_dependencies() -> None:
    assert "huggingface_hub.hub_mixin" in DA3_RUNTIME_DEPENDENCY_MODULES
    assert "safetensors.torch" in DA3_RUNTIME_DEPENDENCY_MODULES
    assert "cv2" in DA3_RUNTIME_DEPENDENCY_MODULES
    assert "torch" in DA3_RUNTIME_DEPENDENCY_MODULES
    assert "torchvision" in DA3_RUNTIME_DEPENDENCY_MODULES


def test_da3_runtime_dependency_verifier_fails_closed_on_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def controlled_import(module_name: str):
        if module_name == "huggingface_hub.hub_mixin":
            raise ModuleNotFoundError(
                "No module named 'huggingface_hub'",
                name="huggingface_hub",
            )
        return real_import(module_name)

    monkeypatch.setattr(
        "depthwizard.geometry_prior.da3_runtime_contract.importlib.import_module",
        controlled_import,
    )
    with pytest.raises(RuntimeError, match="huggingface_hub"):
        verify_da3_runtime_dependencies()
