from pathlib import Path

import pytest

from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file


def test_completed_stage_rejects_mutated_registered_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    artifact = project / "products" / "rdsm.tif"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original scientific bytes")

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "rdsm",
        artifact,
        semantics="dimensionless_relative_surface_height",
        units="relative",
        sha256=sha256_file(artifact),
    )
    manifest.record_stage(
        ProcessingStage.GEOMETRY,
        status="completed",
        artifacts={"rdsm": str(artifact.resolve())},
    )
    assert manifest.stage_completed(ProcessingStage.GEOMETRY) is True

    artifact.write_bytes(b"mutated scientific bytes")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        manifest.stage_completed(ProcessingStage.GEOMETRY)


def test_verified_artifact_path_requires_registered_sha_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    artifact = project / "products" / "dsm.tif"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"dsm")

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.artifacts["dsm"] = {
        "path": str(artifact.resolve()),
        "semantics": "absolute_digital_surface_model",
        "units": "m",
        "sha256": None,
    }
    with pytest.raises(RuntimeError, match="no valid SHA-256 identity"):
        manifest.verified_artifact_path("dsm")
