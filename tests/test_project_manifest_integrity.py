import json
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


def test_project_load_rejects_mutated_registered_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    artifact = project / "products" / "slope.tif"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original slope bytes")

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "slope",
        artifact,
        semantics="surface_slope",
        units="degrees",
        sha256=sha256_file(artifact),
    )
    # Registration seeds the cache. A subsequent filesystem mutation must invalidate it through
    # size/mtime/ctime and force a new SHA-256 check on project load.
    artifact.write_bytes(b"mutated slope bytes with a different identity")

    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        ProjectManifest.load(project)


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


def test_register_artifact_rejects_path_outside_project(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"outside scientific bytes")

    manifest = ProjectManifest.create_or_load(project, source)
    with pytest.raises(RuntimeError, match="escapes the project directory"):
        manifest.register_artifact(
            "dsm",
            outside,
            semantics="absolute_digital_surface_model",
            units="m",
            sha256=sha256_file(outside),
        )


def test_register_artifact_rejects_supplied_hash_that_does_not_match_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    artifact = project / "products" / "dsm.tif"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"metric dsm")

    manifest = ProjectManifest.create_or_load(project, source)
    with pytest.raises(RuntimeError, match="registration hash mismatch"):
        manifest.register_artifact(
            "dsm",
            artifact,
            semantics="absolute_digital_surface_model",
            units="m",
            sha256="0" * 64,
        )


def test_load_rejects_manifest_with_artifact_path_outside_project(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    outside = tmp_path / "outside.tif"
    outside.write_bytes(b"outside scientific bytes")

    manifest = ProjectManifest.create_or_load(project, source)
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["artifacts"] = {
        "dsm": {
            "path": str(outside.resolve()),
            "semantics": "absolute_digital_surface_model",
            "units": "m",
            "sha256": sha256_file(outside),
        }
    }
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="escapes the project directory"):
        ProjectManifest.load(project)
