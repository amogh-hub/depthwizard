import json
from pathlib import Path

import pytest

from depthwizard.pipeline import project as project_module
from depthwizard.pipeline.project import ProjectIntegrityError, ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file


def _rewrite_manifest(manifest: ProjectManifest, **updates: object) -> None:
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload.update(updates)
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_load_rejects_legacy_non_object_stage_as_integrity_error(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    project.mkdir()
    (project / "project-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_path": str(source),
                "stages": {"complete": "corrupted-stage"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectIntegrityError, match="stage entries must be JSON objects"):
        ProjectManifest.load(project)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "teleporting", "status is invalid"),
        ("input_kind", "probably_georeferenced", "input_kind is invalid"),
        ("source_sha256", "not-a-sha", "source_sha256"),
        ("geometry_config_sha256", "A" * 64, "geometry_config_sha256"),
        ("run_config_sha256", 123, "run_config_sha256"),
        ("created_at_utc", 123, "created_at_utc"),
    ],
)
def test_load_rejects_malformed_v2_identity_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    manifest = ProjectManifest.create_or_load(project, source)
    _rewrite_manifest(manifest, **{field: value})

    with pytest.raises(ProjectIntegrityError, match=message):
        ProjectManifest.load(project)


def test_load_rejects_malformed_artifact_metadata_before_file_use(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    artifact = project / "products" / "dsm.tif"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"metric dsm")
    manifest = ProjectManifest.create_or_load(project, source)

    _rewrite_manifest(
        manifest,
        artifacts={
            "dsm": {
                "path": str(artifact.resolve()),
                "semantics": "",
                "units": "m",
                "sha256": sha256_file(artifact),
            }
        },
    )

    with pytest.raises(ProjectIntegrityError, match="artifact semantics are malformed"):
        ProjectManifest.load(project)


def test_load_rejects_malformed_stage_status(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    manifest = ProjectManifest.create_or_load(project, source)
    _rewrite_manifest(
        manifest,
        stages={
            "ingest": {
                "status": "somehow-done",
                "artifacts": {},
                "details": {},
            }
        },
    )

    with pytest.raises(ProjectIntegrityError, match="invalid status"):
        ProjectManifest.load(project)


def test_manifest_read_is_bounded_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(project_module, "PROJECT_MANIFEST_MAX_BYTES", 64)
    (project / "project-manifest.json").write_bytes(b"{" + b"x" * 64)

    with pytest.raises(ProjectIntegrityError, match="manifest exceeds"):
        ProjectManifest.load(project)


def test_save_rejects_invalid_in_memory_state_without_replacing_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    manifest = ProjectManifest.create_or_load(project, source)
    before = manifest.path.read_bytes()
    manifest.status = "impossible-state"

    with pytest.raises(ProjectIntegrityError, match="status is invalid"):
        manifest.save()

    assert manifest.path.read_bytes() == before


def test_load_rejects_stage_artifact_pointer_outside_project(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.write_bytes(b"source")
    project = tmp_path / "project"
    outside = tmp_path / "outside-stage-artifact.tif"
    outside.write_bytes(b"not project owned")
    manifest = ProjectManifest.create_or_load(project, source)
    _rewrite_manifest(
        manifest,
        stages={
            "geometry": {
                "status": "completed",
                "artifacts": {"rdsm": str(outside.resolve())},
                "details": {},
            }
        },
    )

    with pytest.raises(ProjectIntegrityError, match="escapes the project directory"):
        ProjectManifest.load(project)
