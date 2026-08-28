from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from depthwizard.contracts import ProjectRunStatus
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ProjectManifest:
    """Atomic, resumable project state shared by backend, service and desktop.

    Schema v2 separates immutable source/geometry identity from mutable pre-completion calibration
    evidence. This allows a georeferenced project to reconstruct once, pause truthfully for DEM/GCP
    evidence, and resume calibration without recomputing the expensive geometry prior.
    """

    project_dir: Path
    source_path: Path
    project_id: str = field(default_factory=lambda: uuid4().hex)
    job_id: str | None = None
    status: str = ProjectRunStatus.CREATED.value
    created_at_utc: str = field(default_factory=_utc_now)
    updated_at_utc: str = field(default_factory=_utc_now)
    source_sha256: str | None = None
    input_kind: str | None = None
    geometry_config_sha256: str | None = None
    run_config_sha256: str | None = None
    estimator: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.project_dir / "project-manifest.json"

    @classmethod
    def create_or_load(cls, project_dir: str | Path, source_path: str | Path) -> ProjectManifest:
        directory = Path(project_dir)
        manifest_path = directory / "project-manifest.json"
        source = Path(source_path)
        if manifest_path.is_file():
            manifest = cls.load(directory)
            if manifest.source_path.resolve(strict=False) != source.resolve(strict=False):
                raise RuntimeError(
                    "existing DepthWizard project manifest belongs to a different source raster"
                )
            return manifest
        manifest = cls(project_dir=directory, source_path=source)
        manifest.save()
        return manifest

    def mark_status(self, status: ProjectRunStatus | str, *, job_id: str | None = None) -> None:
        self.status = status.value if isinstance(status, ProjectRunStatus) else status
        if job_id is not None:
            self.job_id = job_id
        self.updated_at_utc = _utc_now()
        self.save()

    def set_identity(
        self,
        *,
        source_sha256: str,
        input_kind: str,
        geometry_config_sha256: str,
        run_config_sha256: str,
    ) -> None:
        self.source_sha256 = source_sha256
        self.input_kind = input_kind
        self.geometry_config_sha256 = geometry_config_sha256
        self.run_config_sha256 = run_config_sha256
        self.updated_at_utc = _utc_now()
        self.save()

    def set_estimator(self, estimator: dict[str, Any]) -> None:
        self.estimator = estimator
        self.updated_at_utc = _utc_now()
        self.save()

    def record_stage(
        self,
        stage: ProcessingStage,
        *,
        status: str,
        artifacts: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        previous = self.stages.get(stage.value, {})
        started_at = previous.get("started_at_utc")
        if status == "running" or started_at is None:
            started_at = now
        entry: dict[str, Any] = {
            "status": status,
            "started_at_utc": started_at,
            "updated_at_utc": now,
            "artifacts": artifacts or previous.get("artifacts", {}),
            "details": details or previous.get("details", {}),
        }
        if status in {"completed", "failed", "waiting", "skipped"}:
            entry["completed_at_utc"] = now
        if elapsed_seconds is not None:
            entry["elapsed_seconds"] = float(elapsed_seconds)
        self.stages[stage.value] = entry
        self.updated_at_utc = now
        self.save()

    def verified_artifact_path(self, name: str) -> Path:
        """Return a registered artifact only when its persisted bytes match the manifest identity."""
        payload = self.artifacts.get(name)
        if not payload:
            raise RuntimeError(f"project manifest has no registered {name} artifact")
        raw = payload.get("path")
        expected = payload.get("sha256")
        if not isinstance(raw, str):
            raise RuntimeError(f"project manifest {name} artifact path is malformed")
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"project manifest {name} artifact has no valid SHA-256 identity")
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"persisted {name} artifact does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"persisted {name} artifact hash mismatch; expected {expected}, got {actual}. "
                "DepthWizard will not reuse mutated or stale scientific products."
            )
        return path

    def stage_completed(self, stage: ProcessingStage) -> bool:
        entry = self.stages.get(stage.value, {})
        if entry.get("status") != "completed":
            return False
        # A completed stage is reusable evidence only if every registered artifact it claims still
        # matches the SHA-256 identity frozen into the project manifest.
        stage_artifacts = entry.get("artifacts", {})
        if isinstance(stage_artifacts, dict):
            for name in stage_artifacts:
                if name in self.artifacts:
                    self.verified_artifact_path(name)
        return True

    def register_artifact(
        self,
        name: str,
        path: str | Path,
        *,
        semantics: str,
        units: str | None,
        sha256: str,
    ) -> None:
        artifact_path = Path(path)
        self.artifacts[name] = {
            "path": str(artifact_path.resolve(strict=False)),
            "semantics": semantics,
            "units": units,
            "sha256": sha256,
        }
        self.updated_at_utc = _utc_now()
        self.save()

    def artifact_path(self, name: str) -> Path | None:
        payload = self.artifacts.get(name)
        if not payload:
            return None
        raw = payload.get("path")
        return Path(raw) if isinstance(raw, str) else None

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            self.updated_at_utc = _utc_now()
            self.save()

    def add_error(self, message: str, *, stage: ProcessingStage | None = None) -> None:
        self.errors.append(
            {
                "at_utc": _utc_now(),
                "stage": stage.value if stage is not None else None,
                "message": message,
            }
        )
        self.updated_at_utc = _utc_now()
        self.save()

    def save(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at_utc = _utc_now()
        payload = {
            "schema_version": 2,
            "project_id": self.project_id,
            "job_id": self.job_id,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "input_kind": self.input_kind,
            "geometry_config_sha256": self.geometry_config_sha256,
            "run_config_sha256": self.run_config_sha256,
            "estimator": self.estimator,
            "artifacts": self.artifacts,
            "stages": self.stages,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @classmethod
    def load(cls, project_dir: str | Path) -> ProjectManifest:
        directory = Path(project_dir)
        payload = json.loads((directory / "project-manifest.json").read_text(encoding="utf-8"))
        schema_version = int(payload.get("schema_version", 1))
        if schema_version == 1:
            stages = payload.get("stages", {})
            inferred_status = (
                ProjectRunStatus.COMPLETE.value
                if stages.get(ProcessingStage.COMPLETE.value, {}).get("status") == "completed"
                else ProjectRunStatus.CREATED.value
            )
            return cls(
                project_dir=directory,
                source_path=Path(payload["source_path"]),
                status=inferred_status,
                stages=stages,
            )
        if schema_version != 2:
            raise ValueError(f"unsupported project manifest schema_version={schema_version}")
        return cls(
            project_dir=directory,
            source_path=Path(payload["source_path"]),
            project_id=str(payload["project_id"]),
            job_id=payload.get("job_id"),
            status=str(payload.get("status", ProjectRunStatus.CREATED.value)),
            created_at_utc=str(payload.get("created_at_utc", _utc_now())),
            updated_at_utc=str(payload.get("updated_at_utc", _utc_now())),
            source_sha256=payload.get("source_sha256"),
            input_kind=payload.get("input_kind"),
            geometry_config_sha256=payload.get("geometry_config_sha256"),
            run_config_sha256=payload.get("run_config_sha256"),
            estimator=payload.get("estimator", {}),
            artifacts=payload.get("artifacts", {}),
            stages=payload.get("stages", {}),
            warnings=payload.get("warnings", []),
            errors=payload.get("errors", []),
        )
