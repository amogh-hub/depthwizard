from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from depthwizard.contracts import InputKind, ProjectRunStatus
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file

PROJECT_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_ArtifactSignature = tuple[int, int, int]
_ARTIFACT_INTEGRITY_CACHE: dict[tuple[str, str], _ArtifactSignature] = {}
_ARTIFACT_INTEGRITY_LOCK = RLock()
_VALID_PROJECT_STATUSES = {status.value for status in ProjectRunStatus}
_VALID_INPUT_KINDS = {kind.value for kind in InputKind}
_VALID_STAGE_NAMES = {stage.value for stage in ProcessingStage}
_VALID_STAGE_STATUSES = {"running", "completed", "failed", "waiting", "skipped"}


class ProjectIntegrityError(RuntimeError, ValueError):
    """A persisted DepthWizard project violates its immutable/state-integrity contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _artifact_signature(path: Path) -> _ArtifactSignature:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_file_sha256_cached(path: Path, expected_sha256: str, *, context: str) -> None:
    """Verify immutable project bytes once per unchanged filesystem identity.

    Repeated analyst probes should not re-hash a large DSM on every click. The cache is keyed by
    resolved path + expected digest and invalidated whenever size, mtime or ctime changes. The first
    access after process start still performs a full SHA-256 verification; a file that changes while
    it is being hashed is rejected rather than cached.
    """
    signature_before = _artifact_signature(path)
    key = (str(path), expected_sha256)
    with _ARTIFACT_INTEGRITY_LOCK:
        if _ARTIFACT_INTEGRITY_CACHE.get(key) == signature_before:
            return

    actual = sha256_file(path)
    signature_after = _artifact_signature(path)
    if signature_after != signature_before:
        raise ProjectIntegrityError(f"{context} changed during SHA-256 integrity verification")
    if actual != expected_sha256:
        with _ARTIFACT_INTEGRITY_LOCK:
            _ARTIFACT_INTEGRITY_CACHE.pop(key, None)
        raise ProjectIntegrityError(
            f"{context} hash mismatch; expected {expected_sha256}, got {actual}. "
            "DepthWizard will not consume mutated or stale scientific products."
        )
    with _ARTIFACT_INTEGRITY_LOCK:
        _ARTIFACT_INTEGRITY_CACHE[key] = signature_after


def _read_manifest_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"project manifest does not exist: {path}")
    with path.open("rb") as handle:
        raw = handle.read(PROJECT_MANIFEST_MAX_BYTES + 1)
    if len(raw) > PROJECT_MANIFEST_MAX_BYTES:
        raise ProjectIntegrityError(
            f"project manifest exceeds the {PROJECT_MANIFEST_MAX_BYTES // (1024 * 1024)} MiB limit"
        )
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProjectIntegrityError(f"project manifest is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectIntegrityError("project manifest root must be a JSON object")
    return payload


def _object_field(payload: dict[str, Any], name: str) -> dict[str, Any]:
    raw = payload.get(name, {})
    if not isinstance(raw, dict):
        raise ProjectIntegrityError(f"project manifest field {name!r} must be an object")
    if any(not isinstance(key, str) for key in raw):
        raise ProjectIntegrityError(f"project manifest field {name!r} contains a non-string key")
    return raw


def _optional_string_field(
    payload: dict[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    raw = payload.get(name)
    if raw is None:
        return None
    if not isinstance(raw, str) or (not allow_empty and not raw):
        raise ProjectIntegrityError(f"project manifest field {name!r} must be a string or null")
    return raw


def _timestamp_field(payload: dict[str, Any], name: str, *, default: str) -> str:
    raw = payload.get(name, default)
    if not isinstance(raw, str) or not raw:
        raise ProjectIntegrityError(f"project manifest field {name!r} must be a non-empty string")
    return raw


def _optional_sha256_field(payload: dict[str, Any], name: str) -> str | None:
    raw = payload.get(name)
    if raw is None:
        return None
    if not _is_sha256(raw):
        raise ProjectIntegrityError(
            f"project manifest field {name!r} must be a lowercase SHA-256 hex string or null"
        )
    return raw


def _artifact_registry(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = _object_field(payload, "artifacts")
    registry: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ProjectIntegrityError("project manifest artifact entries must be JSON objects")
        path = value.get("path")
        semantics = value.get("semantics")
        units = value.get("units")
        sha256 = value.get("sha256")
        if not isinstance(path, str) or not path:
            raise ProjectIntegrityError(f"project manifest {name} artifact path is malformed")
        if not isinstance(semantics, str) or not semantics:
            raise ProjectIntegrityError(f"project manifest {name} artifact semantics are malformed")
        if units is not None and not isinstance(units, str):
            raise ProjectIntegrityError(f"project manifest {name} artifact units are malformed")
        if not _is_sha256(sha256):
            raise ProjectIntegrityError(
                f"project manifest {name} artifact has no valid SHA-256 identity"
            )
        registry[name] = dict(value)
    return registry


def _stage_registry(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = _object_field(payload, "stages")
    stages: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if name not in _VALID_STAGE_NAMES:
            raise ProjectIntegrityError(f"project manifest contains unsupported stage {name!r}")
        if not isinstance(value, dict):
            raise ProjectIntegrityError("project manifest stage entries must be JSON objects")

        status = value.get("status")
        if not isinstance(status, str) or status not in _VALID_STAGE_STATUSES:
            raise ProjectIntegrityError(f"project manifest stage {name!r} has an invalid status")
        for timestamp_name in ("started_at_utc", "updated_at_utc", "completed_at_utc"):
            timestamp = value.get(timestamp_name)
            if timestamp is not None and (not isinstance(timestamp, str) or not timestamp):
                raise ProjectIntegrityError(
                    f"project manifest stage {name!r} field {timestamp_name!r} is malformed"
                )

        artifacts = value.get("artifacts", {})
        if not isinstance(artifacts, dict) or any(
            not isinstance(key, str) or not isinstance(path, str)
            for key, path in artifacts.items()
        ):
            raise ProjectIntegrityError(
                f"project manifest stage {name!r} artifacts must map strings to strings"
            )
        details = value.get("details", {})
        if not isinstance(details, dict):
            raise ProjectIntegrityError(
                f"project manifest stage {name!r} details must be a JSON object"
            )
        elapsed = value.get("elapsed_seconds")
        if elapsed is not None and (
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0
        ):
            raise ProjectIntegrityError(
                f"project manifest stage {name!r} elapsed_seconds is malformed"
            )
        stages[name] = dict(value)
    return stages


def _warning_list(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("warnings", [])
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ProjectIntegrityError("project manifest warnings must be an array of strings")
    return list(raw)


def _error_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("errors", [])
    if not isinstance(raw, list) or any(not isinstance(value, dict) for value in raw):
        raise ProjectIntegrityError("project manifest errors must be an array of objects")
    errors: list[dict[str, Any]] = []
    for value in raw:
        at_utc = value.get("at_utc")
        stage = value.get("stage")
        message = value.get("message")
        if not isinstance(at_utc, str) or not at_utc:
            raise ProjectIntegrityError("project manifest error timestamp is malformed")
        if stage is not None and (not isinstance(stage, str) or stage not in _VALID_STAGE_NAMES):
            raise ProjectIntegrityError("project manifest error stage is malformed")
        if not isinstance(message, str) or not message:
            raise ProjectIntegrityError("project manifest error message is malformed")
        errors.append(dict(value))
    return errors


def _validate_v2_payload(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = payload.get("project_id")
    if not isinstance(project_id, str) or not project_id:
        raise ProjectIntegrityError("project manifest project_id must be a non-empty string")

    source_path = payload.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        raise ProjectIntegrityError("project manifest source_path must be a non-empty string")

    status = payload.get("status", ProjectRunStatus.CREATED.value)
    if not isinstance(status, str) or status not in _VALID_PROJECT_STATUSES:
        raise ProjectIntegrityError("project manifest status is invalid")

    job_id = _optional_string_field(payload, "job_id")
    input_kind = _optional_string_field(payload, "input_kind")
    if input_kind is not None and input_kind not in _VALID_INPUT_KINDS:
        raise ProjectIntegrityError("project manifest input_kind is invalid")

    return {
        "project_id": project_id,
        "source_path": source_path,
        "job_id": job_id,
        "status": status,
        "created_at_utc": _timestamp_field(payload, "created_at_utc", default=_utc_now()),
        "updated_at_utc": _timestamp_field(payload, "updated_at_utc", default=_utc_now()),
        "source_sha256": _optional_sha256_field(payload, "source_sha256"),
        "input_kind": input_kind,
        "geometry_config_sha256": _optional_sha256_field(payload, "geometry_config_sha256"),
        "run_config_sha256": _optional_sha256_field(payload, "run_config_sha256"),
        "estimator": _object_field(payload, "estimator"),
        "artifacts": _artifact_registry(payload),
        "stages": _stage_registry(payload),
        "warnings": _warning_list(payload),
        "errors": _error_list(payload),
    }


@dataclass
class ProjectManifest:
    """Atomic, resumable project state shared by backend, service and desktop.

    Schema v2 separates immutable source/geometry identity from mutable pre-completion calibration
    evidence. This allows a georeferenced project to reconstruct once, pause truthfully for DEM/GCP
    evidence, and resume calibration without recomputing the expensive geometry prior.

    Registered artifacts are project-owned state. Their resolved paths must remain beneath the
    project directory and their persisted bytes must match the SHA-256 identity recorded at
    registration. Project load performs an integrity pass, cached against filesystem change
    metadata, so downstream analysis cannot bypass the immutable-artifact contract by reading a raw
    manifest path directly.
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

    @property
    def resolved_project_dir(self) -> Path:
        return self.project_dir.resolve(strict=False)

    def _confined_artifact_path(self, name: str, raw_path: str | Path) -> Path:
        """Resolve one artifact path and require it to remain inside this project directory."""
        resolved = Path(raw_path).resolve(strict=False)
        project_root = self.resolved_project_dir
        try:
            relative = resolved.relative_to(project_root)
        except ValueError as exc:
            raise ProjectIntegrityError(
                f"project artifact {name} escapes the project directory: {resolved}"
            ) from exc
        if not relative.parts:
            raise ProjectIntegrityError(
                f"project artifact {name} points at the project directory itself"
            )
        return resolved

    def _validate_registered_artifact_paths(self) -> None:
        for name, payload in self.artifacts.items():
            raw = payload.get("path")
            if not isinstance(raw, str):
                raise ProjectIntegrityError(f"project manifest {name} artifact path is malformed")
            self._confined_artifact_path(name, raw)

    def _validate_registered_artifact_integrity(self) -> None:
        for name in self.artifacts:
            self.verified_artifact_path(name)

    def _validate_stage_artifact_paths(self) -> None:
        """Keep provenance stage pointers inside the project even though they are not file inputs."""
        for stage_name, entry in self.stages.items():
            artifacts = entry.get("artifacts", {})
            if not isinstance(artifacts, dict):
                raise ProjectIntegrityError(
                    f"project manifest stage {stage_name!r} artifacts are malformed"
                )
            for artifact_name, raw_path in artifacts.items():
                if not isinstance(raw_path, str):
                    raise ProjectIntegrityError(
                        f"project manifest stage {stage_name!r} artifact {artifact_name!r} "
                        "path is malformed"
                    )
                self._confined_artifact_path(
                    f"{stage_name}.{artifact_name}",
                    raw_path,
                )

    @classmethod
    def create_or_load(cls, project_dir: str | Path, source_path: str | Path) -> ProjectManifest:
        directory = Path(project_dir)
        manifest_path = directory / "project-manifest.json"
        source = Path(source_path)
        if manifest_path.is_file():
            manifest = cls.load(directory)
            if manifest.source_path.resolve(strict=False) != source.resolve(strict=False):
                raise ProjectIntegrityError(
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
        """Return a registered artifact only when location and bytes match manifest identity."""
        payload = self.artifacts.get(name)
        if not payload:
            raise ProjectIntegrityError(f"project manifest has no registered {name} artifact")
        raw = payload.get("path")
        expected = payload.get("sha256")
        if not isinstance(raw, str):
            raise ProjectIntegrityError(f"project manifest {name} artifact path is malformed")
        if not isinstance(expected, str) or not _is_sha256(expected):
            raise ProjectIntegrityError(
                f"project manifest {name} artifact has no valid SHA-256 identity"
            )
        path = self._confined_artifact_path(name, raw)
        if not path.is_file():
            raise FileNotFoundError(f"persisted {name} artifact does not exist: {path}")
        _verify_file_sha256_cached(path, expected, context=f"persisted {name} artifact")
        return path

    def stage_completed(self, stage: ProcessingStage) -> bool:
        entry = self.stages.get(stage.value, {})
        if entry.get("status") != "completed":
            return False
        # A completed stage is reusable evidence only if every registered artifact it claims still
        # matches the location and SHA-256 identity frozen into the project manifest.
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
        if not _is_sha256(sha256):
            raise ProjectIntegrityError(
                f"cannot register project artifact {name}: invalid SHA-256 identity"
            )
        artifact_path = self._confined_artifact_path(name, path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"cannot register missing project artifact {name}: {artifact_path}")
        _verify_file_sha256_cached(
            artifact_path,
            sha256,
            context=f"project artifact {name} registration",
        )
        self.artifacts[name] = {
            "path": str(artifact_path),
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
        if not isinstance(raw, str):
            raise ProjectIntegrityError(f"project manifest {name} artifact path is malformed")
        return self._confined_artifact_path(name, raw)

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
        # Validate the full document before replacing durable state. This keeps in-memory corruption
        # or a bad caller from persisting a manifest that the next process cannot safely reopen.
        _validate_v2_payload(payload)
        self._validate_registered_artifact_paths()
        self._validate_stage_artifact_paths()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        if temporary.stat().st_size > PROJECT_MANIFEST_MAX_BYTES:
            temporary.unlink(missing_ok=True)
            raise ProjectIntegrityError(
                f"project manifest exceeds the {PROJECT_MANIFEST_MAX_BYTES // (1024 * 1024)} MiB limit"
            )
        temporary.replace(self.path)

    @classmethod
    def load(cls, project_dir: str | Path) -> ProjectManifest:
        directory = Path(project_dir)
        payload = _read_manifest_payload(directory / "project-manifest.json")
        raw_schema_version = payload.get("schema_version", 1)
        if not isinstance(raw_schema_version, int) or isinstance(raw_schema_version, bool):
            raise ProjectIntegrityError("project manifest schema_version must be an integer")
        schema_version = raw_schema_version
        raw_source_path = payload.get("source_path")
        if not isinstance(raw_source_path, str) or not raw_source_path:
            raise ProjectIntegrityError("project manifest source_path must be a non-empty string")

        if schema_version == 1:
            stages = _stage_registry(payload)
            complete_stage = stages.get(ProcessingStage.COMPLETE.value)
            inferred_status = (
                ProjectRunStatus.COMPLETE.value
                if complete_stage is not None and complete_stage.get("status") == "completed"
                else ProjectRunStatus.CREATED.value
            )
            manifest = cls(
                project_dir=directory,
                source_path=Path(raw_source_path),
                status=inferred_status,
                stages=stages,
            )
            manifest._validate_registered_artifact_paths()
            manifest._validate_stage_artifact_paths()
            return manifest
        if schema_version != 2:
            raise ProjectIntegrityError(
                f"unsupported project manifest schema_version={schema_version}"
            )

        validated = _validate_v2_payload(payload)
        manifest = cls(
            project_dir=directory,
            source_path=Path(validated["source_path"]),
            project_id=validated["project_id"],
            job_id=validated["job_id"],
            status=validated["status"],
            created_at_utc=validated["created_at_utc"],
            updated_at_utc=validated["updated_at_utc"],
            source_sha256=validated["source_sha256"],
            input_kind=validated["input_kind"],
            geometry_config_sha256=validated["geometry_config_sha256"],
            run_config_sha256=validated["run_config_sha256"],
            estimator=validated["estimator"],
            artifacts=validated["artifacts"],
            stages=validated["stages"],
            warnings=validated["warnings"],
            errors=validated["errors"],
        )
        manifest._validate_registered_artifact_paths()
        manifest._validate_stage_artifact_paths()
        manifest._validate_registered_artifact_integrity()
        return manifest
