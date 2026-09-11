from __future__ import annotations

import json
import os
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from depthwizard import __version__
from depthwizard.analysis.project_structure import estimate_project_structure_height
from depthwizard.calibration.gcp_io import inspect_ground_control_point_file
from depthwizard.cancellation import CancellationRequested
from depthwizard.contracts import (
    GroundControlPointFileReport,
    ProcessingRequest,
    ProjectExportReport,
    ProjectExportRequest,
    ProjectMeshBuildRequest,
    ProjectMeshReport,
    ProjectProbeRequest,
    ProjectProbeResult,
    ProjectProfileRequest,
    ProjectProfileResult,
    ProjectRunStatus,
    ProjectStructureHeightRequest,
    ProjectStructureHeightResult,
    RasterMetadata,
    ReferenceValidationReport,
    ReferenceValidationRequest,
)
from depthwizard.evaluation.project_analysis import probe_project, sample_project_profile
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.export.project_package import build_project_export, load_project_export
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.io.raster import inspect_raster
from depthwizard.mesh.project_mesh import build_project_mesh, load_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.visualization.layer_legend import project_layer_legend
from depthwizard.visualization.raster_preview import PreviewLayer, render_project_layer_preview


class InspectRequest(BaseModel):
    path: Path


class ProjectJobState(BaseModel):
    job_id: str
    project_dir: Path
    status: ProjectRunStatus
    manifest_path: Path
    submitted_at_utc: str
    updated_at_utc: str
    error: str | None = None
    failure_kind: Literal["resource_exhausted", "processing_error"] | None = None
    cancellation_requested: bool = False


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _actionable_job_failure(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip() or type(exc).__name__
    lowered = message.casefold()
    memory_markers = (
        "out of memory",
        "cannot allocate memory",
        "memory allocation",
        "mps backend out of memory",
        "cuda error: out of memory",
    )
    if isinstance(exc, MemoryError) or any(marker in lowered for marker in memory_markers):
        return (
            "resource_exhausted",
            (
                "Insufficient memory for this reconstruction. Close other large applications, "
                "reduce the inference tile size, or process a smaller crop; the source file was "
                "not modified."
            ),
        )
    return "processing_error", message


def _session_guard(x_depthwizard_token: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("DEPTHWIZARD_SESSION_TOKEN")
    if not expected:
        # Development mode is still loopback-only by launch policy; production packaging sets token.
        return
    if x_depthwizard_token != expected:
        raise HTTPException(status_code=401, detail="invalid DepthWizard session token")


app = FastAPI(
    title="DepthWizard Local Core",
    version=__version__,
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json" if os.environ.get("DEPTHWIZARD_DEV") == "1" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "http://127.0.0.1:1420",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "x-depthwizard-token"],
)

# One production worker is intentional: it prevents simultaneous jobs from contending for the
# same CUDA/MPS model and keeps accelerator memory bounded. The runtime itself is resumable via the
# project manifest, so a desktop can safely resubmit an interrupted project after service restart.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depthwizard-project")
_jobs: dict[str, ProjectJobState] = {}
_job_futures: dict[str, Future[None]] = {}
_cancel_requested: set[str] = set()
_jobs_lock = Lock()
_production_runtime = ProductionElevationRuntime(prior=DA3MonocularPrior(device="auto"))


def _bounded_environment_integer(name: str, *, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


_MAX_IN_FLIGHT_JOBS = _bounded_environment_integer(
    "DEPTHWIZARD_MAX_IN_FLIGHT_JOBS", default=2, minimum=1, maximum=8
)
_MAX_JOB_HISTORY = _bounded_environment_integer(
    "DEPTHWIZARD_MAX_JOB_HISTORY", default=64, minimum=8, maximum=1024
)
_TERMINAL_JOB_STATUSES = {
    ProjectRunStatus.WAITING_FOR_CALIBRATION,
    ProjectRunStatus.COMPLETE,
    ProjectRunStatus.FAILED,
    ProjectRunStatus.CANCELLED,
}


def _set_job(job_id: str, **updates: object) -> None:
    with _jobs_lock:
        current = _jobs.get(job_id)
        if current is None:
            return
        updates["updated_at_utc"] = _utc_now()
        _jobs[job_id] = current.model_copy(update=updates)


def _prune_job_history_locked() -> None:
    excess = len(_jobs) - _MAX_JOB_HISTORY
    if excess <= 0:
        return
    removable = sorted(
        (
            state
            for state in _jobs.values()
            if state.status in _TERMINAL_JOB_STATUSES and state.job_id not in _job_futures
        ),
        key=lambda state: state.updated_at_utc,
    )
    for state in removable[:excess]:
        _jobs.pop(state.job_id, None)
        _cancel_requested.discard(state.job_id)


def _job_cancelled(job_id: str) -> bool:
    with _jobs_lock:
        return job_id in _cancel_requested


def _release_job_future(job_id: str) -> None:
    with _jobs_lock:
        _job_futures.pop(job_id, None)
        _prune_job_history_locked()


def _run_project_job(job_id: str, request: ProcessingRequest) -> None:
    if _job_cancelled(job_id):
        _set_job(job_id, status=ProjectRunStatus.CANCELLED, error=None)
        return
    _set_job(job_id, status=ProjectRunStatus.RUNNING)
    try:
        result = _production_runtime.run(
            request,
            job_id=job_id,
            cancellation_probe=lambda: _job_cancelled(job_id),
        )
    except CancellationRequested:
        _set_job(job_id, status=ProjectRunStatus.CANCELLED, error=None)
        return
    except Exception as exc:  # noqa: BLE001
        # This is the outermost executor boundary. Scientific/runtime code already records its
        # stage-specific failure in the durable manifest; the worker must additionally convert any
        # ordinary unhandled exception into a terminal job state instead of leaving the UI polling
        # a job that can never complete. BaseException subclasses are deliberately not intercepted.
        failure_kind, error = _actionable_job_failure(exc)
        _set_job(
            job_id,
            status=ProjectRunStatus.FAILED,
            error=error,
            failure_kind=failure_kind,
        )
        return
    _set_job(job_id, status=result.status, error=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/inspect", dependencies=[Depends(_session_guard)])
def inspect(request: InspectRequest) -> RasterMetadata:
    if not request.path.exists():
        raise HTTPException(status_code=404, detail="source raster does not exist")
    try:
        return inspect_raster(request.path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"unable to inspect raster: {exc}") from exc


@app.post(
    "/v1/calibration/gcps/inspect",
    response_model=GroundControlPointFileReport,
    dependencies=[Depends(_session_guard)],
)
def inspect_gcps(request: InspectRequest) -> GroundControlPointFileReport:
    if not request.path.exists():
        raise HTTPException(status_code=404, detail="GCP file does not exist")
    try:
        return inspect_ground_control_point_file(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/projects",
    response_model=ProjectJobState,
    status_code=202,
    dependencies=[Depends(_session_guard)],
)
def submit_project(request: ProcessingRequest) -> ProjectJobState:
    if not request.source.is_file():
        raise HTTPException(status_code=404, detail="source raster does not exist")
    if request.metric_dem_path is not None and not request.metric_dem_path.is_file():
        raise HTTPException(status_code=404, detail="metric DEM does not exist")
    job_id = uuid4().hex
    now = _utc_now()
    state = ProjectJobState(
        job_id=job_id,
        project_dir=request.output_dir,
        status=ProjectRunStatus.QUEUED,
        manifest_path=request.output_dir / "project-manifest.json",
        submitted_at_utc=now,
        updated_at_utc=now,
    )
    with _jobs_lock:
        in_flight = sum(
            state.status in {ProjectRunStatus.QUEUED, ProjectRunStatus.RUNNING}
            for state in _jobs.values()
        )
        if in_flight >= _MAX_IN_FLIGHT_JOBS:
            raise HTTPException(
                status_code=429,
                detail=(
                    "DepthWizard processing queue is full; wait for or cancel an active job "
                    "before submitting another"
                ),
                headers={"Retry-After": "5"},
            )
        requested_project = request.output_dir.resolve(strict=False)
        if any(
            existing.status in {ProjectRunStatus.QUEUED, ProjectRunStatus.RUNNING}
            and existing.project_dir.resolve(strict=False) == requested_project
            for existing in _jobs.values()
        ):
            raise HTTPException(
                status_code=409,
                detail="this project directory already has an active DepthWizard job",
            )
        _prune_job_history_locked()
        _jobs[job_id] = state
    try:
        future = _executor.submit(_run_project_job, job_id, request)
    except RuntimeError as exc:
        _set_job(job_id, status=ProjectRunStatus.FAILED, error=str(exc))
        raise HTTPException(status_code=503, detail="DepthWizard worker is unavailable") from exc
    with _jobs_lock:
        _job_futures[job_id] = future
    future.add_done_callback(lambda _future: _release_job_future(job_id))
    return state


@app.get(
    "/v1/jobs/{job_id}",
    response_model=ProjectJobState,
    dependencies=[Depends(_session_guard)],
)
def job_status(job_id: str) -> ProjectJobState:
    with _jobs_lock:
        state = _jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown DepthWizard job id")
    return state


@app.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=ProjectJobState,
    dependencies=[Depends(_session_guard)],
)
def cancel_job(job_id: str) -> ProjectJobState:
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown DepthWizard job id")
        if state.status in _TERMINAL_JOB_STATUSES:
            return state
        _cancel_requested.add(job_id)
        future = _job_futures.get(job_id)
        updated = state.model_copy(
            update={
                "cancellation_requested": True,
                "updated_at_utc": _utc_now(),
            }
        )
        _jobs[job_id] = updated
    # Future.cancel() may invoke callbacks synchronously; never call it while holding _jobs_lock.
    cancelled_before_start = future.cancel() if future is not None else False
    if cancelled_before_start:
        _set_job(job_id, status=ProjectRunStatus.CANCELLED, error=None)
    with _jobs_lock:
        return _jobs[job_id]


@app.get("/v1/projects/manifest", dependencies=[Depends(_session_guard)])
def project_manifest(project_dir: Path) -> dict[str, object]:
    path = project_dir / "project-manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="project manifest does not exist")
    try:
        manifest = ProjectManifest.load(project_dir)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"unable to read project manifest: {exc}"
        ) from exc
    # Reading the persisted JSON rather than re-serializing the dataclass guarantees the desktop
    # sees the exact durable state that would survive a sidecar restart.
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="project manifest root must be an object")
    return payload


@app.post(
    "/v1/projects/validate",
    response_model=ReferenceValidationReport,
    dependencies=[Depends(_session_guard)],
)
def validate_reference(request: ReferenceValidationRequest) -> ReferenceValidationReport:
    if not (request.project_dir / "project-manifest.json").is_file():
        raise HTTPException(status_code=404, detail="project manifest does not exist")
    if not request.reference_path.is_file():
        raise HTTPException(status_code=404, detail="reference DSM does not exist")
    try:
        return validate_project_reference(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/v1/projects/validation",
    response_model=ReferenceValidationReport,
    dependencies=[Depends(_session_guard)],
)
def validation_report(project_dir: Path) -> ReferenceValidationReport:
    path = project_dir / "metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="project validation metrics do not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ReferenceValidationReport.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"unable to read validation metrics: {exc}"
        ) from exc


@app.post(
    "/v1/projects/probe",
    response_model=ProjectProbeResult,
    dependencies=[Depends(_session_guard)],
)
def project_probe(request: ProjectProbeRequest) -> ProjectProbeResult:
    try:
        return probe_project(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/projects/profile",
    response_model=ProjectProfileResult,
    dependencies=[Depends(_session_guard)],
)
def project_profile(request: ProjectProfileRequest) -> ProjectProfileResult:
    try:
        return sample_project_profile(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/projects/structure-height",
    response_model=ProjectStructureHeightResult,
    dependencies=[Depends(_session_guard)],
)
def project_structure_height(
    request: ProjectStructureHeightRequest,
) -> ProjectStructureHeightResult:
    try:
        return estimate_project_structure_height(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/projects/mesh",
    response_model=ProjectMeshReport,
    dependencies=[Depends(_session_guard)],
)
def project_mesh_build(request: ProjectMeshBuildRequest) -> ProjectMeshReport:
    try:
        return build_project_mesh(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/v1/projects/mesh",
    response_model=ProjectMeshReport,
    dependencies=[Depends(_session_guard)],
)
def project_mesh_report(project_dir: Path) -> ProjectMeshReport:
    try:
        return load_project_mesh(project_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/projects/mesh/lod/{level}", dependencies=[Depends(_session_guard)])
def project_mesh_lod(project_dir: Path, level: int) -> FileResponse:
    if level < 0:
        raise HTTPException(status_code=422, detail="terrain LOD level must be non-negative")
    try:
        report = load_project_mesh(project_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    lod = next((item for item in report.lods if item.level == level), None)
    if lod is None:
        raise HTTPException(status_code=404, detail=f"terrain LOD {level} is not available")
    return FileResponse(
        path=lod.path,
        media_type="model/gltf-binary",
        filename=lod.path.name,
        headers={
            "Cache-Control": "no-store",
            "ETag": f'"{lod.sha256}"',
            "X-DepthWizard-Mesh-SHA256": lod.sha256,
        },
    )


@app.post(
    "/v1/projects/export",
    response_model=ProjectExportReport,
    dependencies=[Depends(_session_guard)],
)
def project_export_build(request: ProjectExportRequest) -> ProjectExportReport:
    try:
        return build_project_export(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/v1/projects/export",
    response_model=ProjectExportReport,
    dependencies=[Depends(_session_guard)],
)
def project_export_report(project_dir: Path) -> ProjectExportReport:
    try:
        return load_project_export(project_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/projects/export/archive", dependencies=[Depends(_session_guard)])
def project_export_archive(project_dir: Path) -> FileResponse:
    try:
        report = load_project_export(project_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileResponse(
        path=report.bundle_path,
        media_type="application/zip",
        filename=report.bundle_path.name,
        headers={
            "Cache-Control": "no-store",
            "ETag": f'"{report.bundle_sha256}"',
            "X-DepthWizard-Export-SHA256": report.bundle_sha256,
        },
    )


@app.get("/v1/projects/preview", dependencies=[Depends(_session_guard)])
def project_preview(
    project_dir: Path,
    layer: str,
    max_side: Annotated[int, Query(ge=64, le=4096)] = 1600,
) -> Response:
    allowed = {
        "optical",
        "rdsm",
        "dsm",
        "slope",
        "reference",
        "residual",
        "confidence",
        "hillshade",
        "contours",
    }
    if layer not in allowed:
        raise HTTPException(status_code=422, detail=f"unsupported project preview layer: {layer}")
    if not (project_dir / "project-manifest.json").is_file():
        raise HTTPException(status_code=404, detail="project manifest does not exist")
    try:
        payload = render_project_layer_preview(
            project_dir,
            cast(PreviewLayer, layer),
            max_side=max_side,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"unable to render project layer: {exc}"
        ) from exc
    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/v1/projects/preview/legend", dependencies=[Depends(_session_guard)])
def project_preview_legend(project_dir: Path, layer: str) -> dict[str, object]:
    allowed = {
        "optical",
        "rdsm",
        "dsm",
        "slope",
        "reference",
        "residual",
        "confidence",
        "hillshade",
        "contours",
    }
    if layer not in allowed:
        raise HTTPException(status_code=422, detail=f"unsupported project preview layer: {layer}")
    if not (project_dir / "project-manifest.json").is_file():
        raise HTTPException(status_code=404, detail="project manifest does not exist")
    try:
        return cast(dict[str, object], project_layer_legend(project_dir, cast(PreviewLayer, layer)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"unable to derive project layer legend: {exc}"
        ) from exc
