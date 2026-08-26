from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from depthwizard import __version__
from depthwizard.contracts import ProcessingRequest, ProjectRunStatus, RasterMetadata
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.io.raster import inspect_raster
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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
_jobs_lock = Lock()
_production_runtime = ProductionElevationRuntime(prior=DA3MonocularPrior(device="auto"))


def _set_job(job_id: str, **updates: object) -> None:
    with _jobs_lock:
        current = _jobs.get(job_id)
        if current is None:
            return
        updates["updated_at_utc"] = _utc_now()
        _jobs[job_id] = current.model_copy(update=updates)


def _run_project_job(job_id: str, request: ProcessingRequest) -> None:
    _set_job(job_id, status=ProjectRunStatus.RUNNING)
    try:
        result = _production_runtime.run(request, job_id=job_id)
    except Exception as exc:
        _set_job(job_id, status=ProjectRunStatus.FAILED, error=str(exc))
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
        _jobs[job_id] = state
    _executor.submit(_run_project_job, job_id, request)
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


@app.get("/v1/projects/manifest", dependencies=[Depends(_session_guard)])
def project_manifest(project_dir: Path) -> dict[str, object]:
    path = project_dir / "project-manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="project manifest does not exist")
    try:
        manifest = ProjectManifest.load(project_dir)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"unable to read project manifest: {exc}") from exc
    # Reading the persisted JSON rather than re-serializing the dataclass guarantees the desktop
    # sees the exact durable state that would survive a sidecar restart.
    import json

    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="project manifest root must be an object")
    return payload
