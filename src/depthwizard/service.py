from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from depthwizard import __version__
from depthwizard.contracts import RasterMetadata
from depthwizard.io.raster import inspect_raster


class InspectRequest(BaseModel):
    path: Path


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
