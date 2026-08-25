# DepthWizard — SIH26175

Final-system engineering repository for **ISRO / Smart India Hackathon 2026** problem statement **SIH26175 — DepthWizard: Single-View Height Estimation and 3D Flythrough**.

This repository is intentionally structured as the final production system from day one. There is no separate MVP/prototype branch or demo-only application architecture.

## Implemented permanent foundations

- explicit PNG/JPG/TIFF/GeoTIFF raster inspection and RGB band handling;
- geospatially correct reprojection of DEM/reference rasters to the prediction grid;
- truthful rDSM export with no invented CRS or metric units;
- GeoTIFF DSM export preserving CRS/geotransform;
- robust Huber/IRLS relative-to-metric affine calibration;
- conservative DEM evidence weighting and low-frequency terrain-bias correction;
- official RMSE / MAE / Pearson correlation metrics plus diagnostic error statistics;
- slope derivation and slope validation;
- residual GeoTIFF + `metrics.json` evidence generation;
- overlap-aware tiled accumulation and deterministic tile-grid generation;
- textured metric terrain GLB generation and LOD-pyramid export;
- SHA-256 provenance helpers and resumable project manifest primitives;
- dataset registry enforcing geographic split integrity and cross-sensor holdout rules;
- permanent geometry-prior interface plus DA3MONO-LARGE adapter boundary;
- loopback-only FastAPI local core service with optional per-session token;
- Tauri/React/Three.js desktop application shell and permanent scientific design system;
- CI skeleton and SIH requirement traceability.

## Python verification

```bash
python -m pip install -e ".[dev]"
pytest
```

Current test suite covers calibration, geospatial reprojection/export, rDSM semantics, official metrics, slope metrics, tiling, mesh generation, dataset split integrity and local-service raster inspection.

## Core CLI

```bash
depthwizard inspect imagery.tif
depthwizard calibrate-dem relative_height.tif srtm.tif dsm.tif
depthwizard validate-dsm dsm.tif reference_lidar.tif evidence/
depthwizard serve --host 127.0.0.1 --port 8765
```

## Desktop application

`apps/desktop` is the permanent Tauri + React + Three.js workstation. The UI does not fabricate scientific content: until a real reconstruction is available, the central canvas remains in a truthful empty/processing state.

```bash
cd apps/desktop
npm install
npm run tauri dev
```

The desktop requires the local DepthWizard core service; production packaging will launch the sidecar automatically with a session token.

## Architecture authority

Read in this order:

1. `MASTER_SPEC.md`
2. `docs/requirements-traceability.yaml`
3. `docs/adr/`

Every feature and scientific claim must trace to an official SIH/ISRO requirement or an explicit competitive-quality objective.
