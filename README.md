# DepthWizard — SIH26175

Final-system engineering repository for **ISRO / Smart India Hackathon 2026** problem statement **SIH26175 — DepthWizard: Single-View Height Estimation and 3D Flythrough**.

DepthWizard is a unified scientific geospatial workstation that converts one optical RGB remote-sensing image into a truthful relative DSM when geodetic scale is unavailable, or an evidence-calibrated metric DSM when georeferencing plus defensible DEM/GCP evidence are available. It then turns the persisted surface into an analytical textured 3D terrain that can be navigated, measured, validated and exported without a user-visible terminal.

## Implemented final-system foundations

- PNG/JPG/JPEG/TIFF/GeoTIFF ingest with explicit RGB-band handling and CRS/transform inspection;
- non-georeferenced **dimensionless rDSM** output with no fabricated metres or CRS;
- georeferenced **metric DSM** calibration from low-resolution DEM, sparse GCPs, or DEM + GCP fusion;
- DA3MONO-LARGE relative-geometry prior with exact upstream source commit, Hugging Face revision and checkpoint SHA identity pinned in source/model manifest;
- robust polarity diagnosis, Huber/IRLS scale-offset fitting, coarse-DEM frequency matching, independent-support anchor sampling and low-frequency terrain-bias correction;
- conservative model-native confidence weighting for DEM calibration when usable, explicitly **not probability calibrated**, with truthful DEM-only fallback when confidence is missing/degenerate;
- local-ground/geodesic spatial scale handling, including Web-Mercator correction for ground distance, slope spacing, mesh XY scale and analyst measurements;
- overlap-aware tiled inference/harmonization plus an objective seam-to-interior discontinuity evaluator for final large-scene evidence;
- RMSE, MAE, Pearson correlation, bias, percentile errors, slope metrics, aligned reference DSM and residual products;
- analyst probe, signed two-point surface Δz, subpixel elevation transects, cumulative profile gain/loss and synchronized reference/residual inspection;
- explicit-footprint structural-height estimation using a robust surrounding local-ground plane rather than mislabelling arbitrary endpoint Δz as building height;
- textured terrain GLB + persistent LOD pyramid with corrected upward-facing geometry, analytical overlays, Orbit/Fly/First Person/Top Down navigation and deterministic flythrough;
- evidence-native 3D analyst interaction: registered surface probing, Measure, Profiles and Structures remain tied to authoritative raster coordinates;
- immutable project/product SHA identities, source/evidence provenance, fail-closed artifact reuse and hash-audited export bundles;
- packaged Tauri + React + Three.js workstation with PyInstaller ONEDIR scientific sidecar, loopback-only authenticated IPC, owned lifecycle and offline-after-model-install operation;
- standalone recovery/error paths, malformed-input rejection and no-terminal packaged processing acceptance;
- exact problem-statement traceability in `docs/requirements-traceability.yaml`.

## Scientific claim boundaries

DA3MONO-LARGE is used as a **relative monocular geometry prior**, not as an absolute satellite-height oracle. Metric elevation is claimed only after explicit geodetic evidence calibration. Model-native confidence is a monotonic reliability signal, not a calibrated probability of correctness. Learned refinement remains unpromoted unless independent evidence beats the production path under the frozen protocol.

Engineering and standalone acceptance already earned remain valid. The final four-terrain geographically disjoint/cross-sensor science campaign, same-input published baselines/ablations, finale-hardware FPS and two-hour soak, clean-machine reproducibility and final submission evidence are separate frozen RT6/RT7 qualification gates; they must not be inferred from integration fixtures.

## Python verification

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests scripts
pyright src
```

The suite covers calibration, confidence weighting, geospatial scale/reprojection/export, rDSM semantics, metrics, slope, analyst profiles, structural height, tiling/seams, mesh geometry, project integrity, dataset split integrity, service APIs and standalone architecture contracts.

## Core CLI

```bash
depthwizard inspect imagery.tif
depthwizard calibrate-dem relative_height.tif srtm.tif dsm.tif
depthwizard validate-dsm dsm.tif reference_lidar.tif evidence/
depthwizard serve --host 127.0.0.1 --port 8765
```

## Desktop application

`apps/desktop` is the permanent Tauri + React + Three.js scientific workstation. The packaged application owns the local scientific sidecar, ephemeral loopback endpoint and per-session token; users do not need to start a terminal service. Scientific layers and controls are enabled only when their corresponding persisted artifacts actually exist.

Developer build:

```bash
cd apps/desktop
npm install
npm run build
npm run test
npm run tauri build
```

Release packaging must use the repository's standalone qualification scripts/targets rather than treating a frontend-only build as standalone scientific acceptance.

## Architecture authority

Read in this order:

1. `MASTER_SPEC.md`
2. `docs/requirements-traceability.yaml`
3. `docs/adr/`
4. release-train evidence/claim-boundary documents

Every feature and scientific claim must trace to an official SIH/ISRO requirement or an explicit competitive-quality objective, and no evidence result may be upgraded beyond the protocol that produced it.
