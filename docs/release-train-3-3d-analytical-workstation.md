# Release Train 3 — Production 3D Analytical Workstation

## Mission

Turn the persisted DepthWizard elevation project into a production 3D analytical workspace without
creating a second, scientifically divergent data path. The terrain renderer must consume the same
hashed DSM/rDSM and source RGB already recorded by the project manifest, while all numerical probes
continue to come from the persisted raster products rather than approximated mesh vertices.

## Integrated scope

- persistent textured GLB terrain generation from the project DSM/rDSM and original RGB;
- deterministic multi-resolution LOD pyramid with edge-preserving sampling;
- adaptive finest-LOD stride so large rasters do not create unbounded browser geometry;
- SHA-256 identity for the source texture, elevation surface, every GLB and mesh manifest;
- idempotent/resumable mesh build when the surface, texture and build configuration are unchanged;
- durable `mesh` stage and mesh artifacts in `project-manifest.json`;
- local API to build/reload the mesh report and stream a requested GLB LOD;
- desktop project mesh loading rather than demo-only terrain;
- Orbit, Fly, First Person and Top Down navigation on the real project terrain;
- deterministic automated flythrough and explicit fit/reset camera action;
- explicit LOD selection plus renderer-FPS-driven adaptive LOD policy;
- live renderer telemetry for frame rate, triangles and draw calls;
- display-only vertical exaggeration;
- UV/raycast 3D picking mapped back to normalized raster coordinates;
- synchronized 3D probe values through the existing project-analysis API;
- measurement/profile workflows can use the same normalized coordinates from 2D or 3D;
- selected measurement/profile paths are drawn back onto the terrain only as visualization geometry;
- DSM/rDSM, slope, confidence and residual raster previews can be projected as UV analytical overlays;
- analytical overlays never modify terrain geometry or become a numerical measurement source;
- deterministic hash-audited project ZIP export with per-member identity, semantics, units and size;
- source imagery excluded from export by default and included only by explicit opt-in;
- export archive download through the local service and desktop Export action;
- display exaggeration, overlays, LOD changes and export packaging never mutate persisted elevation.

## Scientific and geospatial invariants

1. Terrain generation consumes only the persisted project surface and the original source RGB.
   Reference DSM, residual rasters and validation metrics are never inputs to mesh geometry.
2. The mesh builder verifies the recorded source/surface SHA-256 values before generating anything.
3. The GLB is a visualization derivative. Numerical values shown to an analyst come from the
   persisted raster products through the probe/profile subsystem, not from interpolating display
   geometry.
4. Metric DSM projects use physical horizontal GSD in metres. Relative/non-georeferenced projects
   retain pixel horizontal units and relative vertical units rather than inventing metres.
5. Nodata pixels remove terrain faces. Nodata values are sanitized only for unreferenced vertices so
   they cannot corrupt scene bounds; they are not turned into valid terrain.
6. Vertical exaggeration is a renderer transform only. The project DSM/rDSM, mesh manifest and
   analyst-reported elevations remain unchanged.
7. LOD switching may change displayed triangle density but not the underlying project coordinate
   contract or analytical sampling result. Automatic LOD responds only to measured rendering
   performance and selects among already-persisted LOD products.
8. A 3D click is converted through GLB UV coordinates to the same normalized image coordinate used
   by the 2D analytical workspace.
9. Terrain analytical overlays use colorized preview rasters as UV display textures only. Residual,
   slope, confidence or DSM overlays do not feed terrain geometry and do not alter persisted rasters.
10. Profile and measurement lines shown on the terrain are display derivatives of raster-backed
    analysis samples. Their numerical values remain the service response from persisted products.
11. Terrain products are hash-addressed in the project manifest and revalidated when reloaded.
12. A mesh build failure is recorded as a mesh-stage failure and cannot rewrite reconstruction,
    calibration or validation evidence.
13. Export packaging re-hashes every included registered artifact before copying it. A hash mismatch
    aborts export instead of silently packaging modified evidence.
14. The ZIP is a transport derivative. Export does not rerun reconstruction, calibration, validation,
    mesh generation or analyst measurements.

## Accepted mesh slice evidence

The first RT3 local acceptance on the already-accepted RT2 OrthoLoC engineering project completed
without rerunning model inference:

- `make verify`: 109 tests passed, Ruff clean and Pyright clean;
- desktop TypeScript/Vite production build passed;
- 1,048,576 valid DSM pixels;
- 28.123 m recorded terrain relief;
- LOD 0 stride 2: 263,169 vertices and 524,288 faces;
- LOD 1 stride 4: 66,049 vertices and 131,072 faces;
- LOD 2 stride 8: 16,641 vertices and 32,768 faces;
- LOD 3 stride 16: 4,225 vertices and 8,192 faces;
- project manifest recorded `reference_data_used: false` for mesh geometry;
- exact mesh head passed GitHub Python 3.12, Python 3.13 and desktop CI jobs.

This is visualization/product acceptance, not new elevation-accuracy evidence.

## Accepted export slice evidence

The deterministic export acceptance on the same persisted project also completed without rerunning
reconstruction or validation:

- `make verify`: 113 tests passed, Ruff clean and Pyright clean;
- desktop TypeScript/Vite production build passed;
- generated bundle size: 24.13 MiB;
- bundle SHA-256: `cf494b337ab2d63033a6112f6f564c4cede7ce90ebd7c4b2acdfcbe06d9438d8`;
- 14 registered artifacts packaged;
- original source imagery bytes excluded by default;
- export smoke verified archive membership and hashes;
- exact export head `d6681eabe11b396f53db12a6e6f391e59db1fb1e` passed GitHub Python 3.12,
  Python 3.13 and desktop CI jobs.

The bundle is a transport derivative and carries no new scientific accuracy claim.

## Current workstation closure target

Before RT3 can close, the integrated workstation head must pass static verification and desktop build
with the new terrain analytical overlays, 3D profile/measurement visualization, deterministic
flythrough, adaptive LOD telemetry and desktop export interaction. A final local workstation smoke/UI
acceptance will then verify these judge-facing controls against the accepted project without touching
consumed benchmark protocols or rerunning model-promotion evidence.
