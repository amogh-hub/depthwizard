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
- explicit LOD selection and display-only vertical exaggeration;
- Orbit, Fly, First Person and Top Down navigation on the real project terrain;
- UV/raycast 3D picking mapped back to normalized raster coordinates;
- synchronized 3D probe values through the existing project-analysis API;
- measurement/profile workflows can use the same normalized coordinates from 2D or 3D;
- display exaggeration never mutates persisted elevation or analytical measurements.

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
   contract or analytical sampling result.
8. A 3D click is converted through GLB UV coordinates to the same normalized image coordinate used
   by the 2D analytical workspace.
9. Terrain products are hash-addressed in the project manifest and revalidated when reloaded.
10. A mesh build failure is recorded as a mesh-stage failure and cannot rewrite reconstruction,
    calibration or validation evidence.

## First vertical-slice acceptance

Before expanding RT3 into overlays, camera bookmarks, export packaging and performance telemetry:

- Python unit/integration tests, Ruff and Pyright pass;
- desktop TypeScript/Vite production build passes;
- GitHub Python 3.12, Python 3.13 and desktop jobs are green;
- synthetic metric project builds multiple persistent GLB LODs;
- every generated GLB and `mesh-manifest.json` has a verified SHA-256 identity;
- a repeated unchanged build resumes the existing mesh report rather than regenerating evidence;
- mesh API returns the persisted report and serves a valid binary GLB;
- project manifest records `reference_data_used: false` for the mesh stage;
- real desktop project can build 3D terrain after DSM/rDSM production;
- 3D view exposes real LOD controls, navigation modes and display-only vertical exaggeration;
- clicking the terrain drives the same synchronized project probe used by the 2D workspace.

## Train boundary

This first RT3 slice does not yet claim the full rendering/UX half of SIH complete. Remaining RT3
work includes analytical raster overlays on terrain, profile/measurement geometry drawn in 3D,
camera bookmarks and deterministic flythrough paths, automatic LOD policy/performance telemetry,
export packaging, mesh seam/UV checks on large scenes, and standalone desktop reliability. Those are
built on this persistent project-mesh contract rather than on a demo-only GLB path.
