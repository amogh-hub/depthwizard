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
4. Metric horizontal units are used only under a trusted spatial-scale contract. Generic projected
   and geographic rasters use CRS-aware scale checks. OrthoLoC acceptance/benchmark data may use the
   explicit dataset-local metric-affine contract because the official dataset defines DOP/DSM pixel
   scale in metres. That opt-in is never applied to arbitrary rasters.
5. Under the OrthoLoC local-metric affine contract, affine map coordinates are treated as local
   metres and are **not** converted into global longitude/latitude merely because a syntactic CRS tag
   exists on a raw TIFF representation.
6. Relative/non-georeferenced projects retain pixel horizontal units and relative vertical units
   rather than inventing metres.
7. Nodata pixels remove terrain faces. Nodata values are sanitized only for unreferenced vertices so
   they cannot corrupt scene bounds; they are not turned into valid terrain.
8. Vertical exaggeration is a renderer transform only. The project DSM/rDSM, mesh manifest and
   analyst-reported elevations remain unchanged.
9. LOD switching may change displayed triangle density but not the underlying project coordinate
   contract or analytical sampling result. Automatic LOD responds only to measured rendering
   performance and selects among already-persisted LOD products.
10. A 3D click is converted through GLB UV coordinates to the same normalized image coordinate used
    by the 2D analytical workspace.
11. Terrain analytical overlays use colorized preview rasters as UV display textures only. Residual,
    slope, confidence or DSM overlays do not feed terrain geometry and do not alter persisted rasters.
12. Profile and measurement lines shown on the terrain are display derivatives of raster-backed
    analysis samples. Their numerical values remain the service response from persisted products.
13. Terrain products are hash-addressed in the project manifest and revalidated when reloaded.
14. A mesh build failure is recorded as a mesh-stage failure and cannot rewrite reconstruction,
    calibration or validation evidence.
15. Export packaging re-hashes every included registered artifact before copying it. A hash mismatch
    aborts export instead of silently packaging modified evidence.
16. The ZIP is a transport derivative. Export does not rerun reconstruction, calibration, validation,
    mesh generation or analyst measurements.

## Superseded provisional RT3 acceptance evidence

Early RT3 mesh/export/workstation smokes reused the RT2 OrthoLoC engineering project. The visual and
hash-integrity paths passed, but a later workstation check exposed an impossible horizontal profile
of roughly 16,325 km and a roughly 29,136 km scene diagonal for the 1024 × 1024 demo raster. The
scene-diagonal guard had been derived from the same incorrect horizontal scale, so the two numbers
were self-consistent without being physically valid.

Those RT3 smoke results and their export hashes are **superseded** and are not final RT3 acceptance
evidence. They remain useful as a record of the bug discovery, but they must not be cited as proof of
scientifically correct 3D scale.

Root cause: the raw OrthoLoC TIFF representation can carry a syntactic CRS while its DOP/DSM affine
is dataset-local. The official OrthoLoC dataset contract states that `scale` is the scale of one DOP
or DSM pixel in metres. DepthWizard already had an explicit OrthoLoC metric-affine opt-in for
CRS-free unpacked data, but the opt-in was evaluated only after checking for a missing CRS. A
CRS-bearing demo TIFF could therefore be interpreted through global CRS/geodesic semantics and turn
a local pixel scale into tens of kilometres per pixel.

## Corrected RT3 spatial foundation

RT3 now closes that ambiguity before final workstation acceptance:

- the explicit `DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1` contract is evaluated before CRS interpretation;
- under that contract, affine basis-vector lengths are metric pixel spacing regardless of whether the
  raw OrthoLoC TIFF also carries a syntactic CRS tag;
- analyst profiles use direct affine-coordinate distances, including rotation/shear, rather than a
  WGS84 round-trip;
- global longitude/latitude is withheld for the OrthoLoC local-metric override;
- generic non-OrthoLoC raster behavior remains unchanged and requires trustworthy CRS semantics;
- a new project directory, `artifacts/acceptance/release-train-3-ortholoc-spatial`, is used so the
  previous RT2 project and its historical evidence are preserved rather than silently rewritten;
- the fresh spatial-foundation run regenerates DSM-dependent slope and reference-validation slope
  products under the corrected scale contract;
- reference cache chronology is recorded truthfully: a reference already present from RT2 is not
  deleted merely to claim it was downloaded after reconstruction; the enforced boundary is that it
  is not supplied to reconstruction/calibration and validation is invoked only after runtime completion;
- mesh acceptance independently checks reported GSD against the persisted raster affine;
- workstation acceptance independently recomputes the full sampled profile distance from raster
  affine coordinates instead of comparing two quantities derived from the same GSD scalar.

This clean rebuild is product/geospatial-correctness acceptance only. It is not a rerun of Potsdam,
frozen holdouts, Joshimath, or any consumed model-promotion benchmark, and it creates no new model
promotion claim.

## Final RT3 acceptance order

The final local closure order is:

1. `make verify`
2. `make release-train-3-spatial-foundation-smoke`
3. `make release-train-3-mesh-smoke`
4. `make release-train-3-export-smoke`
5. `make release-train-3-workstation-smoke`

PR #4 stays draft until the fresh spatial-foundation, mesh, export and workstation artifacts are all
coherent and the exact latest head is green on Python 3.12, Python 3.13 and the desktop production
build.
