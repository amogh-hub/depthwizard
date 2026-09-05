# DepthWizard Elite Final-System Integration Plan

This branch is the final-system integration line. Work proceeds in **large, production-complete release trains**, not prototype fragments or demo-only shortcuts. Every train must land end-to-end through backend, evidence, desktop integration, tests and documentation before it is considered complete.

## Operating principles

1. Build the permanent architecture directly. No MVP-only forks, fake controls, placeholder scientific outputs or parallel demo pipelines.
2. Preserve scientific truth. Negative benchmark results remain immutable; production uses the safest independently supported estimator path.
3. Integrate vertically. A capability is incomplete until the scientific core, project manifest, local API, desktop state, analytical view, export/provenance and automated acceptance checks agree.
4. Prefer fewer large coherent changes over serial micro-features, while retaining strict tests and rollback-safe commits inside each release train.
5. Keep the main scientific canvas dominant and the UI aligned with the frozen DepthWizard workstation design language.
6. Every user-visible metric, layer and status comes from a real artifact or deterministic computation.

## Release Train 1 — Production Elevation Runtime

Deliver the complete project-processing backbone in one integrated pass:

- permanent estimator policy with calibrated DA3 baseline and independently promoted learned branches only;
- explicit estimator-selection reason and promotion metadata in provenance;
- non-georeferenced rDSM and georeferenced absolute-DSM paths through one job graph;
- DEM-only, GCP-only and DEM+GCP calibration evidence interfaces;
- uncertainty/confidence product semantics;
- slope and derived analytical rasters;
- deterministic geospatial export bundle;
- bounded-memory overlap tiling, seam diagnostics and resumability;
- atomic project manifest, timings, warnings, hashes, product paths and failure recovery;
- end-to-end local API contract consumed by the desktop.

Acceptance: a fresh RGB/GeoTIFF project runs from import to complete geospatial products with no manual backend intervention and produces a self-consistent evidence manifest.

## Release Train 2 — Validation and Scientific Analyst Workspace

Deliver the full real-data validation environment:

- synchronized Optical / Predicted DSM / Reference / Residual / Confidence views;
- split, swipe, overlay and difference comparison;
- RMSE, MAE, Pearson correlation, bias, median, P90/P95 and valid-pixel reporting;
- confidence-versus-error reliability diagnostics;
- calibration diagnostics and evidence-quality status;
- seam diagnostics for tiled scenes;
- cursor-synchronized predicted/reference/residual values;
- validation report generation and evidence export;
- domain/baseline/ablation evidence ingestion without fabricating missing results.

Acceptance: loading a valid reference produces the same metrics in the backend report, desktop inspector and exported evidence package.

## Release Train 3 — Analytical 3D Terrain Workstation

Deliver the complete interactive analytical environment, not a visualization demo:

- geodetically faithful textured terrain mesh and deterministic UV mapping;
- tiled/quadtree LOD, frustum culling and texture mipmapping;
- Orbit, Fly, First-person and Top-down camera modes;
- Texture / DSM / Slope / Confidence / Residual rendering modes;
- synchronized 2D <-> 3D cursor/location state;
- elevation probe;
- slope inspection;
- two-point distance;
- structural-height measurement;
- arbitrary elevation transect/profile;
- contours/hillshade where supported by real products;
- North, Fit, Reset, exaggeration and LOD controls;
- bookmark/view state and provenance-aware screenshots/exports;
- performance instrumentation targeting >=30 FPS on the designated finale hardware.

Acceptance: every enabled control operates on real project artifacts and survives repeated navigation without state drift or crash.

## Release Train 4 — Elite Desktop Product Integration

Deliver the final workstation UX across the frozen state progression:

`Import -> Inspect -> Reconstruct -> Calibrate -> Validate -> Explore -> Export`

Includes:

- persistent top application/status bar;
- narrow left tool rail;
- dominant scientific canvas;
- contextual analytical inspector;
- truthful deterministic processing states with expandable technical details;
- project persistence and recent-project reopening;
- resumable job controls and actionable failure states;
- provenance/history inspection;
- consistent disabled-state policy for capabilities that are genuinely unavailable;
- complete export workflow;
- no placeholder metric cards or decorative AI UI.

Acceptance: an analyst can complete the entire workflow without entering the terminal.

## Release Train 5 — Standalone, Offline, Reliability and Security Closure

Deliver the final deployable application:

- Tauri sidecar lifecycle management;
- loopback-only/session-token boundary;
- packaged model/resource discovery;
- bundled hash-verified model resources and offline-first core operation;
- clean install / clean launch / fresh project acceptance;
- Python tests + Ruff + Pyright;
- frontend build/test;
- Rust build/clippy/tests;
- malformed-input recovery;
- resumability and interrupted-job recovery;
- bounded queue admission/history and cooperative operator cancellation;
- memory/runtime profiling on large scenes;
- 3D navigation benchmark;
- scripted soak test with no unhandled crashes;
- source files never overwritten.

Acceptance: standalone final-system acceptance report is PASS on the designated demo machine.

## Release Train 6 — Scientific Evidence Closure and Submission Lock

Run only after production behavior is stable:

- terrain-diverse urban/sparse/hilly/forest reporting;
- cross-sensor evidence;
- identical-input published baselines where reproducible;
- required ablations;
- reliability and seam analysis;
- final claim registry mapping every pitch/deck statement to evidence;
- documentation/reproducibility dry run;
- final six-slide SIH deck and demo video from the completed system only.

Consumed external targets, including Potsdam external-v2, are never reused for target-driven tuning.

## Completion rule

DepthWizard reaches final elite status only when `docs/elite-completion-gates.md` Gates A-H are PASS with evidence. We optimize for speed by building **large coherent production slices**, never by lowering correctness, scientific integrity, UI quality or verification standards.
