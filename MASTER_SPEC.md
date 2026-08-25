# DepthWizard Master Engineering Specification

**Problem Statement:** SIH26175 — DepthWizard: Single-View Height Estimation and 3D Flythrough  
**Organization:** Indian Space Research Organisation (ISRO)  
**Department:** Department of Space / Indian Space Research Organisation  
**Category:** Software  
**Theme:** Disaster Management  
**Engineering mode:** final system from day one; no MVP/prototype/demo-only architecture.

## 1. Mission

Build a unified standalone software suite that converts a single RGB remote-sensing image into a high-fidelity elevation product and an interactive analytical 3D environment. Non-georeferenced PNG/JPG imagery produces an rDSM. Georeferenced GeoTIFF imagery produces an absolute metric DSM while preserving geospatial integrity. Every official requirement is mapped to implementation and verification evidence.

## 2. Immutable product contract

1. Accept PNG, JPG/JPEG, TIFF and GeoTIFF RGB imagery.
2. Detect whether usable CRS/transform metadata exists.
3. For non-georeferenced input, output a relative DSM without pretending the values are absolute metres.
4. For georeferenced input, output an absolute DSM in metres in a standard geospatial raster format.
5. Use a robust pretrained monocular-depth foundation model as the initial geometric prior.
6. Correct the remote-sensing domain gap instead of exposing the foundation model output directly.
7. Support metric calibration with low-resolution DEM evidence (including SRTM 30 m), sparse GCPs, scene statistics and semantic ground/object priors.
8. Preserve CRS, affine geotransform, NoData policy and provenance in geospatial outputs.
9. Project source RGB texture onto a generated terrain/surface mesh.
10. Support interactive first-person, free-fly and aerial navigation.
11. Support structural height probing, slope analysis, elevation profiles and reference comparison.
12. Provide RMSE, MAE and correlation against reference DSM/LiDAR, plus diagnostic metrics.
13. Demonstrate stability across urban, sparse, hilly and forested scenes.
14. Operate as a unified standalone application, not disconnected notebooks.
15. Ship complete source code, reproducible build/training/evaluation instructions and technical/user documentation.

## 3. Final scientific architecture

### 3.1 Input canonicalization

The ingest layer records raster dimensions, data type, band count, CRS, transform, GSD, NoData, radiometric statistics and a SHA-256 provenance hash. RGB band mapping is explicit. Source pixels are never destructively overwritten; normalized tensors and original radiometry coexist.

Preprocessing uses robust percentile normalization with train/inference parity, optional dynamic-range conversion for high-bit-depth imagery, invalid-pixel masking, deterministic resizing/tiling, and sensor/GSD metadata tokens when available.

### 3.2 Foundation geometry prior

Primary foundation prior: **Depth Anything 3 Monocular Large (DA3MONO-LARGE)**, subject to final local reproducibility verification and model-license capture in the repository. It is used as a pretrained monocular geometric prior, not treated as an absolute satellite-height estimator.

Required baselines include Depth Anything V2 and Metric3D v2 where technically reproducible. The baseline harness must run each method on identical tiles/splits and record runtime, memory, RMSE, MAE and correlation.

### 3.3 Remote-sensing height network

The final estimator is a dual-evidence model with these permanent branches:

- **Foundation geometry branch:** relative depth/geometry prior from the pretrained monocular model.
- **Remote-sensing appearance branch:** multi-scale RGB feature encoder optimized for overhead imagery, texture, shadows, roof/terrain morphology and vegetation patterns.
- **Metadata/context embedding:** GSD and available acquisition/geospatial metadata encoded as conditioning features; missing metadata is represented explicitly rather than guessed.
- **Cross-scale fusion:** bidirectional attention/gated fusion between geometry and remote-sensing appearance at multiple resolutions.
- **Semantic-structure auxiliary head:** predicts ground/building/vegetation/water/other structure probabilities when supervision exists; masked multi-task learning permits datasets without every label.
- **Height-distribution head:** combines continuous regression with height-range classification/ordinal structure to reduce long-tail underestimation of rare tall objects.
- **Boundary/normal head:** preserves roof edges, ridgelines, cliffs and surface gradients.
- **Uncertainty head:** predicts pixelwise aleatoric uncertainty used both in user-facing confidence and metric calibration weighting.

The network outputs a remote-sensing relative surface-height field, semantic probabilities, surface-gradient/normal cues and uncertainty. Absolute geodetic elevation is produced only by the metric-evidence calibration subsystem.

### 3.4 Evidence-calibrated metric elevation

For georeferenced imagery, relative height is transformed into an absolute DSM using a physically constrained evidence hierarchy:

1. Reproject/co-register SRTM or another low-resolution DEM to the image footprint when supplied.
2. Infer ground-confidence regions from semantic and structural evidence.
3. Build reliable DEM anchors preferentially from ground/low-structure cells to avoid double-counting buildings/canopy.
4. Add sparse GCP elevation anchors when supplied.
5. Fit a robust positive global scale and vertical offset using confidence-weighted Huber/IRLS estimation.
6. Fit only a smooth low-frequency terrain bias field; high-frequency object structure remains controlled by the image-derived height network.
7. Reject/downweight anchors with high residual, low semantic-ground probability, NoData, severe slope mismatch or high model uncertainty.
8. Produce calibration residual statistics and a calibration-confidence score.

Conceptual form:

`DSM(x) = T_anchor(x) + alpha * H_relative(x) + beta + B_lowfreq(x)`

where `T_anchor` is the terrain/elevation anchor field when available, `alpha` and `beta` are robust calibration parameters, and `B_lowfreq` is a heavily regularized spatial correction. Exact parametrization is selected by validation, but the evidence hierarchy and constraints are fixed.

For GCP-only calibration, sparse elevation points define the vertical datum/scale and the low-frequency base field through robust spatial interpolation subject to smoothness constraints. For SRTM+GCP mode, GCPs receive highest reliability and SRTM provides broad spatial support.

### 3.5 Non-georeferenced rDSM

PNG/JPG inputs with no spatial metadata produce a dimensionless relative surface model. Output values retain ordering and structural relief but are never labeled metres. The UI must visibly state `Relative elevation — no geodetic scale`. A TIFF/NPY numerical product and PNG preview may be exported; no false CRS is inserted.

### 3.6 Large-image inference

Large rasters are processed window-by-window with overlap. Required controls:

- deterministic tile grid;
- overlap halo sufficient for transformer context;
- cosine/Hann weighted overlap blending;
- overlap-based inter-tile offset/scale harmonization;
- edge-aware seam diagnostics;
- bounded RAM/VRAM scheduling;
- resumable job manifest so an interrupted long scene does not restart from zero;
- global metadata/provenance retained after mosaicking.

### 3.7 Training losses

Final objective is a weighted multi-task loss containing:

- robust height regression loss (Huber/Smooth-L1);
- scale/shift-aware relative-geometry loss for structural consistency;
- gradient loss in x/y;
- surface-normal consistency loss;
- class-balanced height-bin/ordinal loss for the long-tailed height distribution;
- optional semantic cross-entropy/focal loss where semantic labels exist;
- heteroscedastic negative log-likelihood for uncertainty;
- overlap/seam consistency loss for tiled scenes;
- metric-anchor consistency loss on georeferenced supervised samples.

Loss weights are selected using a predeclared validation objective dominated by the official SIH metrics, not by visual preference.

## 4. Data program

Development must use geographically disjoint and sensor-diverse data. Planned dataset registry:

- DFC2019 / WorldView-3 scenes with LiDAR/nDSM reference.
- DFC2023 building-height/DSM data where licensing and access are confirmed.
- ISPRS Vaihingen and Potsdam for very-high-resolution aerial domain diversity.
- GBH for multi-city/global building-height diversity.
- PHDataset/PhiSat-2 for recent cross-continent optical-satellite generalization, treated as secondary supervision because its labels are compiled from multiple public sources.
- Additional public RGB + LiDAR/DSM scenes only after license and georegistration quality checks.
- Official ISRO reference repository monitored continuously; as of 25 Aug 2026 its public repository contains only a README and no released dataset files.

### Split policy

No random adjacent-patch leakage. Train/validation/test splits are geographic at scene/city level. A second cross-sensor holdout contains at least one sensor/domain never used for supervised training. Terrain labels tag urban, sparse, hilly and forested scenes so every official stability requirement has dedicated results.

### Quality gates

Every sample receives registration-quality, NoData fraction, resolution, height-range and provenance metadata. Samples with obvious RGB/DSM misregistration are excluded from primary scoring and retained only for robustness experiments.

## 5. Evaluation contract

### Official primary metrics

- RMSE in metres.
- MAE in metres.
- Pearson correlation coefficient between valid prediction/reference elevations.
- Terrain-wise breakdown for urban, sparse, hilly and forested holdouts.

### Mandatory diagnostics

- mean signed bias;
- median absolute error;
- 90th/95th percentile absolute error;
- error by height range (0–5 m, 5–15 m, 15–30 m, >30 m; thresholds configurable per dataset);
- error by semantic class where labels exist;
- slope MAE in degrees;
- gradient/edge preservation score;
- tile-seam discontinuity metric;
- calibration-anchor residuals;
- uncertainty calibration / error-versus-confidence curves;
- runtime, peak CPU RAM, peak GPU/accelerator memory and throughput.

### Baselines/ablations

The final technical report must include at minimum:

- foundation monocular prior with naive normalization;
- foundation prior + simple affine calibration;
- published remote-sensing monocular-height baseline(s) such as HTC-DC Net and RDAH-Net where reproducibly runnable;
- final full system;
- final system without semantic priors;
- without uncertainty weighting;
- without DEM/GCP evidence calibration;
- without long-tail height head;
- without global tile harmonization;
- backbone substitution experiment.

No claim of superiority is permitted without a same-split, same-resolution comparison or a clearly labeled literature-only comparison.

## 6. Interactive 3D analysis platform

The 3D layer is an analytical application, not a decorative viewer.

### Rendering

- Three.js/WebGL/WebGPU-compatible rendering path.
- Terrain/surface mesh generated from the estimated DSM/rDSM.
- Source RGB orthotexture projected through deterministic UV mapping.
- Quadtree/tiled level-of-detail mesh for large scenes.
- frustum culling and texture mipmaps.
- first-person, orbit, top-down and unrestricted free-fly cameras.
- height exaggeration control explicitly labeled so it cannot be mistaken for metric truth.

### Analysis tools

- cursor elevation probe;
- reference elevation probe when a reference raster is loaded;
- prediction-reference residual at point;
- two-point horizontal/3D distance measurement;
- structural-height measurement relative to local ground estimate;
- slope at point/selected region;
- elevation profile along a drawn transect;
- contour overlay;
- hillshade/slope overlay;
- uncertainty/confidence overlay;
- reference DSM overlay;
- absolute-error/residual heatmap;
- synchronized 2D/3D cursor location;
- metadata/provenance panel;
- camera bookmark/export for documentation.

### UX performance targets

Targets are engineering acceptance goals to be measured before being claimed publicly:

- first interactive frame within 5 s after mesh assets are prepared for a standard evaluation scene;
- 60 FPS preferred, >=30 FPS minimum sustained navigation on the designated finale laptop for the standard scene;
- no blocking full-raster upload into browser memory for large GeoTIFFs;
- zero unhandled crashes in a 2-hour scripted interaction/scene-switch soak test;
- recoverable errors with actionable messages for malformed/unsupported rasters.

## 7. Application architecture

### Desktop shell

- Tauri desktop shell for a low-overhead standalone application.
- React + TypeScript frontend.
- Three.js rendering engine.
- Python inference/geospatial sidecar packaged as a local process.
- Local HTTP/IPC API bound only to loopback with per-session token.
- No cloud dependency for core processing after model weights are installed.

### Python core

- Python 3.12.
- PyTorch for model execution/training.
- Rasterio/GDAL + pyproj for geospatial IO/reprojection.
- NumPy/SciPy for calibration/evaluation.
- Pydantic contracts for jobs/configuration.
- MLflow local tracking for reproducible experiments.

### Permanent service boundaries

`ingest -> preprocess -> geometry_prior -> height_model -> metric_calibration -> geospatial_export -> validation -> mesh_assets -> desktop_analysis`

Every stage writes a provenance record with input hashes, model checkpoint ID, config hash, elapsed time and warnings.

## 8. Standard outputs

For a georeferenced project:

- `dsm.tif` — float32 metric DSM preserving CRS/geotransform.
- `confidence.tif` — float32 confidence/uncertainty layer.
- `slope.tif` — derived slope raster.
- `residual.tif` — if reference is available.
- `metrics.json` — official and diagnostic metrics.
- `calibration.json` — anchors, method, residuals and quality indicators.
- `provenance.json` — source/model/config hashes and environment.
- tiled mesh/texture assets for 3D rendering.
- human-readable validation report.

For non-georeferenced projects:

- `rdsm.tif` or equivalent numeric raster with no false CRS.
- confidence layer.
- mesh/texture assets.
- provenance and processing report.

## 9. Reliability and security

- Validate extension, MIME/raster readability, band count and dimensions.
- Enforce bounded job resource limits.
- Use temporary/work directories with atomic finalization.
- Never overwrite source data.
- Sanitize filenames and project paths.
- No arbitrary shell interpolation from user input.
- Verify model weight hashes.
- Preserve exact dependency lockfiles.
- Detect CRS/transform inconsistencies before DEM/GCP fusion.
- Abort metric calibration if anchor geometry is underdetermined rather than emit fabricated metres.

## 10. Repository engineering standard

Monorepo with permanent production modules; no throwaway demo tree. Required quality tooling:

- `ruff`, `pyright`, `pytest` for Python;
- frontend lint/type/unit tests;
- Rust `clippy`/tests for the Tauri shell;
- CI on every merge;
- deterministic small raster fixtures in CI;
- heavyweight model/benchmark integration suite run on designated accelerators;
- benchmark artifacts stored by commit/checkpoint.

## 11. Proposed differentiation that must be proven, not merely claimed

The system deliberately goes beyond a generic monocular-depth-to-mesh pipeline through:

1. **Geodetic evidence calibration:** confidence-weighted SRTM/GCP/semantic anchoring for true DSM elevation rather than simple depth visualization.
2. **Remote-sensing-specific geometry fusion:** pretrained depth prior is fused with overhead-image appearance, scale/GSD context and structural semantics.
3. **Uncertainty-aware calibration and analysis:** uncertain pixels influence anchor weighting and are visible to analysts.
4. **Cross-sensor generalization program:** geographic/sensor holdouts and sensor-style augmentation explicitly target hidden ISRO evaluation imagery.
5. **Globally consistent large-raster tiling:** overlap blending plus scale/offset harmonization suppresses tile seams and local scale drift.
6. **Evidence-native 3D:** the 3D environment exposes reference/error/confidence/measurement tools, turning the visualization layer into a validation and analysis platform.

These are proposed differentiators. Literature research must continue and no “first-ever” claim may be made without verification.

## 12. Disaster-management positioning

The primary product remains elevation reconstruction. Disaster-management value is expressed through supported analytical workflows, not unrelated feature creep: rapid slope/relief interpretation where current high-resolution elevation data is unavailable; structural-height and terrain inspection; route/terrain awareness; post-event imagery comparison when reference elevation is available; and export of standard geospatial layers into downstream GIS/disaster workflows.

## 13. SIH traceability and acceptance

The repository maintains `docs/requirements-traceability.yaml`. A requirement is complete only when all four fields exist:

`official clause -> implementation module -> automated/manual verification -> evidence artifact`

No presentation claim is allowed unless its evidence artifact exists.

## 14. Submission engineering

The official six-slide PDF is generated from evidence produced by the final system. Content budget:

- Slide 1: exact SIH metadata.
- Slide 2: proposed solution, problem coverage, genuine differentiation.
- Slide 3: architecture/methodology and technology stack.
- Slide 4: feasibility, engineering risks, mitigations and measurable validation plan.
- Slide 5: disaster-management impact plus scalability/sustainability/future progression.
- Slide 6: strongest references, datasets and benchmark evidence.

No paragraphs; use concise points, diagrams, benchmark figures and real system screenshots when available.

## 15. Execution principle

Speed comes from avoiding rework, not reducing the product. Independent lanes—data, ML, calibration, evaluation, 3D, desktop, documentation—work in parallel against frozen contracts. Mature libraries are used for commodity engineering; original effort is concentrated on the model/calibration/generalization/validation areas that affect SIH scoring.

## 16. Locked UI/UX system

UI quality is an official scoring axis, not post-processing polish. The permanent desktop workstation follows the VeilGraph-derived design discipline approved for DepthWizard:

- near-white/neutral application chrome;
- restrained deep/medium blue primary accent;
- scientific color ramps appear in the data canvas, not decorative application furniture;
- generous whitespace, fine borders, subtle shadows, moderate radii and precise alignment;
- one disciplined sans-serif family and tabular figures for measurements;
- fixed spacing scale `4/8/12/16/24/32/48`;
- persistent top bar, narrow tool rail, dominant central canvas, contextual right inspector and compact status line;
- workflow state remains visible as `Import -> Inspect -> Reconstruct -> Calibrate -> Validate -> Explore -> Export` without turning these into disposable product versions;
- deterministic processing language (`Geometry inference`, `Metric calibration`, `Mesh construction`) rather than decorative AI copy;
- no fake metrics, fake terrain, mock scientific overlays or placeholder reconstructions in the final application;
- camera modes: Orbit, Fly, First Person, Top Down;
- analytical layer modes: Optical/Texture, DSM, Slope, Confidence, Reference and Residual;
- validation comparison: Split, Swipe, Difference and Overlay with synchronized navigation;
- official RMSE/MAE/correlation remain immediately visible when reference validation exists;
- all controls are contextual and task-related; no gradient-heavy startup styling, neon/cyber visual language, oversized cards, random bento layouts, excessive glassmorphism or ornamental AI motifs.

Every screen must pass the acceptance questions: professional scientific-software credibility, hierarchy understandable in ~3 seconds, only necessary controls visible, scientific data receives visual emphasis, and the result remains credible as an ISRO-grade workstation.

## 17. Implementation ledger — 25 Aug 2026

The permanent repository has moved from specification into implementation. Current verified foundations:

- geospatial raster inspection and explicit RGB band handling;
- deterministic radiometric normalization;
- reference/SRTM reprojection onto the exact prediction grid;
- truthful non-georeferenced rDSM export with no invented CRS/metric units;
- CRS/geotransform-preserving float GeoTIFF DSM export;
- robust positive-scale Huber/IRLS metric calibration;
- semantic/uncertainty-aware DEM anchor weighting and low-frequency terrain-bias correction;
- official RMSE/MAE/Pearson metrics plus bias/percentile diagnostics;
- slope derivation and slope-error metrics;
- residual GeoTIFF and machine-readable validation report generation;
- deterministic large-raster tile grid and overlap blending primitive;
- textured metric terrain mesh and GLB/LOD-pyramid generation;
- file/config hashing and project-manifest primitives;
- geographic-leakage and cross-sensor-holdout enforcement in dataset registry;
- permanent monocular geometry-prior interface;
- DA3MONO-LARGE adapter boundary converting camera depth to truthful dimensionless relative height;
- loopback-only local FastAPI core with optional session token and constrained CORS;
- Tauri/React/Three.js application shell, design tokens, inspector, workflow status, camera controls and real GLB loader path;
- Python CI/testing infrastructure and architecture-decision records.

The current Python suite passes all implemented unit/integration tests. Heavy-model runtime verification and final frontend/Tauri compilation are environment-dependent gates and must be completed on the designated development/finale machines before any deployment-performance claim is made.
