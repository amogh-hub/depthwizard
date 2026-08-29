# SIH26175 problem-statement traceability and completion contract

This document maps the official SIH26175 problem statement to the concrete DepthWizard implementation and the evidence required before the project may be called **100% problem-statement complete**.

The governing rule is strict: **implemented is not the same as proven**. A UI control, source module, or passing unit test does not close an evaluation-facing clause unless the specified qualification evidence also exists.

## 1. Background challenges → DepthWizard response

| Official challenge | DepthWizard response | Closure evidence |
|---|---|---|
| Stereo/LiDAR/InSAR can be costly, sensor-dependent and computationally intensive | Single-view RGB pipeline driven by a pretrained monocular geometry prior | Exact-head packaged reconstruction acceptance |
| Foundational monocular models are trained mainly on egocentric imagery | Remote-sensing processing, geospatial metadata validation, tiling/harmonization and evidence calibration around the monocular prior | Exact-head packaged acceptance plus final terrain campaign |
| Monocular output is scale-agnostic/relative | Evidence calibration using lower-resolution DEM and/or sparse GCP evidence; fail closed when metric evidence is inadequate | DEM/GCP engineering acceptance plus packaged metric project |
| Static elevation products need operational 3D interpretation | Persistent LOD terrain, optical texture projection, analytical overlays, navigation and measurement | Human-visible operator acceptance plus sustained rendering evidence |

## 2. Input and elevation-estimation requirements

### Non-georeferenced RGB imagery

Official requirement: PNG/JPG without spatial metadata must produce a Relative Digital Surface Model (rDSM).

DepthWizard contract:

- literal PNG and JPG ingestion is qualification-tested;
- absence of CRS/usable georeferencing is preserved;
- no metric GSD or metric elevation is invented;
- DA3 relative geometry can be reconstructed and visualized directly;
- non-georeferenced output remains explicitly dimensionless/relative.

Required evidence:

- `PASS_SIH26175_INPUT_FORMAT_CONTRACT` for literal PNG/JPG ingestion;
- exact-head operator evidence for a completed PNG rDSM and completed JPG rDSM workflow.

### Georeferenced RGB imagery

Official requirement: georeferenced TIFF/GeoTIFF must produce an absolute DSM with metric height values.

DepthWizard contract:

- CRS and affine metadata are inspected rather than assumed;
- ground sample spacing is geodetically evaluated rather than blindly treating projected map metres as ground metres;
- a relative monocular prior is produced first;
- metric elevation is unavailable until adequate DEM/GCP evidence exists;
- the final DSM is emitted in metres in a standard geospatial raster product.

Required evidence:

- literal georeferenced TIFF input-format PASS;
- exact-head packaged metric reconstruction/calibration acceptance;
- exact-head operator metric DSM evidence.

## 3. Pretrained monocular backbone

Official requirement: use a robust pretrained monocular depth-estimation backbone to generate initial relative-depth maps.

DepthWizard production path uses DA3MONO-LARGE as the pretrained monocular prior. The packaged runtime contains an explicit dependency/import contract and frozen-runtime self-check. DepthWizard does not claim that the monocular prior alone is the final DSM; it is one geometric evidence source within the reconstruction pipeline.

Required evidence:

- exact-head frozen DA3 dependency/import self-check;
- packaged offline DA3 inference PASS;
- no claim that an unpromoted custom model beats DA3.

## 4. Scale calibration

Official requirement: map scale-agnostic monocular features to absolute elevation using scene statistics, lower-resolution DEMs such as SRTM, semantic priors, or minimal GCPs.

DepthWizard implements:

- DEM calibration;
- sparse GCP calibration;
- DEM + GCP fusion;
- evidence-quality checks and fail-closed underdetermined calibration;
- calibration provenance and artifact hashes.

The problem statement says a DEM source **such as SRTM**; SRTM itself is not mandatory when another legitimate lower-resolution DEM is used.

Required evidence:

- engineering acceptance for DEM, GCP and DEM+GCP modes;
- packaged absolute-DSM reconstruction using independent calibration evidence;
- no reuse of the final evaluation reference as calibration evidence.

## 5. 3D visualization layer

Official requirement: project the source optical image onto generated terrain and integrate with Unity, Three.js or Babylon.js as a navigable 3D environment.

DepthWizard uses Three.js and persistent GLB LOD terrain assets. The renderer preserves the source RGB texture, supports analytical textures and validates that a real rendered terrain frame exists before declaring readiness.

Required operator evidence:

- optical texture projected onto the terrain;
- DSM, slope, hillshade and contour analytical overlays;
- aerial and low-camera views with continuous, correctly oriented terrain;
- auto/manual LOD and display-only vertical exaggeration.

## 6. Navigation and analyst interaction

Official requirement: seamless first-person navigation and analysis of structural heights and slopes from arbitrary aerial perspectives.

DepthWizard provides:

- Orbit;
- Fly;
- terrain-clearance-constrained First Person;
- Top Down;
- deterministic Flythrough;
- 3D probe;
- geodesic two-point measurement with signed elevation delta;
- elevation transect/profile;
- analyst-defined structural footprint with robust local-ground estimation;
- slope analysis in 2D and projected 3D.

Required evidence:

- exact-head operator PASS for every navigation mode;
- exact-head 3D picking registration PASS;
- an urban structure-height measurement;
- slope analysis from useful aerial/3D perspectives.

## 7. DSM accuracy and validation — 50%

Official evaluation requires RMSE, MAE and correlation against LiDAR/reference truth with stability across **urban, sparse, hilly and forested** landscapes.

DepthWizard final-science evaluation is deliberately reference-isolated:

- geographically disjoint registry splits;
- frozen production prediction hashes before reference evaluation;
- independent reference rasters required;
- prediction/reference identity prohibited;
- calibration/reference byte reuse prohibited;
- exact alignment and valid-pixel masking;
- scene, terrain, sensor and pooled metrics.

The following are mandatory before the PS can be called complete:

- urban RMSE, MAE and Pearson correlation;
- sparse RMSE, MAE and Pearson correlation;
- hilly RMSE, MAE and Pearson correlation;
- forested RMSE, MAE and Pearson correlation;
- pooled test metrics;
- independent reference proof.

Joshimath calibration evidence or any DEM used to calibrate a prediction must not be presented as independent validation truth for that prediction.

## 8. Visualization quality and UX — 50%

Official evaluation covers projection accuracy, visual fidelity, navigability, intuitiveness, stability and standalone deployment.

DepthWizard closure requires:

- human-visible projection/texture registration acceptance;
- usable Texture/DSM/Slope/Hillshade/Contours rendering;
- Orbit/Fly/First Person/Top Down/Flythrough acceptance;
- Reference and Residual views after independent validation;
- structural-height, slope, probe, measure and profile analysis;
- export and project reopen;
- representative **>=30 FPS sustained** navigation evidence, not one rounded toolbar sample;
- two-hour packaged application/owned-sidecar stability soak;
- clean-machine packaged application qualification.

The sustained rendering evidence contract requires at least 60 seconds, at least 55 one-second samples, mean FPS >=30 and 5th-percentile FPS >=30 while navigation is exercised, with both aerial and low-camera positions and both texture and analytical-overlay rendering represented.

## 9. Expected integrated solution

Official expected solution: a unified software suite with complete source code and technical documentation, elevation estimation and an interactive platform that can upload imagery, visualize reconstructed terrain and validate heights against reference datasets.

DepthWizard closure evidence must prove:

- single integrated desktop application;
- PNG/JPG/TIFF ingestion;
- rDSM for non-georeferenced imagery;
- metric DSM for georeferenced imagery after valid evidence calibration;
- standard geospatial DSM output;
- source optical texture on 3D terrain;
- interactive navigation and analysis;
- reference comparison and residual validation;
- project export with integrity checking;
- relaunch/reopen persistence;
- packaged local scientific sidecar owned by the desktop;
- no user-visible terminal requirement;
- offline core after model installation;
- source and technical documentation present in the repository.

## 10. The single authoritative 100% completion command

First produce the literal input evidence:

```bash
python scripts/verify_problem_statement_inputs.py
```

Initialize the remaining human/external evidence records once:

```bash
python scripts/check_sih26175_completion.py --init-templates
```

Then run the fail-closed completion check:

```bash
python scripts/check_sih26175_completion.py --strict
```

DepthWizard is **100% SIH26175 problem-statement complete only when that command emits**:

```text
PASS_SIH26175_PROBLEM_STATEMENT_COMPLETE
```

The checker will refuse completion if any of these are absent or stale against the current exact head:

1. exact-head RT5 packaged standalone engineering acceptance;
2. literal PNG/JPG/TIFF input-format qualification;
3. independent four-terrain RMSE/MAE/correlation campaign;
4. full two-hour packaged stability soak;
5. complete human-visible operator workflow evidence;
6. sustained >=30 FPS representative rendering evidence;
7. clean-machine standalone evidence;
8. source/technical documentation.

Once all eight gates pass, stop product engineering for the problem statement. Remaining work is submission production and separately justified novelty evidence, not additional SIH26175 functionality.
