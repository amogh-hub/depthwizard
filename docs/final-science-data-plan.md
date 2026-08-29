# DepthWizard final SIH26175 four-terrain data plan

This document defines the preferred data-acquisition plan for the final independent RMSE/MAE/correlation campaign. It is deliberately conservative: the reference DSM must never be the same raster, same bytes, or calibration evidence used to produce the prediction.

## 1. Core rule

For every evaluated scene keep three roles separate:

1. **RGB input** — the single-view optical image given to DepthWizard.
2. **Calibration evidence** — a lower-resolution DEM or limited GCP evidence used only to map the monocular relative surface to metric elevation.
3. **Independent reference** — LiDAR/authoritative DSM used only after the prediction bytes have been frozen.

The final evaluator already refuses prediction/reference identity and calibration/reference byte reuse. The campaign operator must additionally avoid conceptual leakage: do not derive calibration evidence by downsampling the same reference DSM that will later be used as truth.

## 2. Preferred terrain sources

### Urban

**Primary:** ISPRS Potsdam.

Why:

- true RGB orthophoto tiles;
- absolute DSM tiles;
- source and DSM on the same published grid;
- very high spatial resolution;
- dense buildings, roads and vegetation make it suitable for structural-height and projection checks.

Use at least one tile that has a publicly available DSM. Preserve the published world-file/georeferencing exactly. Do not silently repair a one-row/one-column mismatch; explicitly align or reject it under the same rules as the rest of DepthWizard.

**Second urban geography when feasible:** ISPRS Vaihingen. This is useful for stability evidence because it is geographically distinct from Potsdam and has a different urban morphology.

Calibration evidence for Potsdam/Vaihingen should come from a separate lower-resolution global DEM or sparse GCP evidence, never from the evaluation DSM.

### Sparse / open landscape

**Preferred:** NEON Central Plains Experimental Range (CPER).

Use matching acquisition/year products:

- NEON high-resolution orthorectified RGB camera mosaic — DP3.30010.001;
- NEON LiDAR Elevation DSM — DP3.30024.001.

CPER is predominantly shortgrass/herbaceous rangeland with very low canopy height, making it a defensible sparse/open-landscape test site.

For stronger stability evidence, add a second geographically disjoint open/rangeland NEON site after verifying the selected year has both RGB and LiDAR DSM products.

### Hilly / mountainous landscape

**Preferred:** NEON Niwot Ridge (NIWO).

Use matching acquisition/year RGB and LiDAR DSM tiles. NIWO spans a steep Southern Rocky Mountains elevation gradient and includes subalpine/alpine terrain, making it a strong hilly/mountain test rather than a synthetic slope fixture.

For stronger stability evidence, add a second geographically disjoint mountainous NEON site with matched RGB/DSM availability.

### Forested landscape

**Preferred:** NEON Harvard Forest (HARV).

Use matching acquisition/year RGB and LiDAR DSM tiles. HARV is dominated by deciduous/evergreen/mixed forest with substantial canopy height, so the DSM contains genuine above-ground vegetation structure rather than bare terrain only.

For stronger stability evidence, add a second geographically disjoint forest NEON site with matched RGB/DSM availability.

## 3. Reference and calibration separation

For every final scene:

- evaluation reference: authoritative DSM/LiDAR DSM;
- calibration: separate coarse DEM or GCP evidence;
- preferred coarse DEM: SRTM/Copernicus or another independently sourced terrain DEM that does not contain the evaluation DSM surface;
- never use a downsampled copy of the evaluation DSM as calibration evidence;
- never use a reference-derived nDSM/CHM as calibration evidence when the parent DSM is the evaluation truth;
- record every calibration-evidence SHA-256 in the prediction manifest.

This separation is particularly important for NEON because its release contains both DTM and DSM products. A different filename is not automatically independent evidence. For the final claim, prefer a separately sourced coarse DEM rather than deriving calibration from the same LiDAR acquisition used for the reference DSM.

## 4. Spatial preparation contract

Each selected evaluation tile must satisfy:

- RGB is genuinely single-view optical input;
- reference is metric elevation in metres;
- CRS is known and parseable;
- reference vertical units are metres;
- no silent resampling or geometry repair occurs before registry freeze;
- any required reprojection is deterministic and documented;
- nodata/validity masks are preserved;
- geographic group is unique across train/validation/test boundaries;
- reference and RGB licensing/citation information is recorded in `license_id` and notes.

Do not crop a reference after looking at model errors. Define the evaluation window before prediction/reference comparison.

## 5. Recommended minimum final registry

For literal SIH26175 closure, the `test` split must contain at least:

| Scene role | Terrain | Preferred source |
|---|---|---|
| test | urban | ISPRS Potsdam |
| test | sparse | NEON CPER |
| test | hilly | NEON NIWO |
| test | forested | NEON HARV |

For stronger final judging evidence, use two geographically distinct scenes per terrain rather than stopping at the minimum four.

The broader DepthWizard novelty campaign additionally requires at least one `cross_sensor_test` scene with sensor lineage separated from the development/training sensor lineage. That is an extra research/generalization claim, not a reason to weaken or delay the four official SIH26175 terrain categories.

## 6. Prediction freeze order

The correct order is mandatory:

1. freeze source head and checkpoint identity;
2. freeze the dataset registry and scene windows;
3. reconstruct/calibrate every prediction without opening the final reference in the evaluator;
4. SHA-256 every prediction;
5. write/freeze the prediction manifest including all calibration-evidence paths;
6. only then run the reference evaluator;
7. preserve all scene/terrain/sensor reports unchanged, including negative results.

Run:

```bash
python scripts/evaluate_final_science_campaign.py \
  /path/to/frozen-registry.yaml \
  /path/to/frozen-predictions.yaml \
  artifacts/final-science
```

Expected artifacts:

- `domain_generalization_report.json`;
- `terrain_breakdown.csv`;
- `scene_metrics.csv`;
- `sensor_breakdown.csv`.

The SIH26175 completion gate consumes `artifacts/final-science/domain_generalization_report.json` by default.

## 7. Claim boundary

Do not claim that a calibration demo is an independent benchmark. Do not claim model superiority from the final campaign unless the same-input baseline protocol actually supports that comparison. The problem statement requires accurate validated DSM estimation and stable performance across four landscape types; that is the first priority.
