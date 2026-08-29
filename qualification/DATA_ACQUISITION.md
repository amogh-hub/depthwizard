# Final-science data acquisition map

This file is operational guidance for the frozen `339bdf...` release. It is not production source and must not be merged into the qualified branch.

## Urban — ISPRS Potsdam

Official dataset page:

https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

Official benchmark download landing page:

https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx

The dataset provides RGB true orthophotos and absolute DSMs on the same UTM grid at 5 cm GSD.

### Fresh-scene rule

Do **not** use the four already-consumed Potsdam external-v2 tiles as the new final campaign scene:

- `2_10`
- `3_13`
- `5_11`
- `6_14`

Select a different Potsdam tile and freeze that scene/window before the final-science prediction is generated. The earlier four remain valid historical external evidence but should not be reframed as untouched final-campaign data.

Calibration evidence must be separately sourced (for example Copernicus GLO-30 or sparse GCPs), never a downsample of the Potsdam DSM.

## Sparse/open — NEON CPER

Site: Central Plains Experimental Range (`CPER`).

RGB product:

https://data.neonscience.org/data-products/DP3.30010.001

LiDAR DSM product:

https://data.neonscience.org/data-products/DP3.30024.001

Use a common acquisition month/year for the RGB and DSM. The data products are tiled; choose one predeclared 1 km window and preserve its CRS/nodata metadata.

## Hilly/mountain — NEON NIWO

Site: Niwot Ridge (`NIWO`).

Use the same product pair:

- RGB: `DP3.30010.001`
- LiDAR DSM: `DP3.30024.001`

Choose one common-acquisition tile/window before evaluating model error.

## Forested — NEON HARV

Site: Harvard Forest (`HARV`).

Use the same product pair:

- RGB: `DP3.30010.001`
- LiDAR DSM: `DP3.30024.001`

Prefer a corrected/released LiDAR acquisition and record the exact release/month. Preserve genuine canopy surface height in the DSM.

## NEON API

Official API documentation:

https://data.neonscience.org/data-api/

Site/product metadata can be queried without authentication; sustained/data download use may require a NEON API token. Determine the latest **common acquisition month** for both RGB and DSM at each site instead of hard-coding mismatched years.

NEON data are distributed under CC BY 4.0 according to the official API/data-use documentation. Record the exact product code, site, acquisition month, release, filename and downloaded SHA-256 in the final evidence notes.

## Cross-sensor holdout — ISPRS Vaihingen

Official ISPRS semantic-labeling benchmark landing page:

https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx

Use a sensor label distinct from the training-lineage sensor and assign a geographic group that appears in no other split.

## Calibration evidence separation

For every final evaluation scene maintain three independent roles:

1. optical RGB input;
2. calibration evidence (coarse DEM or limited GCPs);
3. evaluation-only DSM/LiDAR reference.

Never derive role 2 from role 3, even by downsampling or renaming.

## Freeze sequence

1. select exact scenes/windows and licenses;
2. write the final registry;
3. generate all production metric DSM predictions without using final reference values in the evaluator;
4. create the draft prediction manifest;
5. run `scripts/freeze_final_science_manifest.py` to freeze checkpoint/prediction/calibration-evidence identities;
6. only after the freeze report passes, run `scripts/evaluate_final_science_campaign.py` against the references;
7. preserve all outputs unchanged, including negative results.
