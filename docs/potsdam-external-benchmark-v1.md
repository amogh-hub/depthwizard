# Potsdam External Benchmark v1

This protocol is DepthWizard's first frozen external cross-dataset DSM evaluation after OrthoLoC development was exhausted.

## Scientific separation

- **RGB / accuracy reference:** ISPRS 2D Semantic Labeling — Potsdam.
- **Metric calibration evidence:** Copernicus DEM GLO-30 Public 2021.
- **Foundation geometry:** frozen DA3 MONO Large.
- **Learned estimator:** frozen DepthWizard V4 `confidence-gated-v2` checkpoint.
- The ISPRS Potsdam DSM is **evaluation-only**. It is not used for scale, offset, orientation, blend selection, checkpoint selection, or any other model decision.
- The Copernicus GLO-30 raster is **calibration-only**. It is not used as the accuracy reference.
- Reference-adaptive V5 blending is deliberately disabled because its sparse-anchor policy would require observing the evaluation DSM.

This separation prevents the calibration source from being reused as the claimed independent accuracy reference.

## Public dataset contracts

ISPRS describes Potsdam as 38 true-orthophoto/DSM patches. RGB imagery is available as R-G-B TIFF, the DSM is 32-bit float, both are on the same WGS84 UTM grid, the native ground sampling distance is 5 cm, and every tile is 6000 x 6000 pixels. The official page is:

`https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx`

The independent calibration raster is the public Copernicus GLO-30 COG covering Potsdam:

`Copernicus_DSM_COG_10_N52_00_E013_00_DEM.tif`

Registry:

`https://registry.opendata.aws/copernicus-dem/`

DepthWizard downloads that one public COG directly from the no-account AWS public bucket when it is absent locally.

## Frozen scene membership

Before reference values are loaded, v1 seals these four Potsdam tiles:

- `2_10`
- `3_13`
- `5_11`
- `6_14`

The tiles were chosen before any Potsdam DSM values were observed by the evaluator. They are spatially separated within the public Potsdam mosaic rather than neighbouring patches.

## Evaluation grid

Native 5 cm imagery is area-averaged to a predeclared 25 cm benchmark grid. This keeps the full 300 m x 300 m tile footprint, which is important because a 30 m calibration product provides only about ten independent support cells per axis across one Potsdam tile. Evaluating tiny crops would create too little independent coarse-elevation evidence for defensible metric calibration.

The same 25 cm grid is used for:

- RGB inference;
- frozen DA3 relative geometry;
- frozen V4 relative refinement;
- Copernicus GLO-30 alignment;
- independent ISPRS DSM evaluation.

The native Potsdam DSM is loaded only after the protocol seal exists and is area-averaged onto that same grid.

## Metric calibration

DA3 and V4 are calibrated **independently but identically** against Copernicus GLO-30. The calibration path:

1. reprojects the coarse DSM onto the 25 cm target grid;
2. frequency-matches the relative field to the coarse source's effective 30 m support;
3. samples approximately one anchor per independent coarse support cell;
4. diagnoses vertical orientation from calibration evidence only;
5. fits robust positive scale and offset;
6. adds only a smooth low-frequency terrain/surface bias;
7. preserves the high-frequency image-derived structure in the raw relative field.

Weak or underdetermined calibration evidence rejects the tile rather than fabricating metric elevation.

## Metrics and promotion rule

For every evaluated tile and for the concatenated external aggregate, the evaluator reports:

- RMSE in metres;
- MAE in metres;
- Pearson correlation;
- DA3/V4 calibration diagnostics;
- V4 uncertainty versus absolute-error correlation;
- per-tile RMSE/MAE delta against DA3.

Before loading any reference target, v1 predeclares a strict learned-refiner promotion rule:

- all four frozen tiles must evaluate successfully;
- aggregate V4 RMSE must be lower than aggregate calibrated DA3 RMSE;
- aggregate V4 MAE must be lower than aggregate calibrated DA3 MAE;
- no individual tile may have worse V4 RMSE than DA3.

A failure stays in the evidence record. Tiles, thresholds, checkpoint, calibration source, or evaluator code are not changed post hoc to manufacture a pass.

## Data placement

The evaluator intentionally does not download the large ISPRS benchmark archive automatically. Obtain the official Potsdam data and extract the RGB and DSM TIFFs (with their `.tfw` files) anywhere beneath:

`data/external/isprs-potsdam/`

or set:

`DEPTHWIZARD_POTSDAM_ROOT=/absolute/path/to/extracted/potsdam`

The resolver accepts the standard RGB names such as `top_potsdam_2_10_RGB.tif` and standard DSM names such as `dsm_potsdam_02_10.tif`.

## Command

After the dataset files are present:

```bash
make verify
make potsdam-external-acceptance
```

The first successful preflight creates:

`artifacts/evaluation/potsdam-external-v1/protocol_seal.json`

before any selected Potsdam reference DSM is loaded. The final report is:

`artifacts/evaluation/potsdam-external-v1/potsdam_external_report.json`

## Gate-B interpretation

Even a successful Potsdam result is **not** a complete Gate-B pass. Potsdam is a strong external urban airborne dataset, but Gate B additionally requires terrain-diverse urban/sparse/hilly/forest reporting, published baselines on identical inputs, ablations, reliability/seam diagnostics, and broader cross-sensor evidence.
