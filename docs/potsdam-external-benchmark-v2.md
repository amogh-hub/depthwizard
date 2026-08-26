# Potsdam External Benchmark v2

External-v2 is a **metadata-contract correction only** after the sealed external-v1 run aborted on an official file-shape discrepancy. It does not change the model, calibration evidence, scene membership, metrics, or promotion criteria.

## Why v2 exists

External-v1 sealed protocol SHA-256:

`6452480ae7cc55d63eff5cab9bf6b449bfb00222591e79070854c57dc6e35923`

After that seal existed, v1 evaluated tile `2_10` and then aborted while opening the reference DSM for tile `3_13`. A read-only metadata audit established:

- RGB `3_13`: height `6000`, width `6000`;
- DSM `3_13`: height `6000`, width `5999`;
- RGB/DSM affine transform: identical;
- RGB/DSM world file: identical;
- the reference therefore covers the same upper-left grid but ends one native 5 cm column early at the trailing east edge;
- the audit decoded **no DSM pixel values**.

The other three frozen pairs are exactly `6000 x 6000` and metadata-aligned.

V1 is preserved as an aborted historical protocol. V2 does not rewrite or delete its seal or evidence.

## Frozen scientific decisions retained from v1

External-v2 keeps exactly:

- frozen tiles: `2_10`, `3_13`, `5_11`, `6_14`;
- frozen DepthWizard V4 `confidence-gated-v2` checkpoint;
- frozen DA3 MONO Large foundation baseline;
- Copernicus DEM GLO-30 as **calibration-only** evidence;
- ISPRS Potsdam DSM as **evaluation-only** reference;
- 25 cm benchmark grid;
- identical DA3/V4 metric calibration procedure;
- RMSE, MAE and Pearson correlation reporting;
- the original strict refiner promotion rule.

The promotion rule remains:

1. all four frozen tiles must evaluate successfully;
2. aggregate V4 RMSE must be lower than aggregate DA3 RMSE;
3. aggregate V4 MAE must be lower than aggregate DA3 MAE;
4. no individual tile may have worse V4 RMSE than DA3.

The already observed v1 result for `2_10` is not used to alter any threshold or policy.

## V2 reference metadata contract

Before any reference DSM pixel value is loaded, v2 inspects and seals metadata for **all four** reference files.

A reference is accepted only when:

- its CRS resolves to EPSG:32633;
- its affine transform is identical to the paired RGB transform;
- therefore its upper-left origin and native 5 cm spacing are unchanged;
- it does not extend beyond the RGB footprint;
- it is missing at most **one native pixel** on a trailing right/bottom edge.

A shift in origin, different pixel size, rotation, leading-edge crop, larger raster, or deficit greater than one native pixel is rejected.

This tolerance formalizes the metadata-only discrepancy that stopped v1. It is not inferred from DSM heights.

## Partial-edge evaluation policy

V2 does **not** fabricate or extrapolate the missing `3_13` reference column.

The native DSM is reprojected to the same 25 cm benchmark grid using the same area-average path as v1. In addition, v2 constructs a geometric full-footprint mask from the sealed reference bounds. Any 25 cm benchmark cell whose complete footprint is not covered by the reference DSM is excluded from evaluation.

For the observed one-column native deficit, this conservatively excludes the affected trailing 25 cm benchmark column rather than evaluating a partially supported cell.

## Protocol sealing

V2 uses a new immutable directory:

`artifacts/evaluation/potsdam-external-v2/`

The new protocol seal is:

`artifacts/evaluation/potsdam-external-v2/protocol_seal.json`

The final report is:

`artifacts/evaluation/potsdam-external-v2/potsdam_external_report.json`

The v2 seal hashes both the unchanged v1 core evaluator used for inference/calibration and the v2 metadata-contract adapter, plus the shared Potsdam, calibration and model implementation sources.

The evaluator refuses to start a fresh v2 execution if a v2 seal already exists. If a sealed v2 run aborts, that run remains consumed evidence and any code-changing correction must use a later protocol version rather than deleting the seal.

## Non-consuming preflight

Before external-v2 is sealed, run a preflight that validates the historical v1 seal, the frozen tile files, RGB contracts, reference metadata contracts, Copernicus calibration file, and frozen V4 checkpoint. It computes the candidate v2 protocol digest but **does not** create the v2 seal and **does not** decode Potsdam DSM values:

```bash
make potsdam-external-v2-preflight
```

A successful preflight must explicitly report:

- historical v1 seal verified;
- v2 seal does not already exist;
- all four frozen reference metadata contracts accepted;
- candidate v2 protocol SHA-256;
- Potsdam DSM pixel values read: `NO`;
- v2 protocol seal created: `NO`.

## Commands

Run the repository quality gate first:

```bash
make verify
```

The read-only metadata audit can be repeated without decoding DSM pixels:

```bash
make potsdam-contract-audit
```

Then run the non-consuming v2 preflight:

```bash
make potsdam-external-v2-preflight
```

Only after both verification and preflight pass, run the corrected sealed external benchmark once:

```bash
make potsdam-external-acceptance
```

The historical strict v1 evaluator remains available as:

```bash
make potsdam-external-v1
```

Do not delete or mutate the existing v1 seal.

## Interpretation discipline

A successful external-v2 run would establish a strong independent **urban airborne cross-dataset** result. It still does not by itself complete DepthWizard Gate B. Terrain-diverse sparse/hilly/forest evidence, further identical-input baselines, ablations, seam/reliability diagnostics and broader cross-sensor evidence remain required.
