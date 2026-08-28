# DepthWizard final SIH science campaign

This protocol closes the official SIH26175 DSM-evaluation requirement without reusing calibration
evidence as ground truth or mixing geographically adjacent patches across train/test boundaries.

The campaign is an **evaluation-only** step. It never generates predictions, tunes calibration, or
opens reference DSM values before prediction generation. It consumes a frozen dataset registry plus
already-produced metric DepthWizard DSMs and produces the evidence artifacts required by the
problem-statement traceability contract.

## Required registry contract

Use the existing `depthwizard.data.registry.DatasetRegistry` schema.

A final campaign registry must contain:

- at least one `train` scene so the production sensor lineage is explicit;
- a geographically disjoint `test` split containing **urban, sparse, hilly and forested** scenes;
- at least one `cross_sensor_test` scene whose sensor does not occur in `train`;
- a reference DSM for every `test` and `cross_sensor_test` scene;
- a unique `geographic_group` assignment across splits so adjacent-patch leakage is impossible.

`DatasetRegistry.assert_integrity()` rejects duplicate scene IDs, geographic split leakage, and a
cross-sensor holdout that silently reuses a training sensor.

Relative registry file paths are resolved relative to the registry YAML.

## Prediction manifest

Create a YAML or JSON manifest describing the **already frozen** production predictions. The model
checkpoint SHA-256 is mandatory, and every prediction entry must contain the SHA-256 calculated
immediately after that prediction was produced and before any reference evaluation.

For example:

```yaml
schema_version: 1
model_id: calibrated-da3-production
checkpoint_sha256: 7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960
predictions:
  - scene_id: urban-test-01
    prediction_path: ../predictions/urban-test-01-dsm.tif
    prediction_sha256: 1111111111111111111111111111111111111111111111111111111111111111
    calibration_evidence_paths:
      - ../evidence/urban-test-01-srtm.tif
    prediction_vertical_units: m
    reference_vertical_units: m

  - scene_id: sparse-test-01
    prediction_path: ../predictions/sparse-test-01-dsm.tif
    prediction_sha256: 2222222222222222222222222222222222222222222222222222222222222222
    calibration_evidence_paths:
      - ../evidence/sparse-test-01-gcps.csv

  - scene_id: hilly-test-01
    prediction_path: ../predictions/hilly-test-01-dsm.tif
    prediction_sha256: 3333333333333333333333333333333333333333333333333333333333333333
    calibration_evidence_paths:
      - ../evidence/hilly-test-01-srtm.tif

  - scene_id: forested-test-01
    prediction_path: ../predictions/forested-test-01-dsm.tif
    prediction_sha256: 4444444444444444444444444444444444444444444444444444444444444444
    calibration_evidence_paths:
      - ../evidence/forested-test-01-srtm.tif

  - scene_id: cross-sensor-01
    prediction_path: ../predictions/cross-sensor-01-dsm.tif
    prediction_sha256: 5555555555555555555555555555555555555555555555555555555555555555
    calibration_evidence_paths:
      - ../evidence/cross-sensor-01-srtm.tif
```

The hashes above are illustrative placeholders only. Replace them with the real SHA-256 of the frozen
checkpoint and each frozen DSM. On macOS, for example:

```bash
shasum -a 256 /path/to/checkpoint
shasum -a 256 /path/to/prediction-dsm.tif
```

Every evaluation scene must have exactly one prediction entry, and the prediction manifest may not
contain extra scenes outside the registry's `test`/`cross_sensor_test` evaluation surface.

Prediction-manifest relative paths are resolved relative to the prediction manifest itself.

## Freeze-before-reference rule

The scientific ordering is mandatory:

1. freeze the production model/checkpoint;
2. generate all evaluation predictions without using reference DSM values;
3. SHA-256 every produced prediction and write those hashes into the prediction manifest;
4. freeze the prediction manifest;
5. only then run the reference evaluator.

At evaluation time, the runner re-hashes every prediction **before reading or aligning its reference**.
If any byte changed after the manifest was frozen, evaluation aborts instead of silently accepting a
different prediction set.

The checkpoint hash is an evidence identity for the production model used to generate the already
persisted predictions. The campaign runner is evaluation-only and does not reload or execute the
checkpoint itself.

## Independence rules

For every scene, the runner SHA-256 hashes:

- source RGB;
- prediction DSM and verifies it against the predeclared frozen prediction SHA;
- independent reference DSM;
- every declared calibration-evidence file.

Evaluation aborts if:

- the current prediction SHA does not equal the SHA frozen in the prediction manifest;
- the prediction path equals the reference path;
- prediction bytes are SHA-identical to reference bytes;
- a calibration-evidence path equals the reference path;
- calibration-evidence bytes are SHA-identical to reference bytes.

This catches post-freeze prediction mutation and reference/calibration reuse even when a file has been
copied or renamed.

The evaluator does not infer missing calibration evidence. Each prediction must explicitly declare at
least one calibration-evidence path because DepthWizard does not claim absolute metric elevation
without evidence.

## Run

From the repository root:

```bash
python scripts/evaluate_final_science_campaign.py \
  /path/to/frozen-registry.yaml \
  /path/to/frozen-predictions.yaml \
  /path/to/final-science-evidence
```

The default minimum overlap is 128 valid prediction/reference pixels per scene. A different threshold
must be protocol-justified and supplied explicitly:

```bash
python scripts/evaluate_final_science_campaign.py \
  registry.yaml predictions.yaml evidence/final-science \
  --min-valid-pixels 1024
```

## Generated evidence

The runner writes:

- `domain_generalization_report.json`
- `terrain_breakdown.csv`
- `scene_metrics.csv`
- `sensor_breakdown.csv`

`terrain_breakdown.csv` is based on the standard geographically disjoint `test` split only, so the
official urban/sparse/hilly/forested result is not diluted by cross-sensor experiments.

`domain_generalization_report.json` separately reports:

- frozen model/checkpoint identity;
- a passed prediction-identity-freeze status;
- pooled standard-test RMSE, MAE, Pearson correlation and mean bias;
- pooled cross-sensor RMSE, MAE, Pearson correlation and mean bias;
- per-terrain pooled metrics;
- per-sensor pooled metrics;
- every scene's full DepthWizard elevation metrics;
- valid-pixel coverage;
- RGB/prediction/reference/calibration-evidence SHA identities;
- registry and prediction-manifest SHA identities;
- the explicit evidence-independence status and claim boundary.

Pooled Pearson correlation uses numerically stable mergeable centered moments rather than averaging
per-scene correlation coefficients. Pooled RMSE and MAE are pixel-weighted, not scene-average
approximations.

## What this runner does not prove by itself

A successful run proves that the frozen prediction set was evaluated under the four-terrain,
cross-sensor, geographic-split, prediction-identity and reference-independence contracts. It does
**not** prove that the numeric results are good enough to win, and it does not create missing
benchmark scenes.

The following remain separate evidence surfaces:

- same-input DA3/published-baseline comparison;
- required ablations;
- confidence/error reliability analysis;
- large-raster seam consistency;
- structural-height reference benchmark;
- finale-hardware 3D/FPS/operator acceptance;
- two-hour stability soak;
- clean-machine reproducibility.

Those results must be reported as measured. Negative results remain part of the evidence record and
must never be rewritten as promotions.
