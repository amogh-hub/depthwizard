# ADR 0006: Scene-global monocular mosaicking

## Status

Accepted for corrective engineering on `engineering/scene-global-da3-mosaic-fix-188c511`.

## Context

The operator-workstation qualification of a 6000 x 6000 ISPRS Potsdam urban scene exposed a regular
rounded-square elevation pattern. The same pattern was visible in the persisted DSM, amplified in the
slope raster, and then rendered as false mounds, pits, and cliff-like faces in 3D. Independent reference
validation confirmed that the defect existed upstream of mesh construction.

The production DA3 adapter was converting camera depth to a dimensionless relative-height field inside
each 1024 x 1024 inference call. That conversion clipped each tile to its own P01-P99 range and mapped
it independently toward 0-1 before scene assembly. Consequently, a locally flat tile could be assigned
the same normalized dynamic range as a genuinely high-relief tile. Overlap harmonization then attempted
to reconcile already-rescaled tiles, allowing tile-local contrast and context to become a persistent
mosaic imprint.

## Decision

1. DA3 tile inference returns **affine-preserving height evidence** (`-depth`) rather than a tile-local
   dimensionless normalization.
2. Overlap harmonization operates on that affine evidence before any scene-relative clipping or scaling.
3. Full affine tile scaling is treated as a high-risk correction. It is accepted only when:
   - overlap support is sufficient;
   - both overlap fields contain robust variation;
   - overlap predictions have positive Pearson correlation >= 0.35;
   - the fitted scale remains within [0.25, 4.0]; and
   - affine correction improves median overlap error by at least 10% over offset-only alignment.
   Otherwise, only the robust median offset is applied.
4. After all tiles are harmonized and feather-blended, one P01-P99 normalization defines the
   dimensionless relative-height convention for the complete scene.
5. A near-constant assembled scene remains flat. The normalizer must never manufacture relief when the
   robust scene span is effectively zero.
6. Metric calibration remains downstream and unchanged. The reference DSM remains evaluation-only and
   is never used to tune or construct the corrected prediction.
7. Tile-local height normalization is forbidden for production mosaicking and is regression-tested.

## Scientific invariants

- Higher relative-height values continue to mean surfaces closer to the nadir camera / higher terrain.
- rDSM remains dimensionless and makes no metre claim before evidence calibration.
- DEM/GCP calibration remains the only path to metric elevation.
- RGB radiometric normalization remains scene-global and independent of elevation normalization.
- Confidence remains model-native and is not represented as a calibrated probability.
- The correction must not use the evaluation reference raster as reconstruction evidence.

## Qualification consequences

This is a geometry-affecting source change. Any exact-head science, operator, rendering, soak, or
standalone evidence that depends on generated geometry must be requalified against the new source SHA.
The original Potsdam operator evidence remains valuable as defect evidence but cannot be used to claim
that the corrected geometry has passed.

## Rejected alternatives

### Increase mesh smoothing

Rejected because the repeated pattern is already present in the persisted DSM. Smoothing the renderer
would hide evidence rather than repair the reconstruction.

### Tune against the official Potsdam DSM

Rejected because the official DSM is independent evaluation evidence. Using it to select reconstruction
parameters would collapse the calibration/reference boundary.

### Increase overlap only

More overlap can improve context averaging, but it does not repair the fundamental error of defining a
new relative-height scale inside every tile. Overlap changes may be evaluated later, after the
scene-global normalization correction is independently qualified.

### Disable overlap harmonization

Rejected because monocular predictions can carry legitimate per-tile affine drift. The correct policy is
to preserve affine evidence and make harmonization conservative, not to remove the alignment mechanism.
