# Frozen OrthoLoC Out-of-Place Location Holdout V2

## Why V2 exists

The original `ortholoc-frozen-same-domain-v1` protocol required four previously untouched
`test_outPlace` locations with at least two same-domain scenes per location. Its first execution
stopped during metadata-only selection because the remaining public split cardinality could not
satisfy that requirement after excluding every geography already used by DepthWizard training,
validation, and development.

That V1 stop occurred before a protocol seal was written and before any selected holdout
DSM/reference target was loaded. It therefore did not consume an untouched target result.

V2 does not pretend that the dataset contains more independent geography than it does. It defines
the strongest remaining same-domain OrthoLoC confirmation available without reusing an inspected
location: one previously untouched `test_outPlace` location with two deterministic same-domain
scenes.

## Pre-target selection contract

- Source split: `test_outPlace`.
- Previously used locations `L01` through `L08`, plus `L50`, are excluded.
- Cross-domain variants are excluded from this same-domain protocol.
- Eligible locations must expose at least two matched same-domain DOP/DSM scene names.
- The first eligible numeric location ID is selected.
- The first two filenames in that location are selected.
- Selection uses listing metadata only. No DSM/reference values may influence membership.
- No target-dependent scene replacement is allowed after the seal is written.

Before the first selected DSM/reference is loaded, the evaluator seals the selected scene IDs,
checkpoint SHA-256, model configuration, adaptive-policy parameters, and evaluator source hashes.
A changed contract must use a new protocol version rather than deleting or rewriting the seal.

## Predeclared acceptance rule

The V2 confirmatory result is PASS only when:

1. both sealed scenes evaluate successfully;
2. aggregate adaptive RMSE is strictly lower than DA3;
3. aggregate adaptive MAE is strictly lower than DA3;
4. neither individual scene has higher adaptive RMSE than DA3; and
5. the learned refinement is actually used on at least one scene rather than obtaining PASS by
   falling back to DA3 everywhere.

The adaptive policy still uses only the declared 64 sparse calibration anchors for trust selection.
Raster held-out pixels are never used to choose the refinement weight.

## Scientific scope

V2 is a one-location untouched out-of-place confirmation, not a multi-geography benchmark. It does
not close Gate B. Final Gate B evidence still requires a genuinely external untouched benchmark
with independent geography/reference and a cross-sensor holdout, plus the required terrain
breakdown, published baselines, ablations, uncertainty reliability, and seam diagnostics.

After V2 results are observed, that location becomes development evidence for any future model or
policy revision and must not be reused as an untouched result.
