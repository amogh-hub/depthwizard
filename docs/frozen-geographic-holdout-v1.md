# Frozen Geographic Holdout V1

This document predeclares the first **one-time untouched same-domain geographic holdout** for the DepthWizard learned height refiner and its calibration-evidence adaptive trust policy. It exists to prevent post-hoc scene selection or threshold tuning after holdout targets are observed.

## Frozen system

- Learned model: the existing V4 `confidence-gated-v2` checkpoint selected only from the fixed L06/L07 validation protocol.
- Operational trust policy: the existing V5 evidence-adaptive policy with candidate refinement weights `0.00, 0.25, 0.50, 0.75, 1.00`.
- Policy selection evidence: only the declared 64 sparse calibration anchors, internally cross-validated with 4 folds.
- Minimum anchor-CV improvement required before applying a learned correction: 1%.
- Near-best conservative band: 0.5%.
- Raster held-out evaluation pixels are never available to the blend-selection policy.

No model retraining, threshold change, candidate-weight change, scene-specific exception, or target-conditioned replacement is permitted inside this protocol.

## Geographic membership rule

Source split: official OrthoLoC `test_outPlace`, same-domain `R` scenes only.

Every location already used or inspected during model development is excluded before scene selection: `L01` through `L08`, plus `L50`. The evaluator then:

1. groups the remaining same-domain listing entries by location ID;
2. keeps only locations exposing at least two same-domain scenes;
3. sorts location IDs numerically;
4. selects the first four eligible locations;
5. selects the first two filenames from each selected location.

Therefore the frozen evaluation contains **4 untouched locations × 2 scenes = 8 scenes**. Selection uses filenames/listing metadata only. If four eligible locations are unavailable, the run aborts **before any selected DSM/reference target is loaded**.

## Seal-before-target rule

Before loading any selected holdout DSM/reference, the evaluator writes `protocol_seal.json`. The seal hashes:

- selected scene IDs and locations;
- V4 checkpoint SHA-256 and model config;
- all adaptive-policy constants;
- sparse-anchor protocol constants;
- evaluator source SHA-256;
- adaptive-policy source SHA-256.

If the seal already exists and any of those values differ, the evaluator refuses to run. A modified system must use a new protocol version; deleting the seal to retest a changed system is scientifically invalid.

## Predeclared evidence condition

The same-domain holdout is marked PASS only when all eight scenes are successfully evaluated, aggregate adaptive RMSE is lower than DA3, aggregate adaptive MAE is lower than DA3, no individual scene has worse RMSE than DA3, and the learned refinement is actually used on at least one scene. Exact DA3 fallback is allowed on scenes where calibration evidence cannot defend a learned correction.

A failed condition is recorded as a scientific result, not converted into a software exception or repaired by choosing different scenes.

## Scope and limitation

Passing this holdout is **not Gate B completion**. It is a same-dataset, same-domain geographic generalization result conditional on 64 sparse height anchors. It is not zero-shot metric depth and it does not satisfy the independent cross-sensor/reference requirement by itself.

After results are observed, these eight scenes may not be used to tune V4/V5. If the model or policy is changed because of their outcomes, they become development evidence and a new external untouched benchmark is required for final claims.
