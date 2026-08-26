# Release Train 1 — Production Elevation Runtime

## Scope

This release train converts the existing scientific components into one production project runtime.
It is not a demo pipeline. The same durable job graph is consumed by the local service and desktop.

Implemented in this train:

- evidence-locked estimator selection with calibrated DA3 as the current safe production path;
- learned-refiner eligibility only after an independent frozen promotion PASS;
- atomic schema-v2 project manifests with source/config hashes, stage state, artifact hashes,
  warnings/errors and resumability;
- truthful non-georeferenced rDSM completion with no invented CRS or metre claim;
- georeferenced reconstruction that pauses in `waiting_for_calibration` instead of fabricating
  absolute scale;
- DEM-only, GCP-only and DEM+GCP metric calibration modes;
- explicit GCP polarity diagnosis and positive-scale fitting;
- model-native confidence export with semantics that do not pretend it is calibrated probability;
- metric DSM, slope, calibration and provenance products;
- single-worker local inference queue to bound CUDA/MPS contention;
- durable manifest and job-status API contracts;
- desktop project submission, status polling, geometry reuse and DEM-resume path;
- unit/integration coverage with a deterministic fake prior so CI does not require model downloads.

## Production invariants

1. Source rasters are never overwritten.
2. A learned model cannot silently replace DA3 because it performed well on development data.
3. Geometry artifacts cannot be reused after a geometry-affecting config or estimator-policy change.
4. Completed projects are immutable under their scientific processing config.
5. A waiting georeferenced project can add calibration evidence without recomputing geometry.
6. Missing confidence, GSD or metric evidence is omitted/rejected with an explicit warning/status;
   it is never synthesized.
7. Product semantics and hashes are recorded in the durable manifest and provenance.

## Acceptance before this train is merged

- `make verify` passes with tests, Ruff and Pyright green;
- `make frontend-build` passes;
- deterministic runtime tests cover relative completion, evidence wait/resume, DEM calibration,
  GCP calibration, estimator fallback and configuration-drift rejection;
- service contract tests pass without loading DA3;
- one local Apple-MPS production smoke is performed only after static/unit acceptance is green.

Release Train 1 is not considered complete merely because the backend compiles. The final acceptance
requires the real local model/runtime smoke and a review of its generated manifest/products.
