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

## Acceptance

Release Train 1 acceptance is complete.

- GitHub CI passed on Python 3.12, Python 3.13 and the desktop frontend for head
  `844d2744abee8f8866a9e7b1b40ecf4077785ed3`.
- Local `make verify` passed with 97 tests, Ruff clean and Pyright reporting zero errors,
  warnings or information diagnostics.
- The desktop production build completed successfully.
- A real Apple-MPS production smoke completed on the 1024×1024 Joshimath scene using
  DA3MONO-LARGE, four reconstruction tiles, DEM calibration and the production runtime.
- The resulting project manifest completed every persisted stage with no errors and preserved the
  calibrated-DA3 estimator decision because the frozen V4 branch did not pass external promotion.
- Persisted artifact semantics and hashes are internally consistent across the project manifest and
  provenance document. The calibration JSON SHA-256 is
  `cbdb1986f6148782ebf48b45366c756557b2b0a25be4b297223131824a71ed2b`; the provenance JSON
  SHA-256 is `b0c20f50324b51b70f4cbba17a9c930bf89b78c47f3ddcea30e701574d519251`.
- The smoke emitted one deliberate warning: DA3 exposed no native confidence raster on this path,
  so the runtime omitted confidence rather than fabricating values.

The Joshimath run is an engineering/runtime acceptance smoke, not held-out scientific validation.
Its DEM-fit diagnostics are therefore preserved rather than marketed as accuracy evidence: the
frequency-matched anchor correlation magnitude was about 0.072 and the robust anchor RMSE was about
1031.7 m. Scientific validation quality and user-facing reliability diagnostics remain separate
release-train work; these values are not evidence of Joshimath DSM accuracy.
