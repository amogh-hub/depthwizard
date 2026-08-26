# ADR 0004 — Production estimator policy and resumable project runtime

**Status:** Accepted  
**Scope:** Elite final-system integration / Release Train 1

## Decision

DepthWizard production processing is evidence-gated rather than model-preference-gated.

The default operational estimator is the calibrated `DA3MONO-LARGE` geometry path. A learned
refinement branch may replace that path only after a frozen, independent promotion contract passes.
Development validation or same-geography improvement is insufficient. The current frozen V4 branch
is explicitly not promoted because `potsdam-external-v2` completed with slightly worse aggregate
RMSE/MAE than independently calibrated DA3 and failed the predeclared per-tile non-degradation rule.

The production project runtime is one durable job graph. It owns ingest, geometry reconstruction,
estimator selection, evidence calibration, standard product export, provenance and resumability.
The desktop and CLI/service surfaces must consume this same runtime rather than duplicate scientific
logic.

## Project-state contract

`project-manifest.json` is an atomic schema-versioned state record. It contains source identity,
geometry/run configuration hashes, estimator decision/evidence, stage states, artifact hashes and
semantics, warnings and failures.

Geometry-affecting parameters are immutable once geometry has completed. A georeferenced project is
allowed to stop in `waiting_for_calibration` after producing a truthful rDSM, then resume later with
DEM/GCP evidence without recomputing geometry. A completed project is immutable under its processing
configuration; changed scientific inputs require a new project directory rather than silently
rewriting evidence.

## Metric-elevation contract

- Non-georeferenced imagery can complete only as relative elevation and receives no invented CRS or
  metre claim.
- Georeferenced imagery without DEM/GCP evidence pauses before metric elevation rather than
  fabricating scale.
- DEM-only, GCP-only and DEM+GCP calibration are explicit modes.
- DEM calibration frequency-matches the image-derived field when physical source/DEM GSD can be
  derived; if the ratio cannot be established it is not guessed.
- GCP calibration records and resolves relative-height polarity before a positive physical scale fit.
- DEM+GCP mode uses DEM for broad support, then GCPs as higher-reliability metric refinement.
- Weak or contradictory evidence raises an explicit failure.

## Confidence contract

The current foundation adapter can expose model-native confidence. Until an empirical reliability
calibration maps that signal to a probability/error interval, the product is named and tagged
`model_native_confidence_not_probability_calibrated`. Missing confidence is omitted with a warning;
values are never synthesized to satisfy a UI layer.

## Concurrency and recovery

The local service serializes production inference jobs through one worker to avoid simultaneous
CUDA/MPS model contention and uncontrolled accelerator-memory growth. Durable project state lives on
disk, not only in service memory. A sidecar restart may lose an in-memory job handle, but the project
can be resubmitted and resumed from its verified manifest/artifacts.

## Consequences

This architecture makes learned-model research non-blocking for the final product while preserving a
strict path for future promotion. Scientific failures remain visible evidence, expensive geometry is
reusable, source rasters are never overwritten, and every user-visible production claim can be
traced to persisted artifacts and promotion/calibration evidence.
