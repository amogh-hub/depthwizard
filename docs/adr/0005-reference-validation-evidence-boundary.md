# ADR 0005 — Reference validation is downstream, immutable evidence

**Status:** Accepted  
**Scope:** Elite final-system integration / Release Train 2

## Decision

DepthWizard treats a reference DSM as evaluation evidence, never as reconstruction or calibration
input. The permanent project workflow therefore separates metric calibration from validation even
when both operations occur in the same desktop session.

The validation engine loads the already persisted metric DSM, aligns the supplied reference onto the
prediction grid, computes metrics and writes residual/reference products. No reference values are
returned to the geometry prior, estimator selector, metric fit, learned-refiner policy, or promotion
gates.

## Independence guard

If the supplied reference SHA-256 equals the DEM SHA-256 recorded by metric calibration, validation
is rejected. This prevents the most direct form of circular evidence: scaling against one DEM and
then reporting performance against the same file.

A different SHA-256 is only an exact-file distinction. It is not sufficient evidence of geographic,
temporal, sensor, lineage, or acquisition independence, so the generated human report explicitly
states that limitation.

## Persistence

The first completed reference validation for a project is immutable. `metrics.json`,
`validation-report.md`, aligned reference, residual raster and manifest stage record the reference
and prediction hashes. A later attempt using a different reference is rejected rather than silently
overwriting the earlier scientific record.

## Residual and confidence semantics

Residual is defined as `prediction - reference` in metres and is tagged accordingly. Confidence
reliability is computed only if a persisted confidence artifact exists. Model-native confidence
remains model-native; reliability analysis does not rename it as calibrated probability.

## Consequences

The desktop can provide interactive reference/residual analysis while retaining a defensible audit
trail. Negative validation remains visible, and the system cannot improve a reported result by
feeding the evaluation target back into the production estimator.
