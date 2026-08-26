# Release Train 2 — Scientific Validation and Analytical Workspace

## Mission

Turn the production elevation runtime into an analyst-grade evidence workspace without allowing
reference data to leak back into reconstruction, calibration, or model promotion.

This train is a vertical product integration, not a benchmark-only side path. The same persisted
project artifacts are consumed by the validation engine, local API, desktop raster workspace,
inspector, and later 3D analytical tools.

## Integrated scope

- explicit project reference-DSM ingestion after metric reconstruction;
- exact-file guard preventing the calibration DEM from being reused as claimed independent
  evaluation evidence;
- deterministic reference alignment onto the prediction grid with alignment mode recorded;
- elevation RMSE, MAE, mean bias, median absolute error, P90, P95 and Pearson correlation;
- slope RMSE, MAE and P95 error on the physical prediction grid;
- persisted aligned reference and signed prediction-minus-reference residual GeoTIFFs;
- machine-readable `metrics.json` and human-readable `validation-report.md`;
- confidence/error reliability diagnostics only when a real confidence artifact exists;
- explicit unavailable state when confidence is absent; synthetic confidence is prohibited;
- validation stage, hashes, semantics and warnings recorded in the durable project manifest;
- authenticated local API for validation, validation reload and bounded raster previews;
- production desktop reference-selection workflow;
- real Optical, DSM, Slope, Reference, Residual and Confidence raster preview plumbing;
- project inspector populated from the current project's reference evidence rather than a separate
  benchmark card;
- preservation of separate held-out benchmark evidence as a distinct scientific context.

## Scientific invariants

1. Reference values are evaluation-only. They are never passed into reconstruction, estimator
   selection, metric calibration, or promotion logic.
2. A reference file that is byte-identical to the calibration DEM is rejected.
3. Passing the exact-file check does **not** prove geographic, temporal, sensor, or acquisition
   independence. The human report states this limitation explicitly.
4. Residual sign is permanently defined as `prediction - reference` and tagged in the GeoTIFF.
5. Reference alignment method, reference SHA-256 and prediction SHA-256 are persisted.
6. A completed project validation is immutable for its chosen reference. A different reference must
   be evaluated in a separate project copy so prior evidence is not overwritten.
7. Confidence is interpreted according to its persisted semantics. Model-native confidence is not
   presented as calibrated probability.
8. Validation failure does not rewrite a successfully reconstructed DSM; it is recorded as a
   validation-stage failure with the production project evidence preserved.
9. Preview rendering is bounded and read-only. It never alters the underlying geospatial products.

## Acceptance before merge

- all Python unit/integration tests, Ruff and Pyright pass;
- desktop TypeScript/Vite production build passes;
- GitHub Python 3.12, Python 3.13 and desktop jobs are green;
- synthetic exact-grid tests recover known elevation residuals and zero slope error under a constant
  vertical offset;
- calibration-DEM reuse is rejected;
- second-reference overwrite is rejected;
- raster preview tests emit valid PNGs and reject absent layers;
- local service API validates a generated metric project and reloads the same report;
- one real project validation smoke is run only against a genuinely separate reference surface;
- generated `metrics.json`, residual/reference GeoTIFFs, manifest stage and human report are reviewed;
- no result from the Joshimath calibration DEM itself is misrepresented as independent validation.

## Train boundary

Release Train 2 does not claim Gate B complete by itself. Gate B also requires geography-disjoint
terrain breakdown, cross-sensor holdout, identical-input baseline comparison, required ablations,
error-confidence reliability where confidence exists, and tile-seam diagnostics. This train builds
the permanent production machinery that will host those evidence products rather than fabricating
them before the data exists.
