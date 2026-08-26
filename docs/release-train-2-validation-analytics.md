# Release Train 2 — Scientific Validation and Analytical Workspace

## Mission

Turn the production elevation runtime into an analyst-grade evidence workspace without allowing
reference data to leak back into reconstruction, calibration, or model promotion.

This train is a vertical product integration, not a benchmark-only side path. The same persisted
project artifacts are consumed by the validation engine, local API, desktop raster workspace,
inspector, measurement/profile tools, comparison workspace, and later 3D analytical tools.

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
- normalized-coordinate point probing against persisted DSM/rDSM, slope, reference, residual and
  confidence products;
- deterministic analyst transects with physical distance where georeferencing permits;
- point-to-point measurement with explicit horizontal and endpoint vertical deltas;
- profile sampling with minimum/maximum elevation and cumulative gain/loss;
- production desktop reference-selection workflow;
- real Optical, DSM, Slope, Reference, Residual and Confidence raster preview plumbing;
- prediction/reference swipe comparison with synchronized click sampling;
- project inspector populated from the current project's reference and analytical evidence rather
  than a separate benchmark card;
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
9. Preview rendering and analyst sampling are bounded and read-only. They never alter underlying
   geospatial products or scientific evidence.
10. A measured endpoint elevation delta is reported as an analyst-selected surface difference. It is
    not silently promoted to a building-height or object-classification claim.

## Local integrated acceptance scene

The final local RT2 smoke is intentionally separated from both the consumed Potsdam-v2 protocol and
the Joshimath calibration-only engineering demonstration.

`make release-train-2-validation-smoke` uses:

- **optical source:** TUM OrthoLoC `urban_residential_DOP.tif`;
- **metric-calibration surrogate:** TUM OrthoLoC `urban_residential_xDSM.tif`, deterministically
  downsampled by 16× before production calibration;
- **downstream evaluation only:** TUM OrthoLoC `urban_residential_DSM.tif`, fetched only after the
  production DSM has completed.

The first RT2 acceptance implementation attempted to locate the OrthoLoC demo footprint in global
Terrarium web tiles. That is not a safe assumption for dataset-local georeferencing: the local smoke
reported an implausible 543,780-tile footprint and aborted before reconstruction. The corrected
acceptance therefore does **not** invent a global location. It uses a coarse same-scene xDSM-derived
calibration surrogate strictly for product-path integration and records that this calibration source
is **not lineage-independent** from the downstream OrthoLoC evaluation DSM.

The evaluation DSM is deliberately not downloaded until the production metric DSM is complete.
Calibration-source, coarse-calibration and evaluation-reference SHA-256 values are recorded in
`release-train-2-acceptance.json`, together with explicit `lineage_independent: false` semantics.

This is an **integrated product acceptance**, not new model-promotion or independent accuracy
evidence. The `urban_residential` OrthoLoC scene has prior DepthWizard research use, so its result
must not be presented as unseen Gate B evidence. No threshold, estimator policy or model selection
may be tuned from this smoke.

## Acceptance before merge

- all Python unit/integration tests, Ruff and Pyright pass;
- desktop TypeScript/Vite production build passes;
- GitHub Python 3.12, Python 3.13 and desktop jobs are green;
- synthetic exact-grid tests recover known elevation residuals and zero slope error under a constant
  vertical offset;
- calibration-DEM reuse is rejected;
- second-reference overwrite is rejected;
- raster preview tests emit valid PNGs and reject absent layers;
- point probes and deterministic profile/transect sampling are covered by automated tests;
- local service API validates a generated metric project and reloads the same report;
- `make release-train-2-validation-smoke` completes with the coarse xDSM calibration surrogate and
  the regular OrthoLoC DSM held strictly downstream for evaluation;
- generated `release-train-2-acceptance.json`, `metrics.json`, residual/reference GeoTIFFs, manifest
  stage and human report are reviewed;
- the smoke is never described as calibration/evaluation lineage-independent evidence;
- no result from the Joshimath calibration DEM itself is misrepresented as independent validation;
- no RT2 acceptance result is used to reopen or tune the consumed Potsdam-v2 protocol.

## Train boundary

Release Train 2 does not claim Gate B complete by itself. Gate B also requires geography-disjoint
terrain breakdown, cross-sensor holdout, identical-input baseline comparison, required ablations,
error-confidence reliability where confidence exists, and tile-seam diagnostics. This train builds
the permanent production machinery that will host those evidence products rather than fabricating
them before the data exists.
