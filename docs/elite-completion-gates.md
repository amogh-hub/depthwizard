# DepthWizard Elite Final-System Completion Gates

DepthWizard is not considered complete, and submission-slide production does not begin, until every gate below has an evidence artifact and a PASS status. This file operationalizes the final-system contract in `MASTER_SPEC.md` and `docs/requirements-traceability.yaml`.

## Gate A — Production elevation estimator

- DA3 remains the frozen foundation geometry prior and the externally safe production baseline.
- The production estimator contains an explicit evidence-aware selector: calibrated DA3 is always available; a learned refinement branch may be selected only when its model/version has independent promotion evidence for the applicable operating domain.
- A learned branch that lacks independent promotion evidence must never silently replace the baseline. The selected path and reason are recorded in project provenance.
- DepthWizard's dual-evidence refinement research path fuses RGB overhead appearance, geometry prior and GSD/missing-metadata context at multiple scales and may emit relative surface height, semantic/structural context, distributional height information, normals/boundaries and aleatoric uncertainty where implemented and validated.
- Learned-model training retains robust height, gradient, normal, ordinal/semantic, heteroscedastic and seam/evidence terms where supervision exists.
- Checkpoint provenance, config hash, training command, promotion status and selector policy are recorded.
- The consumed Potsdam external-v2 result is immutable evidence: V4 is **not** promoted; its failure cannot be overridden by target-driven tuning on Potsdam-v2.

Evidence: estimator policy/manifest, `height_model_manifest.json`, training report, checkpoint SHA-256, external-promotion records.

## Gate B — Scientific validation

- Independent evaluation does not reuse the calibration source as the reference source.
- RMSE, MAE and Pearson correlation are reported.
- Geographic splits prevent adjacent-patch leakage.
- Urban, sparse, hilly and forested results are reported separately.
- Cross-sensor holdout is reported.
- DA3-only and published reproducible baselines are run on identical evaluation inputs.
- Required ablations are reported: refinement off, uncertainty weighting off, harmonization off, evidence calibration off and backbone substitution where feasible.
- Error-confidence reliability and tile-seam diagnostics are included.
- Negative results remain in the evidence record and cannot be rewritten as promotions.

Evidence: `domain_generalization_report.json`, `terrain_breakdown.csv`, `ablation_report.json`, `baseline_comparison.json`, frozen external benchmark records.

## Gate C — Metric calibration

- DEM-only calibration is operational and spatial-frequency matched to the DEM's effective resolution.
- GCP-only calibration is operational.
- DEM + GCP fusion is operational with GCPs receiving higher reliability.
- Weak/underdetermined evidence causes an explicit abort instead of fabricated metric elevation.
- Calibration residuals and confidence are emitted.
- High-frequency image-derived structure is preserved while only low-frequency terrain bias is corrected.

Evidence: `calibration_dem_report.json`, `calibration_gcp_report.json`, `calibration_fusion_report.json`.

## Gate D — Standard geospatial products

Georeferenced projects emit:

- `dsm.tif`
- `confidence.tif`
- `slope.tif`
- `residual.tif` when a reference is loaded
- `metrics.json`
- `calibration.json`
- `provenance.json`
- textured LOD mesh assets
- human-readable validation report

Non-georeferenced projects emit a truthful dimensionless rDSM with no invented CRS or metric units, plus confidence, mesh and provenance outputs.

Evidence: round-trip raster tests and complete project artifact manifest.

## Gate E — Unified production workflow

- One project job executes ingest -> preprocess -> geometry prior -> estimator selection/refinement -> metric calibration -> geospatial export -> validation -> mesh assets -> desktop analysis.
- Jobs are atomic and resumable.
- Large imagery uses bounded-memory tiling, overlap blending and harmonization.
- Source files are never overwritten.
- Every stage records timing, warnings, config hash, selected estimator path, calibration evidence and product paths.
- The desktop consumes this same production job graph; demo-only parallel pipelines are prohibited.

Evidence: `end_to_end_acceptance_report.json`, resumability test, malformed-input recovery test, project manifest.

## Gate F — Analytical 3D workstation

The final app has working, non-placeholder controls for:

- Optical, DSM, Reference, Residual and Confidence views
- Texture, DSM, Slope, Confidence and Residual layers
- Orbit, Fly, First-person and Top-down cameras
- elevation probe
- slope probe/region inspection
- structural-height measurement
- two-point distance
- elevation transect/profile
- reference comparison with split/swipe/overlay/difference
- synchronized prediction/reference/residual cursor
- confidence/error interpretation
- metadata/provenance inspection
- export

Evidence: interaction acceptance checklist and screenshots captured from real project artifacts.

## Gate G — Standalone deployment and stability

- Tauri application launches the packaged local scientific sidecar securely.
- Core processing works offline after model installation.
- Loopback/session-token boundary is enforced.
- Production app processes a fresh image, produces products, validates a reference and exports from a clean launch.
- Frontend build, Rust build/clippy/tests and Python quality gates pass.
- Standard-scene navigation sustains >=30 FPS on designated finale hardware.
- No unhandled crashes in the scripted soak test.

Evidence: `standalone_acceptance_report.md`, `software_stability_report.json`, `3d_navigation_benchmark.json`.

## Gate H — Documentation and reproducibility

- Installation and clean-build instructions are complete.
- User workflow and analyst tools are documented.
- Calibration semantics and limitations are documented.
- Training and evaluation commands are reproducible.
- Dataset licenses/provenance are recorded.
- Every presentation claim is traceable to an evidence artifact.

Evidence: reproducibility dry run and final documentation audit.

## Presentation lock

The six-slide SIH deck and final demo video are intentionally last. They are generated only after Gates A-H are PASS, using measurements and screenshots produced by the completed system. No slide claim is permitted to outrun the evidence repository.
