# DepthWizard judge evidence pack

This is the compact index for SIH26175 reviewers. It distinguishes implemented capability from
evidence and prevents a slide, demo, or release claim from outrunning the exact tagged build.

## Product in one sentence

DepthWizard converts one RGB remote-sensing image into a truthful dimensionless rDSM when geodetic
scale is unavailable, or an evidence-calibrated metric DSM when valid georeferencing plus DEM/GCP
evidence are available, and exposes the persisted result in a measurable Three.js terrain
workstation.

## Architecture and algorithm

- One-page standalone architecture: `docs/standalone-architecture.md`
- Geospatial truth and metric-claim boundary: `docs/adr/0002-geospatial-truthfulness.md`
- Production estimator/runtime: `docs/adr/0004-production-estimator-and-project-runtime.md`
- Scene-global tiling: `docs/adr/0006-scene-global-monocular-mosaicking.md`
- Exact SIH requirement mapping: `docs/sih26175-problem-statement-traceability.md`

Production uses pinned DA3MONO-LARGE only as a relative geometry prior. DEM and/or spatially
distributed GCP evidence establishes metric scale through robust positive-scale Huber/IRLS fitting,
coverage/correlation/conditioning/residual gates and leave-one-out GCP validation. DEM + GCP fusion
keeps DEM-established relief and permits GCPs to validate/correct only a robust global vertical-datum
offset. A weak fit fails closed.

Six GCPs are the normal minimum. An expert API caller may explicitly request a four- or five-point
minimum; the resulting calibration evidence is marked `low_confidence` and the project records a
warning.

For non-georeferenced projects, distance/profile tools normally report pixels. An analyst may enter
an explicit metres-per-pixel horizontal scale; DepthWizard labels the resulting distance as analyst
scaled and keeps every vertical rDSM value dimensionless.

## Input and output contract

| Input | Elevation claim | Required evidence | Primary output |
|---|---|---|---|
| PNG/JPG without spatial metadata | Relative only; never metres | None | float32 `rdsm.tif`, preview, provenance and optional GLB |
| TIFF/GeoTIFF without defensible calibration | Relative/uncalibrated | None | rDSM or waiting-for-calibration project |
| Georeferenced GeoTIFF + accepted DEM/GCP | Metric DSM | Passed calibration gates | float32 `dsm.tif`, slope where valid, calibration/provenance and GLB |
| Metric DSM + independent reference | Evaluation only | Separate reference identity | metrics, aligned reference, residual and validation report |

Confidence is exported only when the selected estimator produces a defined native confidence field.
The production DA3 path may truthfully report confidence unavailable; no uncertainty raster is
synthesized.

## Data, benchmarks and licensing

- Frozen campaign protocol: `docs/final-science-campaign.md`
- Data acquisition/separation policy: `docs/final-science-data-plan.md`
- Third-party baseline boundary: `docs/third-party-baselines.md`
- Model identity: `model-manifests/da3mono-large.yaml`
- Dependency inventory command: `python -m scripts.generate_supply_chain_reports`

The final headline report must be regenerated on the exact candidate commit and committed under
`evidence/submission/` on `qualification/evidence-<candidate-SHA>`. The release workflow accepts only
the branch whose name matches the tagged SHA and records its qualification commit in the release
archive. The report must cover urban, sparse, hilly and forested standard tests plus a cross-sensor
holdout. Same-input baselines and ablations are separate evidence and are not implied by the
terrain-coverage report.

## Installation and offline quick start

For judges, use the `.dmg` and `SHA256SUMS` from the GitHub Release. Verify the checksum, install the
application, disconnect the machine, launch DepthWizard, import imagery, choose a project directory,
and select **Reconstruct**. Georeferenced projects pause before metric elevation until the operator
adds acceptable DEM/GCP evidence. Non-georeferenced projects remain relative.

For source verification:

```bash
uv lock --check
uv sync --frozen --python 3.12 --extra dev --extra ml
.venv/bin/python scripts/verify.py
cd apps/desktop && npm ci --no-audit --no-fund && npm test && npm run build
```

The core reconstruction is offline after the installer contains the verified model snapshot. Online
map tiles, cloud inference and undocumented Hugging Face caches are not required.

## Operator workflow

1. Import PNG/JPG/TIFF/GeoTIFF and review CRS, GSD, validity and source-quality warnings.
2. Reconstruct relative geometry.
3. For georeferenced inputs, add DEM, six-point GCP CSV, or both; inspect warnings and calibration.
4. Build terrain and use Orbit, Fly, First Person, Top Down, LOD and labelled Z exaggeration.
5. Probe elevation/slope, measure two points, draw a profile, or select a structural footprint.
6. Load an independent reference for registered comparison and residual metrics.
7. Use **Screenshot** in the 3D toolbar to export a PNG with the elevation mode, active layer,
   vertical exaggeration and exact packaged Git SHA burned into the image.
8. Export the hash-audited project bundle; close and reopen it to confirm persistence.

## Troubleshooting

- **Waiting for calibration:** expected for a georeferenced source without DEM/GCP evidence.
- **GCP rejection:** add spatially distributed points; avoid duplicates, clusters and collinearity.
- **Metric fit rejected:** inspect correlation, coverage, relief and residual gates; do not relabel the
  rDSM as metres.
- **Memory exhausted:** close large applications, reduce tile size, or process a smaller crop. The
  application reports this as `resource_exhausted` and does not modify the source.
- **Confidence unavailable:** the selected estimator emitted no native confidence; this is not an
  error and must not be replaced with fabricated values.
- **3D unavailable:** retain the original RGB source and rebuild the persistent mesh assets.

The packaged API is loopback-only, session-token protected and intentionally does not expose Swagger
outside development mode. Request/response schemas are defined in `src/depthwizard/contracts.py` and
mirrored by `apps/desktop/src/api.ts`.

## Known limitations

- Single-view monocular geometry cannot remove occlusion or building lean; tagged off-nadir imagery
  receives an explicit risk warning.
- Bright/low-chroma and deep-shadow diagnostics are conservative candidates, not semantic masks.
- Absolute accuracy is bounded by calibration/reference quality and vertical-datum compatibility.
- Native pixel uncertainty is unavailable on estimator paths that do not emit it.
- Large-raster and low-memory qualification must match the declared judging hardware.
- No learned refiner is production-promoted without independent same-input improvement evidence.

## Release proof

The tag-triggered workflow refuses to publish unless the tag equals authoritative `main`, every
language/source gate passes, all exact-commit evidence files pass the completion checker, the pinned
DA3 checkpoint is verified, the standalone sidecar/application build succeeds and checksums plus the
SBOM/license inventory are generated. See `docs/release-playbook.md`.
