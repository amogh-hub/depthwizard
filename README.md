# DepthWizard

DepthWizard is the SIH26175 engineering repository for **single-view height estimation and 3D flythrough** from optical remote-sensing imagery.

The repository is being built as the final scientific/geospatial system rather than as a disposable hackathon prototype. Core requirements include geospatially honest relative-vs-metric elevation handling, robust calibration, deterministic large-raster processing, validation evidence, and an analytical 3D desktop workflow.

## Current engineering gates

Run the Python quality gate with:

```bash
make verify
```

For the frozen ISPRS Potsdam external benchmark, the historical strict v1 run is preserved and the active acceptance protocol is **external-v2**, which changes only the reference metadata contract after a documented one-column official-file discrepancy:

```bash
make potsdam-contract-audit
make potsdam-external-acceptance
```

Protocol details:

- `docs/potsdam-external-benchmark-v1.md` — immutable aborted historical run;
- `docs/potsdam-external-benchmark-v2.md` — sealed metadata-contract correction used by the current external acceptance target.

## Product modes

- **Non-georeferenced RGB** → dimensionless relative DSM (rDSM), never fake metres.
- **Georeferenced RGB** → absolute DSM in metres when adequate independent calibration evidence exists.
- **3D analysis** → textured terrain/structure mesh with measurement, profile, validation, confidence and geospatial export workflows as implementation gates are completed.

## Scientific discipline

DepthWizard keeps development, calibration and evaluation evidence separate. External targets are not used to tune checkpoint selection, reference-adaptive blending, metric scale/orientation, or post-hoc promotion thresholds. Failed or aborted frozen evaluations remain part of the evidence record rather than being silently rewritten.

See `MASTER_SPEC.md` for the authoritative final-system engineering direction.
