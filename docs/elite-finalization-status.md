# Elite finalization status

This is a fail-closed status record, not a marketing claim. A capability can be implemented while
its exact-release evidence is pending. Source changes invalidate older packaged, science, operator,
performance, soak and clean-machine evidence unless the applicable protocol says otherwise.

## Implemented in the release candidate

- literal PNG/JPG/TIFF/GeoTIFF ingest and truthful relative-versus-metric output policy;
- calibrated DA3 production selection with exact source/checkpoint identity and offline packaging;
- DEM, GCP and DEM + GCP calibration with robust positive-scale fitting, polarity resolution,
  correlation/coverage/relief/RMSE/conditioning/convergence gates and leave-one-out validation;
- six-GCP production default; explicit four/five-point expert overrides persist a low-confidence
  evidence classification and warning;
- DEM-frequency matching and low-frequency terrain correction that preserves image-derived surface
  structure, with separate preliminary-affine and final post-bias residual evidence;
- explicit vertical CRS/datum/elevation-reference metadata and fail-closed datum semantics;
- source masks/NoData through inference, output and metrics; full rotated/sheared ground Jacobian for
  slope;
- bounded sampled quality warnings for saturation, insufficient texture, bright/cloud-like,
  deep-shadow and metadata-declared off-nadir building-lean risk;
- persistent DSM/rDSM, previews, validation, residual, project export and textured GLB LOD assets;
- Orbit, Fly, First Person, Top Down, flythrough, probe, measure, profile, structural-height and
  registered reference/residual analysis;
- explicitly labelled analyst metres-per-pixel horizontal scaling for relative-project distance
  tools, without changing relative vertical semantics;
- provenance-burned 3D screenshot export carrying the exact packaged source SHA and display state;
- bounded queue/history, duplicate rejection, cooperative cancellation, crash recovery, actionable
  memory exhaustion and Three.js resource disposal;
- locked Python/npm/Cargo resolution, Linux source/desktop CI, portable macOS/Windows geospatial CI,
  deterministic SBOM/license inventory and split frontend vendor chunks;
- guarded tag workflow that can publish only from authoritative `main` after exact-commit completion
  evidence passes.

## Previously measured, but requiring regeneration for the final changed SHA

An earlier release-candidate commit completed the packaged RT5 workflow, literal input contract,
four-terrain/cross-sensor campaign, 27-item operator workflow, sustained >=30 FPS navigation, a full
two-hour packaged soak and fresh macOS ARM64 offline build/reopen qualification. The clean-machine
record remains immutable historical evidence on its qualification branch. Other transient evidence
was not durably committed and must not be reconstructed from notes.

## Blocking the final tag

1. Freeze the new release-candidate SHA and rerun all source/frontend/Rust checks.
2. Regenerate packaged RT5 and literal input reports on that SHA.
3. Regenerate and commit the independent urban/sparse/hilly/forested plus cross-sensor campaign.
4. Produce the same-input baseline comparison and declared ablation report. Preserve negative results.
5. Rerun and durably commit operator, projection, structural-height, sustained FPS and two-hour soak
   evidence.
6. Run the complete published-installer clean-machine rehearsal, including PNG, JPG, GeoTIFF,
   DEM/GCP, navigation, measurement, repeated jobs, export and reopen.
7. Place only authentic compact JSON/CSV/plots under `evidence/submission/` on the separate
   `qualification/evidence-<candidate-SHA>` branch, then require
   `scripts/check_sih26175_completion.py --strict` to pass from a clean candidate checkout.
8. Protect `main`, merge without tree drift, tag the exact SHA and let the guarded release workflow
   publish the installer, checksums, model manifest, evidence archive and supply-chain reports.

No release, slide or verbal claim may turn a pending item into a pass. The detailed sequence is in
`docs/release-playbook.md`; the reviewer-facing index is `docs/judge-evidence-pack.md`.
