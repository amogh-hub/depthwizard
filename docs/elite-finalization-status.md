# Elite finalization status

This record separates implemented source capability from evidence that must be produced on the
exact release commit. It is intentionally fail-closed: an implementation check does not become a
scientific, clean-machine, performance, or stability claim.

## Implemented in the elite-hardening integration

- calibration gates for correlation, independent support, spatial coverage, metric relief, anchor
  RMSE, normalized RMSE, robust-fit convergence and conditioning;
- GCP minimum count, duplicate/cluster/collinearity rejection, convex-hull coverage and leave-one-out
  validation;
- DEM + GCP fusion that retains DEM-established relief scale and limits GCP correction to one robust
  global vertical-datum offset;
- physical-support-derived low-frequency correction defaults with an explicit metres-to-pixels
  override contract;
- explicit vertical CRS/datum/elevation-reference metadata and a visible distinction between metric
  calibrated elevation and datum-resolved absolute elevation;
- source NoData propagation through tiled inference, confidence, DSM/rDSM and recorded valid fraction;
- slope derived from the complete local pixel-to-ground Jacobian for rotated/sheared grids;
- Spearman correlation and NMAD in elevation validation, plus final-campaign vertical-datum
  compatibility enforcement;
- exact DA3 checkpoint-byte verification before model load and inclusion of the verified snapshot in
  the standalone resource tree for offline-first reconstruction;
- bounded queue admission/history, duplicate-active-project protection, cooperative cancellation and
  durable cancelled state across service, runtime and desktop;
- disposal of late/stale Three.js GLTF resources during terrain replacement;
- frozen Python/npm/Cargo CI consumption (`uv sync --frozen`, `npm ci`, locked Cargo) with no implicit
  lock regeneration;
- content/config identity checks for derived Potsdam RGB and DA3 geometry caches.

## Required before the words “final elite system complete” are used

1. Commit the integration and rerun all Python, frontend and Rust gates on that exact commit.
2. Build the standalone sidecar and application with the real pinned DA3 snapshot; prove a fresh
   offline reconstruction with no pre-populated model cache.
3. Run the frozen final-science v2 campaign on geographically disjoint urban, sparse, hilly and
   forested scenes plus a cross-sensor scene. Freeze prediction bytes before reference evaluation.
4. Run same-input baselines and the declared ablations. Preserve negative results.
5. Complete operator interaction evidence, projection QA, structural-height checks and real
   screenshots from persisted project artifacts.
6. Record finale-hardware FPS, peak memory/latency, the two-hour packaged soak and a clean-machine
   install/reopen/export run.
7. Run `scripts/check_sih26175_completion.py` on the exact release commit and require every gate to
   pass before producing the final deck or making accuracy/superiority claims.

The first item is ordinary repository verification. Items 2–6 require the real model payload,
external datasets, designated hardware and/or human observation; they cannot be truthfully replaced
by synthetic fixtures or source inspection.
