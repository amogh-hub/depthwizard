# DepthWizard final SIH26175 qualification order

This is the frozen finish line. It exists to prevent two opposite failures: declaring DepthWizard
complete before evidence exists, or continuing to add features after the official problem statement
and locked differentiators are already satisfied.

Once final evidence generation begins, **do not change source code unless a qualification step exposes
a concrete defect**. Any source-changing fix invalidates newer exact-head build/acceptance evidence and
requires the affected gates to be rerun.

## 0. Freeze one exact source head

- checkout `engineering/elite-finalization`;
- `git pull --ff-only`;
- record `git rev-parse HEAD`;
- require an empty worktree before qualification;
- every generated report must identify the exact source/build it evaluates where the corresponding
  harness supports that identity.

## 1. Freeze dependency resolution

Generate the three real resolver outputs from the exact manifests:

```bash
make dependency-locks
```

Review and commit:

- `uv.lock`
- `apps/desktop/package-lock.json`
- `apps/desktop/src-tauri/Cargo.lock`

Then require:

```bash
make reproducibility-audit
```

The strict audit must pass before a final release can claim reproducible dependency resolution.
Never hand-write a resolver lock.

## 2. Exact-head source verification

Run:

```bash
python scripts/verify.py
```

This includes the full DepthWizard-owned Python tests, Ruff and Pyright. The final-science campaign
and packaged-soak contracts are part of this verification surface.

Then run frontend and Rust verification from the committed dependency graph. Final release execution
must use the committed npm/Cargo locks (`npm ci`, Cargo `--locked`) rather than resolving a new graph.

## 3. Rebuild the scientific runtime and desktop from the same head

The packaged Python runtime must be rebuilt from the exact source head before Tauri bundles it.
Never qualify a new React/Tauri build against an older sidecar tree.

The existing RT5 build/acceptance machinery remains authoritative for:

- frozen geospatial self-check;
- owned PyInstaller ONEDIR runtime;
- identity-bound Tauri/sidecar boot;
- offline core policy;
- authenticated loopback boundary;
- malformed-input rejection and recovery;
- fresh project processing;
- validation;
- mesh/LOD construction;
- export bundle integrity;
- desktop-owned sidecar shutdown.

Historical RT5 passes remain evidence only for the commits that produced them. The final head must
earn its own acceptance.

## 4. Real operator workstation acceptance

Use a real metric project and record the complete workflow, not isolated screenshots.

Exercise:

1. Optical source view.
2. DSM view and metric/geodetic metadata.
3. Texture 3D.
4. DSM analytical overlay.
5. Slope overlay.
6. Hillshade.
7. Contours.
8. Confidence when a real native confidence product exists.
9. Reference and residual when validation exists.
10. Orbit.
11. Fly.
12. First Person with terrain clearance.
13. Top Down.
14. Deterministic Flythrough.
15. Auto and manual LOD.
16. Vertical exaggeration.
17. 3D probe.
18. 3D two-point Measure.
19. 3D elevation Profile.
20. 3D structural footprint measurement.
21. Reference comparison/validation.
22. Export.
23. Relaunch and reopen the persisted project.

A button being present is not a pass. A renderer claiming `ready` is not a pass unless the displayed
result is visibly usable. Analytical-overlay acceptance must include both aerial and low-camera
positions so the historical back-face/material defect cannot regress unnoticed.

Record sustained navigation performance over a representative camera path. The frozen acceptance
floor is **>=30 FPS sustained on the designated finale hardware**; a single rounded toolbar sample is
not sufficient evidence.

## 5. Two-hour packaged stability soak

After the exact-head app bundle passes finite acceptance, run:

```bash
python scripts/release_train_7_soak.py
```

The harness starts the real packaged Tauri app, proves the owned sidecar identity, repeatedly exercises
authenticated raster inspection, records desktop/sidecar RSS and request latency, and verifies clean
owned-process shutdown.

Only a full **7,200 seconds of monitored healthy time after successful boot** earns:

`PASS_TWO_HOUR_PACKAGED_SOAK`

A shorter run is explicitly `NON_QUALIFYING_SHORT_SOAK`. Failure creates a structured failure record
and must not be presented as stability evidence. This soak does not replace the separate GPU/FPS and
visual operator acceptance.

## 6. Official four-terrain / cross-sensor science campaign

Materialize and freeze a licensed registry with geographically disjoint:

- urban test scenes;
- sparse test scenes;
- hilly test scenes;
- forested test scenes;
- at least one cross-sensor holdout;
- independent reference DSM/LiDAR evidence.

Then run the evaluation-only protocol documented in `docs/final-science-campaign.md`:

```bash
python scripts/evaluate_final_science_campaign.py \
  /path/to/frozen-registry.yaml \
  /path/to/frozen-predictions.yaml \
  /path/to/final-science-evidence
```

Required primary artifacts include:

- `domain_generalization_report.json`
- `terrain_breakdown.csv`
- `scene_metrics.csv`
- `sensor_breakdown.csv`

The runner fails closed on incomplete four-terrain coverage, invalid cross-sensor lineage, missing
references, missing predictions, prediction/reference identity, and calibration/reference reuse.

## 7. Locked novelty evidence

Do not turn research components into marketing claims without their own evidence.

Final novelty closure requires defensible evidence for the claims actually used in the submission:

- **Geodetic Evidence Calibration** — production implementation and calibration evidence.
- **Remote-Sensing-Specific Geometry Fusion** — research architecture/ablation unless independently
  promoted; do not claim it beats DA3 when it does not.
- **Uncertainty-Aware Calibration & Analysis** — production confidence weighting plus final reliability
  analysis; model-native confidence is not a calibrated probability.
- **Cross-Sensor Generalization** — report measured holdout results.
- **Globally Consistent Large-Raster Tiling** — report measured seam-to-interior diagnostics; do not
  claim global superiority from overlap blending alone.
- **Evidence-Native 3D** — real reference/residual/measurement/analysis workflows plus operator
  acceptance.

Same-input published/reproducible baselines and required ablations must use the frozen evaluation
inputs. Negative results stay in the record. The V4 non-promotion decision remains valid unless a new,
independent protocol legitimately supersedes it.

## 8. Completion lock

DepthWizard is complete when all of the following are true:

- the official SIH26175 functional workflow passes on the exact packaged head;
- RMSE, MAE and correlation evidence exists across urban, sparse, hilly and forested landscapes;
- cross-sensor/generalization and the submission's novelty claims have defensible evidence;
- rendering/navigation/operator acceptance passes;
- sustained finale-hardware FPS passes;
- the two-hour packaged soak passes;
- dependency locks and clean-machine reproducibility pass;
- final source and technical documentation are frozen.

At that point, **stop product engineering**. Remaining work is submission production: the six-slide
SIH PDF, demo video, final screenshots, documentation packaging and judge Q&A.
