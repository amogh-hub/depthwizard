# DepthWizard SIH26175 final evidence kit

This branch is an **operations-only companion** to the already-qualified production source.

## Frozen production identity

Production branch:

`engineering/rt5-rt7-hardening-one-go`

Frozen qualified commit:

`339bdf485149f552db846543b9e09377b567c19c`

Stable alias:

`release/sih26175-frozen-339bdf4`

**Do not merge this qualification branch into production.** Run real qualification while the working checkout itself remains at the frozen production SHA. RT5 is already closed on that source identity.

GitHub Actions are not required for the remaining work.

## Bootstrap without checking this branch out

While the working checkout is still the frozen production branch:

```bash
cd /Users/amoghrb/Documents/depthwizard
git fetch origin qualification/evidence-kit-339bdf4
bash <(git show origin/qualification/evidence-kit-339bdf4:qualification/bootstrap_evidence_kit.sh)
```

The bootstrap materializes this operations kit under `/tmp` and leaves the repository clean.

## Exactly five remaining gates

1. `four_terrain_science`
2. `operator_workstation`
3. `sustained_rendering`
4. `two_hour_stability`
5. `clean_machine_standalone`

## 1. Four-terrain science — automated path

The evidence kit now contains a guarded end-to-end runner:

`qualification/run_four_terrain_science_from_frozen_head.sh`

It performs:

1. deterministic co-acquired NEON RGB/DSM selection and download for:
   - CPER → sparse
   - NIWO → hilly
   - HARV → forested
2. deterministic OrthoLoC training-lineage scene acquisition;
3. frozen registry construction;
4. fixed untouched ISPRS Potsdam scene policy:
   - `2_14` → urban `test`
   - `3_14` → `cross_sensor_test`
5. production `DA3MONO-LARGE` reconstruction through the frozen CLI;
6. independent Copernicus GLO-30 calibration evidence acquisition/mosaicking;
7. metric DSM calibration;
8. prediction/checkpoint/calibration-evidence SHA freeze;
9. only after the freeze, independent reference evaluation;
10. final four-terrain contract verification;
11. SIH26175 completion checker refresh.

The four Potsdam tiles previously consumed by DepthWizard external-v2 are hard-excluded:

`2_10`, `3_13`, `5_11`, `6_14`

### Required local inputs

NEON's download endpoint requires a personal API token. Keep it only in the shell:

```bash
export NEON_API_TOKEN='YOUR_TOKEN'
```

Never commit that credential.

The existing Potsdam root must contain official RGB + absolute DSM files for `2_14` and `3_14`:

`data/external/isprs-potsdam`

The runner fails closed if those untouched files are missing.

### Reference-separation rule

For every evaluated scene:

1. RGB is the single-view input;
2. Copernicus GLO-30 is calibration-only evidence;
3. the ISPRS/NEON DSM is evaluation-only reference.

The NEON downloader is intentionally reference-safe: it may download reference bytes but does not open or hash the DSM. `prepare_metric_predictions.py` never opens registry reference paths. `freeze_final_science_manifest.py` then freezes prediction/checkpoint/calibration identities. Only after that PASS does the final evaluator decode reference elevations.

Expected output:

`artifacts/final-science/domain_generalization_report.json`

plus terrain, scene and sensor CSV reports.

## 2. Operator workstation — real human-visible evidence

Use:

- `qualification/templates/operator-observation.yaml`
- `qualification/build_operator_evidence.py`

Every frozen operator check starts `observed: false`. The builder emits `PASS_SIH26175_OPERATOR_ACCEPTANCE` only when every required check is explicitly observed and has an evidence reference.

Required coverage includes literal JPG/PNG/GeoTIFF behavior, optical/DSM views, 3D texture and analytical overlays, all camera modes, LOD, vertical exaggeration, probe/measure/profile, genuine urban structure height, independent validation/reference/residual, projection accuracy, export and relaunch/reopen.

## 3. Sustained rendering — computed from real telemetry

Use:

- `qualification/templates/rendering-samples.example.csv`
- `qualification/build_rendering_evidence.py`

Required CSV columns:

`elapsed_seconds,fps,camera_position,rendering_mode,navigating`

The builder computes the statistics itself and refuses PASS unless:

- duration ≥ 60 seconds;
- sample count ≥ 55;
- mean FPS ≥ 30;
- p05 FPS ≥ 30;
- navigation was exercised;
- both `aerial` and `low` camera positions occurred;
- both `texture` and `analytical_overlay` modes occurred.

## 4. Two-hour stability — fully automated real packaged soak

Run the existing guarded helper while checked out at the frozen production SHA:

```bash
git fetch origin qualification/evidence-kit-339bdf4
bash <(git show origin/qualification/evidence-kit-339bdf4:qualification/run_two_hour_soak_from_frozen_head.sh)
```

It invokes the frozen `scripts/release_train_7_soak.py`. Only ≥7200 monitored healthy seconds earns `PASS_TWO_HOUR_PACKAGED_SOAK`.

## 5. Clean-machine standalone — real second-Mac evidence

Use on the clean Mac:

- `qualification/templates/clean-machine-observation.yaml`
- `qualification/build_clean_machine_evidence.py`

Every observation starts false. The builder requires all clean-machine behaviors to be explicitly true and computes a deterministic SHA-256 tree identity of the installed `DepthWizard.app` bundle before emitting `PASS_CLEAN_MACHINE_STANDALONE`.

## Final staging

After real operator, rendering and clean-machine evidence files exist, use:

`qualification/stage_manual_evidence_from_frozen_head.sh`

It first checks the frozen git identity and PASS literals, then copies the three evidence files to the authoritative artifact paths and runs `scripts/check_sih26175_completion.py` for full field-level validation.

## Source-change lock

Do not reopen icons, navbar, raw Cargo, Tauri packaging architecture, production estimator policy, or other core engineering unless one of the five remaining evidence gates exposes a concrete reproducible defect.
