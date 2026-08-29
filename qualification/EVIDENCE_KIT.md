# DepthWizard SIH26175 final evidence kit

This branch is an **operations-only companion** to the already-qualified production source.

## Non-negotiable source identity

Production branch: `engineering/rt5-rt7-hardening-one-go`

Frozen qualified commit:

`339bdf485149f552db846543b9e09377b567c19c`

Stable alias:

`release/sih26175-frozen-339bdf4`

**Do not merge this qualification branch into production.** Run every real acceptance harness while the working checkout itself is at the frozen production SHA. The files on this branch are templates/runbooks only.

## Closed already

RT5 full packaged engineering acceptance is closed on `339bdf...`. Do not rerun icon, Cargo, navbar, Tauri packaging architecture, or core engineering unless a remaining gate exposes a concrete reproducible defect.

## Five remaining gates

1. `four_terrain_science`
2. `operator_workstation`
3. `sustained_rendering`
4. `two_hour_stability`
5. `clean_machine_standalone`

## GitHub Actions

The final qualification does **not** depend on GitHub Actions. The remaining gates are inherently data/hardware/human evidence and must run on local/finale machines or a clean Mac.

## Four-terrain science

Use `final-science-registry.template.yaml` and `final-science-predictions-draft.template.yaml` as scaffolding only. Replace every `FILL_ME` path with real licensed data before freezing.

Minimum evidence roles:

- urban test: ISPRS Potsdam RGB + independent absolute DSM
- sparse test: NEON CPER RGB + LiDAR DSM
- hilly test: NEON NIWO RGB + LiDAR DSM
- forested test: NEON HARV RGB + LiDAR DSM
- >=1 genuine cross-sensor holdout

For every evaluated scene keep three sources independent:

1. RGB input
2. coarse DEM or sparse GCP calibration evidence
3. final independent DSM/LiDAR reference

Never derive calibration evidence by downsampling the final reference.

Mandatory ordering:

1. freeze source head
2. freeze registry and evaluation windows
3. generate production metric DSM predictions without opening final references in the evaluator
4. freeze prediction SHA-256 identities with `scripts/freeze_final_science_manifest.py`
5. only then run `scripts/evaluate_final_science_campaign.py`

Expected final files in `artifacts/final-science/`:

- `domain_generalization_report.json`
- `terrain_breakdown.csv`
- `scene_metrics.csv`
- `sensor_breakdown.csv`

## Two-hour stability

While checked out at the frozen production SHA, execute the helper from this branch without checking this branch out:

```bash
git fetch origin qualification/evidence-kit-339bdf4
bash <(git show origin/qualification/evidence-kit-339bdf4:qualification/run_two_hour_soak_from_frozen_head.sh)
```

That helper refuses to run unless HEAD is exactly `339bdf...` and the worktree is clean.

## Operator / FPS / clean-machine evidence

The JSON files under `qualification/templates/` are **pending templates only**. Never change their status to PASS unless the corresponding real evidence has been observed and recorded.

The authoritative checker is still `scripts/check_sih26175_completion.py` on the frozen production commit.
