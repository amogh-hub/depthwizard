# Exposed Potsdam 2_14 diagnostic protocol

## Status

Frozen development diagnostic: `potsdam-2_14-building-diagnostic-v1`.

This protocol is restricted to the already-exposed ISPRS Potsdam `2_14` scene and to the exact staged
V6 metric DSM whose SHA-256 is:

```text
45e9ca10fee63a1a3078d2f0b2b978ab8d82d82c53cec9bb051fe6a3a3fb1062
```

It is downstream evaluation evidence only. It does **not** authorize opening, inspecting, deriving labels
for, or otherwise consuming sealed blind tiles `4_12` or `6_12`.

## Purpose

The V6 operator test showed that a visibly multi-storey building could measure only `0.345 m` above
local ground. Aggregate V6 metrics were therefore insufficient to decide whether the dominant failure
was roof elevation, local-ground elevation, or both. This protocol makes that distinction explicit
before any next-generation model architecture is selected.

The canonical diagnostic produces:

1. project-level reference validation and residual evidence;
2. deterministic official-ISPRS building and strict-ground masks;
3. reference-selected per-building height evaluation;
4. separate roof/top and local-ground error metrics;
5. one hash-audited top-level diagnosis artifact.

## Frozen scientific policy

- Tile: `2_14` only.
- Prediction: exact staged V6 native metric DSM, SHA-256
  `45e9ca10fee63a1a3078d2f0b2b978ab8d82d82c53cec9bb051fe6a3a3fb1062`.
- Reference data are evaluation-only and never enter inference or calibration.
- Building support comes from the official ISPRS building class.
- Canonical ground support is **strict impervious surface only**.
- Low vegetation may be evaluated later only as an explicitly labeled sensitivity analysis. It is not
  permitted to replace the canonical strict-ground result after results are inspected.
- Roof inset: `0.50 m`.
- Ground exclusion buffer: `1.50 m`.
- Ground outer support: `8.00 m`.
- Minimum building area: `20 m²`.
- Minimum reference structure height: `2.00 m`.
- A model cannot remove difficult buildings from evaluation: reference-selected eligible instances that
  fail on the candidate are recorded as prediction failures.
- Future promotion cannot be earned by compensating errors: roof/top MAE and local-ground MAE each have
  an explicit non-regression gate in addition to building-height improvement.
- Qualification evidence must be generated from a Git-attributable checkout with no tracked local
  modifications. The top-level diagnosis records the exact Git SHA and runner-file SHA-256.
- If the staged project DSM does not hash exactly to the frozen V6 prediction above, the canonical run
  stops before reference validation. A different candidate requires a separately attributable diagnosis
  rather than silently reusing the V6 evidence label.

## Workstation execution

From the repository:

```bash
cd ~/Documents/depthwizard
git fetch origin
git switch engineering/terrain-structure-vnext
git pull --ff-only origin engineering/terrain-structure-vnext
git status --short
```

`git status --short` should show no tracked modifications before qualification. Untracked local data are
not used to decide the source identity.

The already-staged V6 project is expected at:

```text
/Users/amoghrb/Documents/depthwizard/workspace/operator-urban-potsdam-2-14-v6
```

The exposed reference DSM is expected at:

```text
/Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif
```

Run one canonical diagnosis from the Potsdam dataset root:

```bash
python qualification/run_exposed_potsdam_2_14_diagnosis.py \
  --project-dir /Users/amoghrb/Documents/depthwizard/workspace/operator-urban-potsdam-2-14-v6 \
  --reference /Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif \
  --dataset-root /Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam \
  --output-dir /Users/amoghrb/Documents/depthwizard/qualification/evidence/potsdam-2_14-v6-diagnosis
```

Before any reference metric is produced, the runner verifies that the project's persisted metric DSM has
the frozen V6 SHA-256 above.

The runner searches only TIFF candidates whose filename identifies exposed tile `2_14` and looks like a
label/ground-truth raster. A candidate is accepted only if it is a three-band raster on the exact
reference grid, decodes completely under the official six-class ISPRS palette, and contains both
building and impervious pixels. Exactly one valid candidate must resolve. If none or more than one valid
candidate exists, the run stops before reference validation rather than guessing.

If the local dataset intentionally contains multiple valid official label variants, freeze the desired
variant **before looking at diagnostic results** and pass it explicitly:

```bash
python qualification/run_exposed_potsdam_2_14_diagnosis.py \
  --project-dir /Users/amoghrb/Documents/depthwizard/workspace/operator-urban-potsdam-2-14-v6 \
  --reference /Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif \
  --semantic-label '/ABSOLUTE/PATH/TO/FROZEN_OFFICIAL_POTSDAM_2_14_LABEL.tif' \
  --output-dir /Users/amoghrb/Documents/depthwizard/qualification/evidence/potsdam-2_14-v6-diagnosis
```

The explicit label path is independently checked against the same tile identity, palette, and exact-grid
contract. RGB orthophotos, DSMs, model-generated segmentation, unknown palettes, wrong grids, and blind
tiles are rejected.

## Expected evidence

The top-level output is:

```text
qualification/evidence/potsdam-2_14-v6-diagnosis/exposed-potsdam-2_14-diagnosis.json
```

Additional evidence includes:

```text
semantic-masks/potsdam-2_14-building-mask.tif
semantic-masks/potsdam-2_14-ground-mask-strict.tif
semantic-masks/potsdam-2_14-semantic-mask-manifest.json
building-height/building-height-report.json
building-height/building-height-instances.csv
```

The top-level artifact records:

- exact qualification Git SHA;
- qualification runner SHA-256;
- exact frozen/actual V6 prediction SHA-256;
- semantic-label resolution mode and label SHA-256;
- project manifest SHA-256;
- reference SHA-256;
- reference-validation metrics;
- building-height, roof/top, and local-ground metrics.

Project reference validation also persists aligned reference, residual, metrics, and the human-readable
validation report through the existing downstream validation subsystem.

## Interpretation order

Read results in this order:

1. global elevation RMSE/MAE and residual pattern;
2. building-height MAE/RMSE/P90 and catastrophic `>3 m` rate;
3. building top-elevation MAE;
4. building local-ground MAE;
5. per-building CSV outliers and failure IDs.

Interpretation examples:

- **Large top MAE, small ground MAE:** roof/above-ground structure reconstruction is the dominant defect.
- **Small top MAE, large ground MAE:** local terrain/ground reconstruction is the dominant defect.
- **Both large:** the next architecture must improve both terrain and structure evidence rather than
  merely increasing residual amplitude.
- **Low aggregate error but high building-height error:** do not optimize another scene-average loss;
  building-instance supervision and promotion remain mandatory.

## What happens after this diagnostic

The next Terrain/Structure Expert architecture is selected only after these exposed-scene diagnostics are
recorded. No blind tile is opened during architecture selection. Any candidate must then satisfy the
material building-height gate, roof/top and local-ground non-regression gates, terrain
non-degradation/steep-terrain gate, whole-scene safety metrics, and human-visible operator validation
before blind evaluation is authorized.
