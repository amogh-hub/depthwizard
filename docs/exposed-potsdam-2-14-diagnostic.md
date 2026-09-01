# Exposed Potsdam 2_14 diagnostic protocol

## Status

Frozen development diagnostic: `potsdam-2_14-building-diagnostic-v1`.

This protocol is restricted to the already-exposed ISPRS Potsdam `2_14` scene. It is downstream
evaluation evidence only. It does **not** authorize opening, inspecting, deriving labels for, or
otherwise consuming sealed blind tiles `4_12` or `6_12`.

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

## Workstation execution

From the repository:

```bash
cd ~/Documents/depthwizard
git fetch origin
git switch engineering/terrain-structure-vnext
git pull --ff-only origin engineering/terrain-structure-vnext
```

The already-staged V6 project is expected at:

```text
/Users/amoghrb/Documents/depthwizard/workspace/operator-urban-potsdam-2-14-v6
```

The exposed reference DSM is expected at:

```text
/Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif
```

Do **not** guess the local official semantic-label filename. First enumerate only candidate files that
identify the exposed tile:

```bash
find /Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam \
  -type f \
  \( -iname '*2_14*.tif' -o -iname '*02_14*.tif' -o -iname '*2_14*.tiff' -o -iname '*02_14*.tiff' \) \
  -print | sort
```

Choose the **official ISPRS semantic ground-truth label raster**, not an RGB orthophoto, DSM, participant
prediction, or model-generated segmentation. The preparation code independently validates the official
six-class RGB palette and rejects unknown label colors.

Then run exactly one canonical diagnosis, substituting only the verified semantic-label path:

```bash
python qualification/run_exposed_potsdam_2_14_diagnosis.py \
  --project-dir /Users/amoghrb/Documents/depthwizard/workspace/operator-urban-potsdam-2-14-v6 \
  --reference /Users/amoghrb/Documents/depthwizard/data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif \
  --semantic-label '/ABSOLUTE/PATH/TO/OFFICIAL_POTSDAM_2_14_LABEL.tif' \
  --output-dir /Users/amoghrb/Documents/depthwizard/qualification/evidence/potsdam-2_14-v6-diagnosis
```

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
