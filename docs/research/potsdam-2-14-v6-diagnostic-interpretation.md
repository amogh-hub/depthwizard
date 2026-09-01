# Exposed Potsdam 2_14 V6 diagnostic interpretation

## Evidence identity

This note records the completed frozen `potsdam-2_14-building-diagnostic-v1` protocol plus a read-only
signed-error summary derived from its persisted per-instance report. It is corrective-engineering evidence
only; it is not independent validation and does not authorize access to the sealed Potsdam `4_12` or
`6_12` tiles.

- qualification source: `0d503390fb90f629bd3498fd836c881559160c37`
- staged V6 native prediction SHA-256:
  `45e9ca10fee63a1a3078d2f0b2b978ab8d82d82c53cec9bb051fe6a3a3fb1062`
- official semantic label SHA-256:
  `1abed95e8514c68d76d9e9cd7c9896eacde1f066224c424915f22f445f78495d`
- semantic-label resolution: explicit official `top_potsdam_2_14_label.tif`
- ground policy: strict impervious-surface ground
- building pixels: 2,479,903
- strict-ground pixels: 5,538,723
- unlabeled pixels: 0

The semantic label was acquired from the official Potsdam archive by HTTP byte ranges. The full outer
archive was not downloaded; only the compressed `5_Labels_all.zip` member was reconstructed temporarily,
only `top_potsdam_2_14_label.tif` was retained, and `4_12` / `6_12` raster payloads were not extracted.

## Recorded V6 results

### Whole-scene geometry

- DSM RMSE: **4.431807 m**
- DSM MAE: **3.756197 m**
- slope RMSE: **27.353128 degrees**

### Reference-selected building benchmark

- evaluated buildings: **24**
- prediction failures: **0**
- prediction valid on reference-selected buildings: **1.000000**
- building-height MAE: **2.009846 m**
- building-height RMSE: **2.950318 m**
- building-height P90 absolute error: **4.749116 m**
- fraction within 2 m: **0.666667**
- roof/top MAE: **2.794966 m**
- local-ground MAE: **2.717511 m**

### Signed per-instance diagnosis

A read-only aggregation of the 24 persisted instance records produced:

- building-height bias: **-1.227208 m**
- roof/top bias: **+1.261420 m**
- local-ground bias: **+2.485358 m**
- median building-height error: **-0.694848 m**
- median roof/top error: **+1.289062 m**
- median local-ground error: **+2.977527 m**
- roof/ground signed-error correlation: **0.570653**
- same-sign roof/ground error fraction: **0.666667**
- mean common-mode `(roof error + ground error) / 2`: **+1.873389 m**
- mean reference building height: **8.478829 m**
- mean predicted building height: **7.251620 m**

Using a diagnostic magnitude threshold of 0.5 m only for directional counting, not for promotion:

- roof too high: **14 / 24**
- roof too low: **7 / 24**
- ground too high: **20 / 24**
- ground too low: **3 / 24**
- building height overestimated: **5 / 24**
- building height underestimated: **13 / 24**

The most severe building-height failures demonstrate differential compression rather than a single scalar
offset. Examples from the persisted report include:

- instance 32: reference 11.877 m, prediction 3.687 m, height error -8.191 m,
  roof error -6.272 m, ground error +1.924 m;
- instance 33: reference 13.104 m, prediction 6.289 m, height error -6.815 m,
  roof error -5.400 m, ground error +1.468 m;
- instance 22: reference 16.323 m, prediction 11.196 m, height error -5.127 m,
  roof error -2.496 m, ground error +2.712 m.

Other structures exhibit large common-mode displacement with much smaller AGL error. For example,
instance 4 has roof error +6.085 m, ground error +5.234 m, but building-height error only +0.830 m.
This is exactly the compensation pattern that height-only supervision or qualification can hide.

## Interpretation

The exposed operator failure is not explained by the old 8-pixel measurement ring alone. The corrected
physical measurement protocol still finds material urban reconstruction error across reference-selected
building instances.

The signed evidence resolves the earlier directional ambiguity. Local ground is systematically high on
this exposed scene, while roof error is less uniform. The average ground displacement exceeds average
roof displacement by approximately 1.22 m, consistent with the observed negative mean AGL bias. The
moderate positive roof/ground correlation and two-thirds same-sign fraction show a real common-mode
vertical component, but the catastrophic tall-building examples prove that common-mode correction alone
cannot solve the urban problem.

The failure therefore contains at least two coupled modes:

1. **terrain/common-mode displacement** — especially systematically high local ground; and
2. **differential structure compression** — roofs may remain too low while nearby ground is too high,
   producing severe AGL underestimation on individual structures.

The 27.35-degree slope RMSE independently rules out treating the scene as a pure scalar vertical bias. The
scientific surface also contains incorrect local relief and gradients.

## Architectural and loss consequence

The evidence rejects another monolithic residual-only V7 as the primary corrective path. The research
candidate exposes:

`relative DSM = relative terrain + relative above-ground height`.

The terrain branch is supervised on absolute ground/topographic surface, signed surface bias, and slope.
The structure branch is supervised on continuous AGL, true cumulative ordinal height thresholds,
structure support, roof surface, signed roof bias, boundaries, and roof geometry. Recomposition is
evaluated in addition to, not instead of, the two components.

The structure model additionally separates:

- **conditional AGL amplitude** — the height magnitude to learn when a structure is present; and
- **structure support probability** — whether that AGL should contribute to the recomposed DSM.

Continuous AGL supervision is applied to the ungated conditional amplitude. This prevents uncertain
support probability from mathematically attenuating the height target during early training. The
recomposed DSM still uses the support-gated AGL field, so support errors remain visible through
recomposition/roof losses.

Continuous AGL also receives an equal-authority metric-height-stratified loss over the predeclared height
bins. This is a generic long-tail safeguard: occupied tall strata cannot be numerically drowned by the
larger low/mid-rise pixel population. The strata are fixed by the model configuration, not tuned from
`2_14`.

The research implementation remains isolated from production in:

- `src/depthwizard/height_model/terrain_structure.py`
- `src/depthwizard/height_model/terrain_structure_loss.py`
- `src/depthwizard/height_model/terrain_structure_targets.py`
- `tests/test_terrain_structure_model.py`
- `tests/test_terrain_structure_targets.py`

Target preparation is auditable and training-only: strict semantic ground directly supervises terrain;
robust physical local-ground planes extend terrain beneath only supported building instances; unsupported
buildings are rejected and recorded rather than silently fabricated.

## Promotion status

**V6 remains rejected as the urban building-height fix.**

The terrain-structure module remains a research candidate only. No trained TSD checkpoint exists yet, no
production policy has changed, and no blind evidence has been opened. Exposed Potsdam `2_14` is corrective
engineering/evaluation evidence and must not become training supervision. Cross-sensor `3_14` remains an
evaluation scene. Sealed `4_12` and `6_12` remain untouched until the final candidate and thresholds are
frozen.
