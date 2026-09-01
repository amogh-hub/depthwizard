# Exposed Potsdam 2_14 V6 diagnostic interpretation

## Evidence identity

This note records the first completed run of the frozen
`potsdam-2_14-building-diagnostic-v1` protocol. It is corrective-engineering evidence only; it is not
independent validation and does not authorize access to the sealed Potsdam `4_12` or `6_12` tiles.

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

## Interpretation

The exposed operator failure is not explained by the old 8-pixel measurement ring alone. The corrected
physical measurement protocol still finds material urban reconstruction error across reference-selected
building instances.

Both absolute surfaces are wrong at roughly the same scale: roof/top MAE is 2.795 m and local-ground MAE
is 2.718 m. Building-height MAE is lower at 2.010 m because roof-to-ground subtraction can cancel part of
the two absolute-surface errors. Therefore a height-only objective or promotion gate is insufficient.
Separate roof and ground supervision/non-regression remains mandatory.

The MAE values do **not** reveal the signed direction of the errors. This evidence alone cannot claim that
roofs are systematically too low, ground is systematically too high, or vice versa. Signed bias and
per-instance covariance must be read from the persisted instance report before making such a directional
claim.

The 27.35-degree slope RMSE is also too large to treat the urban problem as a pure scalar-height bias. It
supports the prior visual observation that the scientific surface contains local geometric roughness and
incorrect relief.

## Architectural consequence

The evidence rejects another monolithic residual-only V7 as the primary corrective path. The research
candidate must expose at least two inspectable quantities:

`relative DSM = relative terrain + relative above-ground height`.

The terrain branch must be supervised on absolute ground/topographic surface and slope. The structure
branch must be supervised on continuous AGL, ordinal height, structure support, roof surface, boundaries,
and roof geometry. Recomposition is evaluated in addition to, not instead of, the two components.

The first research implementation is intentionally isolated from production in:

- `src/depthwizard/height_model/terrain_structure.py`
- `src/depthwizard/height_model/terrain_structure_loss.py`
- `tests/test_terrain_structure_model.py`

The model begins from the geometry-prior identity but has an independent nonnegative above-ground path,
so structural amplitude is no longer constrained by the same small residual that limited V6. The terrain
head is lower-resolution; the structure head decodes to the native input grid. Separate roof and ground
losses make equal-shift compensation explicitly costly.

## Promotion status

**V6 remains rejected as the urban building-height fix.**

The terrain-structure module is a research candidate only. No checkpoint exists yet, no production policy
has changed, and no blind evidence has been opened. Before training/promotion, the remaining target
preparation must define an auditable bare-earth terrain target beneath buildings from training-only
semantic/reference evidence and must record unsupported structures rather than silently fabricating them.
