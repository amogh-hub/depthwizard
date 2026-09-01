# ADR 0007: Terrain-structure decomposition for elite reconstruction

## Status

Accepted for research and corrective engineering on `engineering/terrain-structure-vnext`.

This ADR does **not** production-promote a new estimator. The production estimator policy remains
unchanged until every promotion gate in this document is satisfied on a frozen candidate.

## Context

The exposed ISPRS Potsdam urban operator scene demonstrated a failure mode that aggregate DSM metrics
did not capture. The V6 research candidate improved exposed aggregate/structure-band RMSE by only a
small amount and therefore passed its automated candidate gate, yet a visibly multi-storey building
returned a human-visible structure-height measurement of only **0.345 m**. The same operator scene
contained warped local urban relief, mound/pit artefacts, and poor roof-to-ground geometry.

Three distinct resolutions were involved:

1. native operator imagery / DSM: 0.05 m GSD, 6000 x 6000;
2. V6 refinement/evaluation grid: 0.25 m GSD, 1200 x 1200; and
3. finest interactive terrain mesh: approximately 0.60 m vertex spacing.

A small 0.25 m residual lifted bilinearly to 0.05 m cannot reconstruct missing native roof-edge and
building-height geometry by itself. In addition, the project structure-height tool historically used an
8-pixel surrounding ring. At 0.05 m GSD that ring represented only 0.40 m of physical support, allowing
eaves, wall transitions, vegetation, neighbouring objects, and DSM edge artefacts to contaminate the
local-ground estimate.

The urban defect therefore cannot be solved honestly by training the same monolithic residual longer,
increasing correction bounds without evidence, or smoothing the rendered terrain.

Natural terrain presents a different optimization problem. Mountains, hills, ridges, and valleys benefit
from topographic continuity and multi-scale context, while buildings require sharp discontinuities,
instance consistency, planar/structured roofs, and explicit height-above-ground reasoning. One uniform
prior is not sufficient for both at elite fidelity.

## Decision

DepthWizard vNext research adopts **terrain-structure decomposition** as the architectural principle:

`scientific DSM = terrain elevation + above-ground structure height`

The implementation may learn/fuse these terms jointly, but the conceptual and evaluation contracts stay
separate.

### 1. Terrain expert

The terrain path is responsible for low-frequency geodetic/topographic structure and natural relief. It
may consume:

- optical RGB evidence;
- one or more relative monocular geometry priors;
- trustworthy GSD/context metadata;
- coarse DEM/DTM evidence when available;
- optional sparse independent elevation anchors when explicitly available.

Terrain losses/evaluation must include elevation and topographic shape. Reference-slope strata are
mandatory so easy low-slope pixels cannot hide failure on steep terrain.

### 2. Structure expert

The urban/above-ground path is explicitly structure aware. Research candidates should represent and,
where supervision exists, learn:

- building/structure probability or instances;
- roof interior and boundaries;
- height bins / ordinal height;
- continuous height above local ground;
- normals/boundaries and roof consistency;
- uncertainty/reliability;
- optional roof-footprint/parallax geometry for off-nadir imagery;
- optional physically interpretable shadow evidence only when acquisition/solar geometry is trustworthy.

The network must not rely on whole-scene RMSE as a proxy for individual building height.

### 3. Evidence-aware fusion

Fusion must distinguish what each evidence source can legitimately constrain. A coarse DEM may constrain
datum and low-frequency terrain but must not be treated as evidence for an individual roof height. A
monocular prior remains relative until the geodetic calibration layer establishes physical scale.
Optional evidence that is missing or untrustworthy must receive zero authority rather than guessed
values.

### 4. Native / multi-resolution urban path

Urban research must retain native or near-native structural information. A coarse residual may be used as
context, but the final structure path cannot depend only on 0.25 m predictions bilinearly expanded to a
0.05 m scientific surface. Fine roof boundaries and local height discontinuities need a dedicated
high-resolution path plus wider contextual features.

### 5. Scientific raster versus derived 3D reconstruction

DepthWizard maintains two explicit products:

1. **Scientific DSM/rDSM** — immutable evidence-bearing raster used for metrics, probes, slope,
   residuals, exports, and quantitative measurements.
2. **Derived analytical 3D reconstruction** — a visualization/analysis representation that may use
   inferred building footprints, roof planes, vertical walls, breaklines, and adaptive LOD.

The derived 3D layer may never overwrite or cosmetically alter the scientific DSM. Structure-aware mesh
construction is intended to avoid representing a vertical building wall as a sloped height-field triangle,
not to hide scientific reconstruction error.

## Measurement contract

Project structure height is defined in physical units only when trustworthy metric GSD exists.
Project-level measurement uses:

- an analyst/structure footprint;
- an inward roof-edge exclusion in metres;
- a physically scaled inner exclusion buffer around the structure;
- a wider physical ground annulus;
- optional explicit semantic ground candidates;
- robust local-ground plane fitting with high-object rejection;
- support/dispersion warnings and fail-closed behaviour when evidence is insufficient.

A fixed pixel ring must never again define project-level physical ground support.

## Promotion contract

A future urban estimator is **not** eligible for blind evaluation or production promotion unless all of
the following are true on frozen exposed development evidence:

1. candidate and baseline use the same reference-selected building instances;
2. baseline and candidate have no measurement failures on reference-eligible buildings;
3. building-height MAE improves by at least 15% by default;
4. building-height RMSE improves by at least 10% by default;
5. building-height P90 absolute error improves by at least 5% by default;
6. fraction within 2 m does not regress;
7. catastrophic >3 m building-error rate does not increase;
8. roof/top MAE does not regress beyond the larger of 0.05 m or 2% of baseline top MAE;
9. local-ground MAE does not regress beyond the larger of 0.05 m or 2% of baseline ground MAE;
10. whole-scene DSM and slope safety gates remain satisfied;
11. natural terrain does not regress materially;
12. human-visible operator validation passes on exposed imagery.

The explicit top/ground non-regression gates prevent a candidate from appearing to improve
`roof - ground` while moving both absolute surfaces in the wrong direction. Building-height improvement
therefore cannot be obtained by compensating roof and ground errors.

These default effect-size thresholds are deliberately much stronger than the millimetre-scale improvement
that allowed V6 to become an automated candidate. Thresholds may be revised only **before** a benchmark
is consumed and with an explicit documented reason; they may not be weakened after seeing a candidate.

For terrain-specialist promotion, >=30 degree reference-slope pixels must be evaluated separately. The
default research gate requires material steep-terrain elevation and slope RMSE improvement while limiting
regression in every easier reference-slope stratum.

Passing either specialist gate is necessary but not sufficient for production promotion.

## Blind-evidence boundary

Potsdam `4_12` and `6_12` remain sealed after the V6 human-visible failure. They may not be used for
training, architecture selection, threshold tuning, loss selection, debugging, or qualitative model
selection. They are opened only after the final urban candidate and its promotion thresholds are frozen.

The exposed `2_14` scene may be used for corrective engineering and building-instance diagnostics, with
the official DSM used strictly downstream as evaluation/reference evidence.

## Rejected alternatives

### V7 = V6 plus more epochs

Rejected. The failure is architectural/evaluative, not evidence that V6 merely stopped training too early.

### Increase residual correction bounds without new supervision

Rejected. Larger unconstrained corrections increase the ability to damage terrain without teaching the
model which building should be corrected or by how much.

### Smooth the scientific DSM or mesh until buildings look clean

Rejected. Rendering cosmetics cannot be used to conceal incorrect Z geometry. Display-only anti-aliasing
and structure-aware meshing are allowed only as derived representations with provenance.

### Use the official reference DSM during reconstruction

Rejected. Reference data remains downstream evaluation evidence. Feeding it into the exposed prediction
would invalidate the scientific claim and would make blind evaluation meaningless.

### Use one global metric as the urban promotion criterion

Rejected. V6 demonstrated that scene-level or frequency-band RMSE can improve while actual building
height remains unusable.

## Consequences

- Existing production policy remains frozen until a candidate passes the stronger evidence chain.
- New model work should target terrain/structure experts and meaningful task-specific supervision rather
  than another generic residual version number.
- Final scientific-source promotion will invalidate affected exact-SHA qualification evidence and require
  rerunning the relevant science, operator, rendering, stability, and standalone gates.
