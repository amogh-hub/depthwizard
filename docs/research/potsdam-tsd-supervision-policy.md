# Potsdam TSD supervision policy

Status: research protocol, not production promotion evidence.

## Purpose

Terrain–Structure Decomposition (TSD) must not gain accuracy by consuming evaluation evidence as
supervision. This document defines the conservative Potsdam supervision population and the first
predeclared spatial campaign used by the TSD research lane.

## Primary dataset authority

ISPRS describes the Potsdam 2D Semantic Labeling dataset as 38 co-registered 5 cm TOP/DSM patches.
The historical benchmark supplied labels for only part of the data and used the remaining scenes as
challenge-test evidence. ISPRS now also states on the benchmark landing page that, after ending the
challenge in 2018, the full reference data was released for download. DepthWizard deliberately does
**not** treat that later release as permission to widen the new TSD supervision population.

Primary sources:

- https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
- https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx
- https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx

The official semantic classes include impervious surfaces and buildings, which are the two classes
used directly by the current TSD target-construction policy for strict local-ground candidates and
structure support.

## Historical tile enumeration

The current ISPRS per-tile availability table is rendered as an image and is not suitable as a
machine-readable protocol dependency. The concrete historical train/test tile enumeration is
therefore cross-checked against the long-standing TorchGeo `Potsdam2D` challenge split
implementation:

- https://docs.torchgeo.org/en/v0.6.1/_modules/torchgeo/datasets/potsdam.html

TorchGeo is **not** treated as the primary authority for the scientific meaning of the labels; ISPRS
is. TorchGeo is used only to make the historical participant/evaluation tile membership explicit and
auditable in source code.

## Participant ground-truth population

The historical participant-ground-truth tile ids encoded in
`depthwizard.height_model.terrain_structure_split.PARTICIPANT_GROUND_TRUTH_TILE_IDS` are:

`2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12, 5_10, 5_11, 5_12, 6_10, 6_11, 6_12, 6_7, 6_8, 6_9, 7_10, 7_11, 7_12, 7_7, 7_8, 7_9`.

DepthWizard applies a stricter project-specific reservation:

- `2_14`: already-exposed corrective/evaluation evidence; never train/dev.
- `3_14`: external/cross-sensor evaluation evidence; never train/dev.
- `4_12`, `6_12`: sealed blind evidence; never train/dev and not opened before final freeze.

Because `4_12` and `6_12` are part of the historical participant-labelled population but are reserved
by DepthWizard as blind evidence, the legal TSD supervision pool contains 22 tiles.

## Historical challenge-test scenes

The historical challenge-test ids encoded in `HISTORICAL_CHALLENGE_TEST_TILE_IDS` are:

`2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13`.

These scenes are prohibited from TSD training, development loss, early stopping, target generation,
and hyperparameter selection even though complete reference packages are now obtainable. This
intentionally tightens the new TSD claim relative to V6 research, which had used `3_13` and `6_14`
during training.

## First predeclared urban campaign

Campaign protocol: `potsdam-tsd-urban-spatial-v1`.

The first campaign split was selected from **tile coordinates only**, before acquiring the missing
supervision and without opening new raster content. The purpose is to prevent both content-driven
split selection and obvious spatial leakage into the two sealed blind scenes.

Training block (8 tiles):

`6_7, 6_8, 6_9, 6_10, 7_7, 7_8, 7_9, 7_10`.

Development block (5 tiles):

`2_10, 2_11, 2_12, 3_10, 4_10`.

Blind spatial buffer (8 otherwise-legal supervision tiles):

`3_11, 3_12, 4_11, 5_11, 5_12, 6_11, 7_11, 7_12`.

Every legal participant-supervision tile that is an 8-neighbour of sealed `4_12` or `6_12` is in
that blind buffer. These tiles are intentionally withheld from the first campaign even though their
labels are legal participant ground truth.

Train/dev spatial buffer (1 tile):

`5_10`.

This prevents the northern development block and southern training block from being immediate tile
neighbours. Train + development + the two buffer groups exactly partition all 22 legal supervision
tiles; there is no unclassified remainder.

This is a conservative first campaign rather than a claim that 13 active tiles are universally
optimal. The unused legal tiles can support a later separately frozen campaign only after the first
campaign has been evaluated without modifying its evidence boundary.

## Current local inventory result

On source `e6d3206747b2800f1e82738e49572538d643de64`, the metadata-only v2 inventory reported only
`2_10` and `5_11` locally present, each with RGB + DSM but no semantic label. `5_11` is now a blind
spatial-buffer tile, so it is not required by the first campaign. `2_10` is a development tile and
requires only its missing label. The other 12 active campaign tiles require RGB, DSM, and label.

The deterministic planner `qualification/plan_tsd_potsdam_acquisition.py` converts the inventory JSON
into the exact active-component acquisition list without opening raster content.

## Acquisition package policy

The current ISPRS benchmark landing page points Potsdam downloads to the Leibniz Hannover Seafile
share. DepthWizard does not depend on an unofficial mirror for scientific acquisition provenance.
The first campaign uses the canonical Potsdam packages:

- RGB: `2_Ortho_RGB.zip`
- absolute reference DSM: `1_DSM.zip`
- historical participant semantic labels: `5_Labels_for_participants.zip`

The later `5_Labels_all.zip` package is deliberately **not** accepted by the TSD acquisition stager.
Although full labels are publicly downloadable after the benchmark ended, allowing that package into
the new training lane would weaken the supervision boundary and makes accidental challenge-test
consumption easier. The participant-label package is sufficient for all 13 active campaign tiles.

`qualification/stage_tsd_potsdam_acquisition.py` validates the three local archives before copying
anything. It requires canonical package names, validates ZIP member paths and rejects symlinks,
requires exactly one archive member for every planned missing file, rejects historical challenge-test
labels in the supplied label archive, enforces member/total size safety limits, refuses stale plans or
overwrites, and verifies enough free disk space. Only the 37 files listed by the current acquisition
plan (12 RGB + 12 DSM + 13 labels) can be copied. `4_12`, `6_12`, buffer tiles, and all challenge-test
payloads remain unextracted.

The stager defaults to a dry run. `--execute` is required to copy selected raster bytes. In both modes
it hashes the exact local source archives and records the acquisition-plan hash and source Git SHA.
During execution it hashes each staged file while copying it atomically. This command does **not**
decode raster pixels; archive central-directory metadata may enumerate package members, and selected
active raster bytes are read only for byte-for-byte staging and hashing.

The acquisition provenance intentionally does not claim a cryptographic remote-origin proof: the
current ISPRS workflow does not expose a remote SHA-256 for these packages to DepthWizard. The
operator must obtain the three canonical archives from the official ISPRS/Leibniz Hannover share;
DepthWizard then records their exact local SHA-256 identities for reproducibility.

## Fail-closed implementation

Split protocol version: `terrain-structure-training-split-v2`.

The following controls are executable rather than advisory:

1. `assert_tsd_supervision_tile_allowed()` rejects reserved, historical challenge-test, and arbitrary
   nonparticipant ids before filesystem resolution.
2. `initial_tsd_campaign_split()` materializes the predeclared 8-train / 5-dev partition and verifies
   that train + dev + buffers exactly cover the 22 legal supervision tiles.
3. `freeze_tsd_training_split.py --campaign potsdam-tsd-urban-spatial-v1` freezes that exact campaign,
   requires a tracked-clean source, validates only its 13 active tiles, and records exact
   RGB/DSM/semantic-label SHA-256 identities and geospatial metadata.
4. `inventory_tsd_potsdam_supervision.py` constructs diagnostics only for the fixed legal supervision
   population. Historical challenge-test files are not promoted to candidates even if they exist.
5. `plan_tsd_potsdam_acquisition.py` reports only components missing from the 13 active campaign
   tiles; buffer tiles are explicitly not required.
6. `stage_tsd_potsdam_acquisition.py` accepts only the three canonical source-package roles, selects
   exactly the plan-listed missing members, refuses challenge-test labels in the label archive, and
   never copies nonactive raster payloads.
7. Duplicate candidate/archive members fail as ambiguous rather than being selected implicitly.
8. Inventory and acquisition planning remain filename-metadata-only. Acquisition staging may copy and
   hash only selected active raster bytes, but it does not decode or inspect raster pixels.

## Claim boundary

This policy prevents one class of benchmark and spatial leakage. It does not by itself prove model
generalization, DSM accuracy, or production readiness. A trained TSD checkpoint still has to pass the
predeclared exposed urban gate, geographically disjoint terrain gates, calibration/uncertainty gates,
operator-visible structure-height checks, and finally the one-time sealed blind evaluation before any
production promotion can be considered.
