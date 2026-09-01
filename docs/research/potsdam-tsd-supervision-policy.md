# Potsdam TSD supervision policy

Status: research protocol, not production promotion evidence.

## Purpose

Terrain–Structure Decomposition (TSD) must not gain accuracy by consuming evaluation evidence as
supervision. This document defines the conservative Potsdam supervision population used by the TSD
research lane.

## Primary dataset authority

ISPRS describes the Potsdam 2D Semantic Labeling dataset as 38 co-registered 5 cm TOP/DSM patches.
Its benchmark documentation states that labelled ground truth is supplied for only part of the data,
that participants should use the ground-truth portion for training/internal evaluation, and that the
remaining scenes are benchmark evaluation evidence.

Primary source:

- https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
- https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx

The official semantic classes include impervious surfaces and buildings, which are the two classes
used directly by the current TSD target-construction policy for strict local-ground candidates and
structure support.

## Historical tile enumeration

The current ISPRS web page renders its per-tile ground-truth availability table as an image, which is
not suitable as a machine-readable protocol dependency. The concrete historical train/test tile
enumeration is therefore cross-checked against the long-standing TorchGeo `Potsdam2D` challenge
split implementation:

- https://docs.torchgeo.org/en/v0.6.1/_modules/torchgeo/datasets/potsdam.html

TorchGeo is **not** treated as the primary authority for the scientific meaning of the labels; ISPRS
is. TorchGeo is used only to make the historical participant/evaluation tile membership explicit and
auditable in source code.

## Participant ground-truth population

The historical participant-ground-truth tile ids encoded in
`depthwizard.height_model.terrain_structure_split.PARTICIPANT_GROUND_TRUTH_TILE_IDS` are:

`2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12, 5_10, 5_11, 5_12, 6_10, 6_11, 6_12, 6_7, 6_8, 6_9, 7_10, 7_11, 7_12, 7_7, 7_8, 7_9`.

DepthWizard then applies a stricter project-specific reservation:

- `2_14`: already-exposed corrective/evaluation evidence; never train/dev.
- `3_14`: external/cross-sensor evaluation evidence; never train/dev.
- `4_12`, `6_12`: sealed blind evidence; never train/dev and not opened before final freeze.

Because `4_12` and `6_12` are part of the historical participant-labelled population but are reserved
by DepthWizard as blind evidence, the legal TSD supervision pool contains 22 tiles.

## Historical challenge-test scenes

The historical challenge-test ids encoded in
`HISTORICAL_CHALLENGE_TEST_TILE_IDS` are:

`2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13`.

These scenes are prohibited from TSD training, development loss, early stopping, target generation,
and hyperparameter selection even if a later or third-party package places label rasters on disk.
This intentionally tightens the new TSD claim relative to V6 research, which had used `3_13` and
`6_14` during training.

## Fail-closed implementation

Protocol version: `terrain-structure-training-split-v2`.

The following controls are executable rather than advisory:

1. `assert_tsd_supervision_tile_allowed()` rejects reserved, historical challenge-test, and arbitrary
   nonparticipant ids before filesystem resolution.
2. `freeze_tsd_training_split.py` validates all requested ids before dataset traversal, then records
   exact RGB/DSM/semantic-label SHA-256 identities and geospatial metadata for legal tiles only.
3. `inventory_tsd_potsdam_supervision.py` constructs diagnostics only for the fixed legal supervision
   population. Historical challenge-test files are not promoted to candidates even if they exist.
4. Duplicate candidate files fail as ambiguous rather than being selected implicitly.
5. The inventory remains filename-metadata-only and does not open raster pixels.

## Claim boundary

This policy prevents one class of benchmark leakage. It does not by itself prove model
generalization, DSM accuracy, or production readiness. A trained TSD checkpoint still has to pass the
predeclared exposed urban gate, geographically disjoint terrain gates, calibration/uncertainty gates,
operator-visible structure-height checks, and finally the one-time sealed blind evaluation before any
production promotion can be considered.
