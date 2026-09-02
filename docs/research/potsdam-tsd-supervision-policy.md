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
into the exact active-component acquisition list without opening raster content. For the recorded
inventory, that plan contains 37 missing logical components: 12 RGB TIFFs, 12 DSM TIFFs, and 13
participant-label TIFFs.

## Official selective range-acquisition policy

The current official ISPRS/Leibniz Hannover interface exposes Potsdam as one remote `Potsdam.zip`
rather than as individually downloadable tile files. DepthWizard therefore uses the same
authenticated HTTP byte-range mechanism already proven by earlier qualification acquisition instead
of downloading the complete outer archive.

The verified remote hierarchy relevant to TSD is:

- outer `Potsdam.zip`: approximately 13.3 GB and HTTP `206 Partial Content` capable;
- `Potsdam/2_Ortho_RGB.zip`: STORE-compressed in the outer ZIP, so its central directory and selected
  member byte ranges can be addressed remotely without transferring the complete nested RGB ZIP;
- `Potsdam/1_DSM.rar`: STORE-compressed in the outer ZIP, RAR4 and non-solid, so RAR headers can be
  walked while packed payloads are skipped mathematically and only selected member byte ranges are
  fetched;
- `Potsdam/5_Labels_for_participants.zip`: DEFLATE-compressed in the outer ZIP and small enough that
  only this approximately 17 MB outer compressed member is transferred and reconstructed
  transiently before selecting the required participant labels.

The later `Potsdam/5_Labels_all.zip` is deliberately **not** used in the TSD supervision lane.
Although full labels are publicly available after the benchmark ended, preserving the historical
participant-label package makes accidental challenge-test supervision substantially harder.

Each newly acquired RGB and DSM TIFF must also be accompanied by its matching official `.tfw` world
file. The 37 logical missing components therefore expand to 61 physical files for transport:

- 12 RGB TIFFs + 12 RGB TFW sidecars;
- 12 DSM TIFFs + 12 DSM TFW sidecars;
- 13 participant-label TIFFs.

`qualification/download_tsd_potsdam_official_ranges.py` is the sole first-campaign acquisition path.
It binds directly to the frozen acquisition-plan JSON, requires the exact dataset root used to create
that plan, rejects sealed/nonactive tile requirements, and defaults to a preflight mode with no dataset
outputs written. `--execute` is required before selected files can be materialized.

The downloader fails closed if the official server does not return HTTP 206 ranges; it never falls
back to downloading the full archive. It validates the exact outer member names and compression
modes, reads remote ZIP/RAR metadata, verifies unique planned members, and verifies size/CRC for each
materialized output. RGB and DSM packed member downloads are resumable in a local cache, while final
outputs use partial/atomic staging. DSM extraction reconstructs a one-member RAR4 container and uses
`bsdtar`, matching the previously proven Mac acquisition method.

For labels, the complete small participant-label nested ZIP representation is reconstructed only in a
temporary directory because the outer DEFLATE layer prevents random access inside that nested ZIP.
The package is checked for historical challenge-test label entries, and only the 13 planned label
entries are decompressed to dataset outputs. Nonselected participant-label raster entries are not
decoded.

The acquisition report records source Git identity, acquisition-plan SHA-256, exact selected member
metadata, output SHA-256/CRC32 values, HTTP request count and transferred bytes. It explicitly records:

- `full_outer_archive_downloaded=false`;
- `full_rgb_inner_archive_downloaded=false`;
- `full_dsm_rar_downloaded=false`;
- `nonselected_participant_label_entries_decompressed=false`;
- `raster_pixels_decoded_by_downloader=false`;
- `sealed_blind_tile_payloads_extracted=false`.

No remote SHA-256 for the complete official `Potsdam.zip` is asserted by this workflow. Provenance is
the authenticated official ISPRS/Leibniz Hannover share plus exact archive/member metadata and local
output hashes; the claim is intentionally no stronger than the available evidence.

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
6. `download_tsd_potsdam_official_ranges.py` binds the frozen plan to the official single remote
   `Potsdam.zip`, selects only active-campaign TIFF/TFW/participant-label payloads, refuses full-archive
   fallback, and never extracts sealed blind payloads.
7. Duplicate candidate/archive members fail as ambiguous rather than being selected implicitly.
8. Inventory and acquisition planning remain filename-metadata-only. Acquisition may copy/hash only
   selected active raster bytes, but it does not decode or inspect raster pixels.

## Claim boundary

This policy prevents one class of benchmark and spatial leakage. It does not by itself prove model
generalization, DSM accuracy, or production readiness. A trained TSD checkpoint still has to pass the
predeclared exposed urban gate, geographically disjoint terrain gates, calibration/uncertainty gates,
operator-visible structure-height checks, and finally the one-time sealed blind evaluation before any
production promotion can be considered.
