from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "src"))

import numpy as np
import rasterio
import torch

from depthwizard.contracts import ProjectRunStatus
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.training import fit_rgb_ranges, normalize_rgb, patch_windows
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file
from scripts import train_urban_structure_v6 as v6

QUALIFIED_V6_SOURCE_SHA = "f1c2d5aac075430033b75502e5ef566c4ca1e8c9"
EXPECTED_V6_CHECKPOINT_SHA256 = "b0d2fbfc8929d05403d54ed500a239179577bb461679fa74f3c9d291e3641bb0"
EXPECTED_V6_BEST_EPOCH = 8
EXPECTED_CALIBRATION_DEM_SHA256 = "96e3c9de4049cff5d60f5e927c6d574f8cafcfde6164fa427a3ceab9fb11126d"
EXPECTED_V2_RDSM_SHA256 = "620b0430d0b22c7854733cc61bddd319f4769d9d273f87d29baa7a396764a35e"
EXPECTED_V2_DSM_SHA256 = "8bae324c5c6732d92dacd4af0bb321849a85eece0792f80526f369356ef59fe7"


def _label_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage exposed Potsdam 2_14 V6 operator project")
    parser.add_argument("--repo", type=Path, required=True)
    return parser


def _load_v6_model(repo: Path, device: torch.device) -> DepthWizardHeightModel:
    checkpoint_path = (
        repo
        / "artifacts"
        / "training"
        / "urban-structure-v6"
        / "height_model_urban_structure_v6.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing V6 checkpoint: {checkpoint_path}")
    actual = sha256_file(checkpoint_path)
    if actual != EXPECTED_V6_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"V6 checkpoint SHA mismatch: expected {EXPECTED_V6_CHECKPOINT_SHA256}, got {actual}"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("V6 checkpoint payload is not a dictionary")
    if checkpoint.get("best_epoch") != EXPECTED_V6_BEST_EPOCH:
        raise RuntimeError(
            f"unexpected V6 best epoch: expected {EXPECTED_V6_BEST_EPOCH}, "
            f"got {checkpoint.get('best_epoch')}"
        )
    config_payload = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise TypeError("V6 checkpoint is missing config/state_dict")
    model = DepthWizardHeightModel(HeightModelConfig(**config_payload))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _write_metric_dsm(path: Path, values: np.ndarray, transform: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=v6.POTSDAM_CRS,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(values.astype(np.float32), 1)
        dst.set_band_description(1, "DepthWizard V6 exposed-development metric DSM")
        dst.update_tags(
            DEPTHWIZARD_PRODUCT="METRIC_DSM_M",
            MODEL="V6_URBAN_STRUCTURE",
            SOURCE_SHA=QUALIFIED_V6_SOURCE_SHA,
            CHECKPOINT_SHA256=EXPECTED_V6_CHECKPOINT_SHA256,
            POTSDAM_TILE="2_14",
            DEVELOPMENT_ONLY="true",
        )


def main() -> int:
    args = _build_parser().parse_args()
    repo = args.repo.expanduser().resolve(strict=True)

    if set(v6.BLIND_TILE_IDS) != {"4_12", "6_12"}:
        raise RuntimeError("V6 blind-tile contract changed")
    if v6.URBAN_DEVELOPMENT_TILE_ID != "2_14":
        raise RuntimeError("V6 exposed development tile changed")

    dataset_root = repo / "data" / "external" / "isprs-potsdam"
    exposed_v2_root = (
        repo / "workspace" / "urban-mosaic-corrective" / "5e87670-potsdam-2_14"
    )
    output_project = repo / "workspace" / "operator-urban-potsdam-2-14-v6"

    device = v6._resolve_device()
    model = _load_v6_model(repo, device)
    tile = v6.resolve_potsdam_tile_paths(dataset_root, "2_14")

    rdsm_path = v6._find_tif_by_sha(
        exposed_v2_root,
        EXPECTED_V2_RDSM_SHA256,
        "V2 rDSM",
    )
    base_dsm_path = v6._find_tif_by_sha(
        exposed_v2_root,
        EXPECTED_V2_DSM_SHA256,
        "V2 metric DSM",
    )
    calibration_dem = (
        repo
        / "workspace"
        / "final-science-data"
        / "predictions"
        / "potsdam-2-14-urban-test"
        / "calibration"
        / "copernicus-glo30-mosaic.tif"
    )
    if not calibration_dem.is_file():
        raise FileNotFoundError(
            f"missing historical Copernicus calibration evidence: {calibration_dem}"
        )
    calibration_sha = sha256_file(calibration_dem)
    if calibration_sha != EXPECTED_CALIBRATION_DEM_SHA256:
        raise RuntimeError(
            f"calibration DEM SHA mismatch: expected {EXPECTED_CALIBRATION_DEM_SHA256}, "
            f"got {calibration_sha}"
        )

    height, width, transform = v6._target_grid(tile.rgb)
    rgb_raw, rgb_valid = v6._read_rgb_025m(tile.rgb, height, width)
    rdsm, rdsm_valid = v6._read_float_025m(rdsm_path, height, width)
    base_dsm, dsm_valid = v6._read_float_025m(base_dsm_path, height, width)
    input_valid = rgb_valid & rdsm_valid & np.isfinite(rdsm)
    ranges = fit_rgb_ranges(rgb_raw, input_valid)
    rgb = normalize_rgb(rgb_raw, ranges)

    scene = v6.SceneData(
        scene_id="POTSDAM_2_14_V6_OPERATOR",
        location_id="2_14",
        domain="potsdam",
        role="development",
        rgb=rgb,
        geometry=rdsm,
        reference_m=np.full_like(rdsm, np.nan, dtype=np.float32),
        input_valid=input_valid,
        supervision_valid=input_valid,
        gsd_m=v6.POTSDAM_BENCHMARK_GSD_M,
        rgb_ranges=ranges,
        target_prior=None,
        prior_reference_fit=None,
        windows=patch_windows(
            input_valid,
            patch_size=v6.PATCH_SIZE,
            stride=v6.PATCH_STRIDE,
            min_valid_fraction=0.90,
        ),
    )
    refined_relative, refined_valid = v6._predict_scene(model, scene, device)
    metric_scale, _metric_offset, affine_rmse = v6._sample_scale_fit(
        rdsm,
        base_dsm,
        input_valid & dsm_valid,
    )
    correction = refined_relative - rdsm
    refined_dsm = np.full_like(base_dsm, np.nan, dtype=np.float32)
    valid = refined_valid & dsm_valid & np.isfinite(correction) & np.isfinite(base_dsm)
    refined_dsm[valid] = base_dsm[valid] + correction[valid] * np.float32(metric_scale)

    if output_project.exists():
        shutil.rmtree(output_project)
    products = output_project / "products"
    products.mkdir(parents=True)
    dsm_path = products / "dsm.tif"
    _write_metric_dsm(dsm_path, refined_dsm, transform)

    source_sha = sha256_file(tile.rgb)
    prediction_sha = sha256_file(dsm_path)
    manifest = ProjectManifest.create_or_load(output_project, tile.rgb)
    manifest.set_identity(
        source_sha256=source_sha,
        input_kind="georeferenced",
        geometry_config_sha256=_label_sha(
            f"operator-stage:{QUALIFIED_V6_SOURCE_SHA}:V6:potsdam-2-14:geometry"
        ),
        run_config_sha256=_label_sha(
            f"operator-stage:{QUALIFIED_V6_SOURCE_SHA}:V6:potsdam-2-14:metric"
        ),
    )
    manifest.set_estimator(
        {
            "selected_model_id": "DA3MONO-LARGE + URBAN-STRUCTURE-V6",
            "qualified_v6_source_sha": QUALIFIED_V6_SOURCE_SHA,
            "v6_checkpoint_sha256": EXPECTED_V6_CHECKPOINT_SHA256,
            "v6_best_epoch": EXPECTED_V6_BEST_EPOCH,
            "exposed_development_only": True,
            "blind_tiles_touched": False,
        }
    )
    manifest.register_artifact(
        "dsm",
        dsm_path,
        semantics="metric_dsm_v2_broad_terrain_plus_v6_structure_correction",
        units="m",
        sha256=prediction_sha,
    )
    manifest.record_stage(
        ProcessingStage.INGEST,
        status="completed",
        details={"source_path": str(tile.rgb), "source_sha256": source_sha},
    )
    manifest.record_stage(
        ProcessingStage.GEOMETRY,
        status="completed",
        details={
            "model_id": "DA3MONO-LARGE + URBAN-STRUCTURE-V6",
            "v6_checkpoint_sha256": EXPECTED_V6_CHECKPOINT_SHA256,
            "v6_best_epoch": EXPECTED_V6_BEST_EPOCH,
        },
    )
    manifest.record_stage(
        ProcessingStage.CALIBRATION,
        status="completed",
        artifacts={"dsm": str(dsm_path)},
        details={
            "method": "inherited_frozen_v2_metric_affine_plus_v6_local_structure_correction",
            "recovered_v2_metric_scale_m_per_relative_unit": metric_scale,
            "v2_affine_reconstruction_rmse_m": affine_rmse,
            "evidence": {
                "dem": {
                    "path": str(calibration_dem),
                    "sha256": calibration_sha,
                    "role": "independent_coarse_calibration_evidence",
                }
            },
        },
    )
    manifest.record_stage(
        ProcessingStage.COMPLETE,
        status="completed",
        details={
            "operator_validation_required": True,
            "reference_not_consumed_by_staging_helper": True,
            "reference_path_for_manual_ui_selection": str(tile.reference_dsm),
            "reserved_blind_tiles": list(v6.BLIND_TILE_IDS),
            "blind_tiles_touched": False,
        },
    )
    manifest.mark_status(ProjectRunStatus.COMPLETE)
    ProjectManifest.load(output_project)

    correction_m = np.abs(correction[valid] * np.float32(metric_scale))
    print("PASS_OPERATOR_URBAN_V6_PROJECT_STAGED")
    print(f"project_dir={output_project}")
    print(f"source={tile.rgb}")
    print(f"v6_checkpoint_sha256={EXPECTED_V6_CHECKPOINT_SHA256}")
    print(f"v6_best_epoch={EXPECTED_V6_BEST_EPOCH}")
    print(f"prediction_sha256={prediction_sha}")
    print(f"metric_scale_m_per_relative_unit={metric_scale:.9f}")
    print(f"correction_mean_abs_m={float(np.mean(correction_m)):.6f}")
    print(f"correction_p95_abs_m={float(np.percentile(correction_m, 95)):.6f}")
    print(f"reference_for_manual_validation={tile.reference_dsm}")
    print("reference_consumed_by_staging_helper=false")
    print("blind_tiles_4_12_6_12_touched=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
