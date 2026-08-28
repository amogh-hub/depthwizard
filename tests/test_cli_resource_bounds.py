from typer.testing import CliRunner

from depthwizard.cli import app

runner = CliRunner()


def test_reconstruct_cli_rejects_tile_size_above_runtime_bound() -> None:
    result = runner.invoke(
        app,
        ["reconstruct-da3", "source.tif", "output", "--tile-size", "4097"],
    )

    assert result.exit_code == 2
    assert "4096" in result.output


def test_reconstruct_cli_rejects_overlap_above_runtime_bound() -> None:
    result = runner.invoke(
        app,
        ["reconstruct-da3", "source.tif", "output", "--overlap", "4096"],
    )

    assert result.exit_code == 2
    assert "4095" in result.output


def test_calibrate_dem_cli_rejects_unbounded_smoothing_kernel() -> None:
    result = runner.invoke(
        app,
        [
            "calibrate-dem",
            "relative.tif",
            "dem.tif",
            "dsm.tif",
            "--sigma-px",
            "4097",
        ],
    )

    assert result.exit_code == 2
    assert "4096" in result.output
