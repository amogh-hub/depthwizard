from __future__ import annotations

from depthwizard.data.ortholoc import (
    location_id_from_filename,
    pair_scene_names,
    parse_apache_tif_listing,
    select_geographic_scenes,
)


def test_apache_listing_parser_only_returns_tiff_basenames() -> None:
    html = """
    <html><body>
      <a href="../">Parent</a>
      <a href="L01_R0000.tif">a</a>
      <a href="L02_R0003.tiff?download=1">b</a>
      <a href="notes.txt">notes</a>
      <a href="subdir/">subdir</a>
      <a href="https://example.test/L03_R0001.tif">c</a>
    </body></html>
    """
    assert parse_apache_tif_listing(html) == [
        "L01_R0000.tif",
        "L02_R0003.tiff",
        "L03_R0001.tif",
    ]


def test_geographic_selection_is_deterministic_and_excludes_groups() -> None:
    names = [
        "L01_R0000.tif",
        "L01_R0001.tif",
        "L02_R0000.tif",
        "L03_R0000.tif",
        "L04_R0000.tif",
    ]
    selected = select_geographic_scenes(
        names,
        max_locations=2,
        samples_per_location=1,
        excluded_locations={"L01"},
    )
    assert selected == ["L02_R0000.tif", "L03_R0000.tif"]
    assert {location_id_from_filename(name) for name in selected} == {"L02", "L03"}


def test_dop_dsm_pairing_requires_exact_filename_match() -> None:
    assert pair_scene_names(
        ["L01_R0000.tif", "L02_R0000.tif"],
        ["L01_R0000.tif", "L03_R0000.tif"],
    ) == ["L01_R0000.tif"]
