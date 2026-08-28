from __future__ import annotations

import csv
import hashlib
import io
from math import isfinite
from pathlib import Path

from depthwizard.contracts import GroundControlPoint, GroundControlPointFileReport

_MAX_GCP_FILE_BYTES = 16 * 1024 * 1024
_MAX_GCP_POINTS = 10_000


def _read_bounded_snapshot(source: Path) -> bytes:
    """Read at most the allowed GCP payload plus one sentinel byte.

    Parsing and SHA identity must describe the same bytes. Reading one bounded snapshot avoids the
    parse-then-hash TOCTOU ambiguity of hashing the path after CSV parsing has already completed.
    The production runtime later hashes the current file again before calibration, so any mutation
    after this snapshot is also rejected.
    """
    with source.open("rb") as handle:
        payload = handle.read(_MAX_GCP_FILE_BYTES + 1)
    if len(payload) > _MAX_GCP_FILE_BYTES:
        raise ValueError(
            f"GCP CSV exceeds the {_MAX_GCP_FILE_BYTES // (1024 * 1024)} MiB ingestion limit"
        )
    return payload


def inspect_ground_control_point_file(path: str | Path) -> GroundControlPointFileReport:
    """Parse a bounded deterministic CSV GCP interchange file for desktop calibration workflows."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("GCP file does not exist")
    if source.suffix.lower() != ".csv":
        raise ValueError("GCP import currently requires a CSV file")

    raw = _read_bounded_snapshot(source)
    snapshot_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("GCP CSV must be UTF-8 text") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise ValueError("GCP CSV has no header row")
    fields = {name.strip().lower(): name for name in reader.fieldnames if name is not None}
    if "x" not in fields or "y" not in fields:
        raise ValueError("GCP CSV must contain x and y columns")
    elevation_key = next(
        (candidate for candidate in ("elevation_m", "elevation", "z") if candidate in fields),
        None,
    )
    if elevation_key is None:
        raise ValueError("GCP CSV must contain elevation_m, elevation, or z")
    weight_key = fields.get("weight")

    points: list[GroundControlPoint] = []
    for line_number, row in enumerate(reader, start=2):
        if len(points) >= _MAX_GCP_POINTS:
            raise ValueError(f"GCP CSV exceeds the {_MAX_GCP_POINTS:,}-point ingestion limit")
        try:
            x = float(row[fields["x"]])
            y = float(row[fields["y"]])
            elevation_m = float(row[fields[elevation_key]])
            weight = (
                float(row[weight_key])
                if weight_key and row.get(weight_key) not in (None, "")
                else 1.0
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric GCP value on CSV line {line_number}") from exc
        if not all(isfinite(value) for value in (x, y, elevation_m, weight)):
            raise ValueError(f"non-finite GCP value on CSV line {line_number}")
        if weight <= 0.0:
            raise ValueError(f"GCP weight must be positive on CSV line {line_number}")
        points.append(
            GroundControlPoint(
                x=x,
                y=y,
                elevation_m=elevation_m,
                weight=weight,
            )
        )

    if len(points) < 2:
        raise ValueError("GCP calibration requires at least two control points")
    elevations = [point.elevation_m for point in points]
    return GroundControlPointFileReport(
        source_path=source,
        sha256=snapshot_sha256,
        point_count=len(points),
        minimum_elevation_m=min(elevations),
        maximum_elevation_m=max(elevations),
        points=points,
        semantics=(
            "Parsed sparse metric ground-control evidence. Coordinates are interpreted in the source "
            "image CRS by the calibration runtime; this parser does not transform or infer coordinates."
        ),
    )
