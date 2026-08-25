from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    *,
    source_path: str | Path,
    config: dict[str, Any],
    model_manifest: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    source = Path(source_path)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "config_sha256": canonical_json_hash(config),
        "model": model_manifest,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "warnings": warnings or [],
    }
