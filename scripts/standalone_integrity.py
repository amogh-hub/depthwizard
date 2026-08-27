from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Collection
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(
    root: Path,
    *,
    excluded_relative_paths: Collection[str] = (),
) -> tuple[str, int, int, int]:
    """Return deterministic tree SHA, regular-file count, symlink count, and logical bytes.

    Identity includes each relative path, Unix-style permission mode, symlink target, and regular-file
    bytes. Directory metadata is intentionally excluded. Optional exclusions are exact POSIX relative
    paths and are used only to avoid the runtime manifest hashing itself.
    """
    excluded = frozenset(excluded_relative_paths)
    digest = hashlib.sha256()
    file_count = 0
    symlink_count = 0
    logical_bytes = 0

    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative_text = path.relative_to(root).as_posix()
        if relative_text in excluded:
            continue
        relative = relative_text.encode("utf-8")
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8")
            digest.update(b"L\0" + relative + b"\0" + str(mode).encode() + b"\0" + target + b"\n")
            symlink_count += 1
            continue
        if path.is_file():
            digest.update(b"F\0" + relative + b"\0" + str(mode).encode() + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    logical_bytes += len(chunk)
            digest.update(b"\n")
            file_count += 1

    return digest.hexdigest(), file_count, symlink_count, logical_bytes
