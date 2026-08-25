from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints" / "baselines" / "rdah" / "sweep"
LEGACY_SELECTION = ROOT / "checkpoints" / "baselines" / "rdah" / "checkpoint_selection.json"
MANIFEST_PATH = CHECKPOINT_DIR / "checkpoint_sweep.json"
FIGSHARE_ARTICLE = 31986864
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
MAX_BYTES = 750 * 1024 * 1024
MIN_BYTES = 10 * 1024 * 1024
USER_AGENT = "DepthWizard-SIH26175/0.2 reproducible-baseline-sweep"


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload: Any = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"unexpected JSON payload from {url}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, output: Path, expected_size: int) -> None:
    if output.exists() and output.stat().st_size == expected_size:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        copied = 0
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            if copied and copied % (50 * 1024 * 1024) < len(chunk):
                print(f"  {output.name}: {copied / (1024 * 1024):.0f} MiB")
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {output.name}: "
            f"{temporary.stat().st_size} != {expected_size}"
        )
    temporary.replace(output)


def legacy_checkpoint() -> tuple[int | None, Path | None]:
    if not LEGACY_SELECTION.exists():
        return None, None
    payload = json.loads(LEGACY_SELECTION.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None, None
    raw_id = payload.get("figshare_file_id")
    file_id = int(raw_id) if isinstance(raw_id, int | str) and str(raw_id).isdigit() else None
    raw_path = payload.get("checkpoint")
    path = Path(str(raw_path)) if raw_path else None
    return file_id, path if path is not None and path.exists() else None


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = request_json(FIGSHARE_API)
    files = metadata.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Figshare record has no files list")

    old_id, old_path = legacy_checkpoint()
    prepared: list[dict[str, object]] = []
    candidates: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        size = int(item.get("size", 0) or 0)
        suffix = Path(name).suffix.lower()
        url = item.get("download_url")
        file_id = item.get("id")
        if suffix not in {".pth", ".pt", ".ckpt"}:
            continue
        if not (MIN_BYTES <= size <= MAX_BYTES):
            continue
        if not isinstance(url, str) or not isinstance(file_id, int):
            continue
        candidates.append(item)

    if not candidates:
        raise RuntimeError("no compact published RDAH checkpoints found on Figshare")

    print(f"Preparing {len(candidates)} published RDAH checkpoint candidates...")
    for item in candidates:
        name = str(item["name"])
        size = int(item["size"])
        file_id = int(item["id"])
        url = str(item["download_url"])
        output = CHECKPOINT_DIR / f"figshare-{file_id}-{name}"

        if old_id == file_id and old_path is not None and old_path.stat().st_size == size:
            if not output.exists() or output.stat().st_size != size:
                shutil.copy2(old_path, output)
                print(f"  reused cached checkpoint for Figshare file {file_id}")
        else:
            download(url, output, size)

        prepared.append(
            {
                "figshare_file_id": file_id,
                "figshare_file_name": name,
                "size_bytes": size,
                "checkpoint": str(output.resolve()),
                "sha256": sha256(output),
                "training_domain": "unresolved_from_public_filename",
            }
        )

    manifest = {
        "figshare_article_id": FIGSHARE_ARTICLE,
        "figshare_title": metadata.get("title"),
        "figshare_doi": metadata.get("doi"),
        "checkpoint_count": len(prepared),
        "checkpoint_identity_note": (
            "The public Figshare record exposes duplicate generic checkpoint filenames. "
            "DepthWizard therefore preserves every compact published checkpoint by immutable "
            "Figshare file ID and does not invent a dataset label."
        ),
        "checkpoints": prepared,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("DepthWizard RDAH checkpoint sweep setup: PASS")
    for item in prepared:
        print(
            f"  file {item['figshare_file_id']} | {item['figshare_file_name']} | "
            f"sha256 {str(item['sha256'])[:12]}..."
        )
    print(f"Manifest: {MANIFEST_PATH}")
    print("Training-domain labels remain unresolved unless the publisher metadata states them.")


if __name__ == "__main__":
    main()
