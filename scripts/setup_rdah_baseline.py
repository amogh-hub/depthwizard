from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / ".vendor"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "baselines"
ARTIFACT_ROOT = ROOT / "artifacts" / "baselines" / "rdah"

RDAH_REPO = "https://github.com/Elenairene/RDAH-Net.git"
RDAH_COMMIT = "373bca28299683ab0e5d892dfc87598ab967b564"
RDAH_DIR = VENDOR / "RDAH-Net"

DA2_REPO = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DA2_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
DA2_DIR = VENDOR / "depth-anything-v2"
DA2_WEIGHT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
    "depth_anything_v2_vits.pth?download=true"
)
DA2_WEIGHT = CHECKPOINT_ROOT / "da2" / "depth_anything_v2_vits.pth"

FIGSHARE_ARTICLE = 31986864
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"
RDAH_DOWNLOAD_DIR = CHECKPOINT_ROOT / "rdah"
RDAH_EXTRACT_DIR = RDAH_DOWNLOAD_DIR / "extracted"
SELECTION_PATH = RDAH_DOWNLOAD_DIR / "checkpoint_selection.json"
MAX_RDAH_DOWNLOAD_BYTES = 750 * 1024 * 1024
USER_AGENT = "DepthWizard-SIH26175/0.2 reproducible-baseline"


def run(*args: str, cwd: Path | None = None) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd or ROOT, check=True)


def ensure_repo(url: str, path: Path, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        run("git", "clone", "--filter=blob:none", url, str(path))
    run("git", "fetch", "origin", commit, cwd=path)
    run("git", "reset", "--hard", commit, cwd=path)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if actual != commit:
        raise RuntimeError(f"failed to pin {path.name}: expected {commit}, got {actual}")


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected JSON payload from {url}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, *, expected_size: int | None = None) -> Path:
    if path.exists() and path.stat().st_size > 0:
        if expected_size is None or path.stat().st_size == expected_size:
            return path
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        copied = 0
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if copied and copied % (100 * 1024 * 1024) < len(chunk):
                print(f"  downloaded {copied / (1024 * 1024):.0f} MiB -> {path.name}")
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {path.name}: "
            f"{temporary.stat().st_size} != {expected_size}"
        )
    temporary.replace(path)
    return path


def score_name(name: str, *, size: int) -> int:
    lower = name.lower()
    suffix = Path(lower).suffix
    score = 0
    if suffix in {".pth", ".pt", ".ckpt"}:
        score += 220
    elif suffix in {".zip", ".tar", ".tgz", ".gz"}:
        score += 40
    for term, value in {
        "checkpoint": 120,
        "ckpt": 120,
        "weight": 100,
        "model": 50,
        "track1": 90,
        "dfc": 90,
        "best": 50,
    }.items():
        if term in lower:
            score += value
    for term, value in {"dataset": -160, "training": -60, "swiss": -20, "hongkong": -20}.items():
        if term in lower:
            score += value
    if size > MAX_RDAH_DOWNLOAD_BYTES:
        score -= 500
    return score


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()

    def ensure_member(path: Path) -> None:
        resolved = path.resolve()
        if root != resolved and root not in resolved.parents:
            raise RuntimeError(f"unsafe archive member: {path}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                ensure_member(destination / member.filename)
            zf.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                ensure_member(destination / member.name)
            tf.extractall(destination, filter="data")
        return
    raise RuntimeError(f"unsupported checkpoint archive: {archive}")


def select_checkpoint(paths: list[Path]) -> Path:
    candidates = [path for path in paths if path.suffix.lower() in {".pth", ".pt", ".ckpt"}]
    if not candidates:
        raise RuntimeError("downloaded RDAH package contains no .pth/.pt/.ckpt checkpoint")
    ranked = sorted(
        candidates,
        key=lambda path: score_name(path.name, size=path.stat().st_size),
        reverse=True,
    )
    best = ranked[0]
    print("RDAH checkpoint candidates:")
    for path in ranked[:8]:
        print(f"  {path.name}: {path.stat().st_size / (1024 * 1024):.1f} MiB")
    return best


def setup_rdah_checkpoint() -> Path:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = request_json(FIGSHARE_API)
    files = metadata.get("files", [])
    if not isinstance(files, list) or not files:
        raise RuntimeError("Figshare record exposes no downloadable files")

    manifest: list[dict[str, object]] = []
    ranked: list[tuple[int, dict[str, Any]]] = []
    print("Figshare files relevant to RDAH-Net:")
    for raw in files:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "unnamed"))
        size = int(raw.get("size", 0) or 0)
        score = score_name(name, size=size)
        item = {
            "id": raw.get("id"),
            "name": name,
            "size": size,
            "download_url": raw.get("download_url"),
            "score": score,
        }
        manifest.append(item)
        ranked.append((score, raw))
        print(f"  {name} | {size / (1024 * 1024):.1f} MiB | candidate score {score}")

    (ARTIFACT_ROOT / "figshare_manifest.json").write_text(
        json.dumps(
            {
                "article_id": FIGSHARE_ARTICLE,
                "title": metadata.get("title"),
                "doi": metadata.get("doi"),
                "files": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ranked.sort(key=lambda item: item[0], reverse=True)
    score, chosen = ranked[0]
    chosen_name = str(chosen.get("name", "rdah-checkpoint"))
    chosen_size = int(chosen.get("size", 0) or 0)
    chosen_url = chosen.get("download_url")
    if score <= 0 or chosen_size > MAX_RDAH_DOWNLOAD_BYTES or not isinstance(chosen_url, str):
        raise RuntimeError(
            "could not safely auto-select a compact RDAH checkpoint package; "
            "inspect artifacts/baselines/rdah/figshare_manifest.json"
        )

    package = download(
        chosen_url,
        RDAH_DOWNLOAD_DIR / chosen_name,
        expected_size=chosen_size or None,
    )
    if package.suffix.lower() in {".pth", ".pt", ".ckpt"}:
        checkpoint = package
    else:
        if RDAH_EXTRACT_DIR.exists():
            shutil.rmtree(RDAH_EXTRACT_DIR)
        safe_extract(package, RDAH_EXTRACT_DIR)
        checkpoint = select_checkpoint([path for path in RDAH_EXTRACT_DIR.rglob("*") if path.is_file()])

    selection = {
        "figshare_article_id": FIGSHARE_ARTICLE,
        "figshare_file_name": chosen_name,
        "figshare_file_id": chosen.get("id"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "upstream_rdah_commit": RDAH_COMMIT,
        "intended_training_domain": "prefer_DFC2019_Track1_when_available",
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return checkpoint


def main() -> None:
    print("Pinning external baseline sources without installing their legacy requirements...")
    ensure_repo(RDAH_REPO, RDAH_DIR, RDAH_COMMIT)
    ensure_repo(DA2_REPO, DA2_DIR, DA2_COMMIT)

    print("Preparing the official Depth Anything v2 Small relative-depth checkpoint...")
    download(DA2_WEIGHT_URL, DA2_WEIGHT)
    if DA2_WEIGHT.stat().st_size < 20 * 1024 * 1024:
        raise RuntimeError("Depth Anything v2 checkpoint is unexpectedly small")

    print("Discovering and preparing the published RDAH-Net checkpoint package...")
    checkpoint = setup_rdah_checkpoint()

    print("DepthWizard RDAH baseline setup: PASS")
    print(f"RDAH source commit: {RDAH_COMMIT}")
    print(f"DA2 source commit: {DA2_COMMIT}")
    print(f"DA2 prior: {DA2_WEIGHT}")
    print(f"RDAH checkpoint: {checkpoint}")
    print(f"Selection manifest: {SELECTION_PATH}")
    print("Legacy RDAH requirements were NOT installed into the DepthWizard environment.")


if __name__ == "__main__":
    main()
