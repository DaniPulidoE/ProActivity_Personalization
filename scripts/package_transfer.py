#!/usr/bin/env python3
"""Package the study data for transfer to another machine.

Compresses (LZMA2) then encrypts (AES-256, encrypted headers) the minimal set
needed to rebuild the derived datasets on the far side:

    data/raw_data.jsonl        the frames
    data/user_loa_labels.csv   the driver labels
    data/calibration_data/     per-driver baselines

labeled_data.jsonl and preprocessed_data.jsonl are deliberately excluded --
regenerate them with scripts/build_loa_dataset.py after unpacking.

Usage
-----
Pack, from the repo root:

    uv run --with py7zr python scripts/package_transfer.py --out study_data.7z

Unpack on the receiving machine, then rebuild the derived datasets:

    uv run --with py7zr python scripts/package_transfer.py \
        --verify study_data.7z --into ./restored

Write the archive OUTSIDE the repo, and extract to disk rather than to the
removable drive you carried it on -- deleting plaintext off flash does not
reliably erase it.

The passphrase is read from a prompt so it stays out of shell history. Set
PV_XFER_PASSPHRASE to bypass the prompt (convenient for scripting, but the
value is visible to other processes on the machine -- prefer the prompt).
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_ITEMS = [
    "data/raw_data.jsonl",
    "data/user_loa_labels.csv",
    "data/calibration_data",
]

MANIFEST_NAME = "MANIFEST.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(root: Path, item: Path):
    """Yield (absolute path, archive-relative path) for a file or directory."""
    if item.is_dir():
        for p in sorted(item.rglob("*")):
            if p.is_file():
                yield p, p.relative_to(root).as_posix()
    else:
        yield item, item.relative_to(root).as_posix()


def read_passphrase(confirm: bool) -> str:
    env = os.environ.get("PV_XFER_PASSPHRASE")
    if env:
        print("[warn] using PV_XFER_PASSPHRASE from the environment")
        return env
    pw = getpass.getpass("Passphrase: ")
    if not pw:
        sys.exit("empty passphrase; aborting")
    if confirm and getpass.getpass("Confirm:    ") != pw:
        sys.exit("passphrases did not match")
    return pw


def build_manifest(root: Path, items: list[str]) -> tuple[dict, list[tuple[Path, str]]]:
    entries, files, missing = [], [], []
    for rel in items:
        target = (root / rel).resolve()
        if not target.exists():
            missing.append(rel)
            continue
        for abs_path, arc in iter_files(root, target):
            print(f"  hashing {arc} ...", flush=True)
            entries.append(
                {
                    "path": arc,
                    "bytes": abs_path.stat().st_size,
                    "sha256": sha256_file(abs_path),
                }
            )
            files.append((abs_path, arc))
    if missing:
        sys.exit(f"not found under {root}: {', '.join(missing)}")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_repo": root.name,
        "note": (
            "Derived datasets excluded by design. Rebuild with "
            "scripts/build_loa_dataset.py after extracting."
        ),
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    return manifest, files


def cmd_pack(args: argparse.Namespace) -> None:
    import py7zr

    root = Path(args.repo).resolve()
    if not root.is_dir():
        sys.exit(f"repo path is not a directory: {root}")

    out = Path(args.out).resolve()
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists (use --force to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"source: {root}")
    manifest, files = build_manifest(root, args.items)
    raw_mb = manifest["total_bytes"] / 1e6
    print(f"\n{len(files)} file(s), {raw_mb:.1f} MB raw")

    pw = read_passphrase(confirm=True)

    filters = [
        {"id": py7zr.FILTER_LZMA2, "preset": args.level},
        {"id": py7zr.FILTER_CRYPTO_AES256_SHA256},
    ]

    print(f"\ncompressing at preset {args.level} (this takes a few minutes) ...")
    started = time.time()
    with py7zr.SevenZipFile(out, "w", password=pw, filters=filters) as archive:
        archive.set_encrypted_header(True)
        archive.writestr(json.dumps(manifest, indent=2), MANIFEST_NAME)
        for abs_path, arc in files:
            print(f"  + {arc}", flush=True)
            archive.write(abs_path, arc)

    size = out.stat().st_size
    elapsed = time.time() - started
    print(f"\narchive : {out}")
    print(f"size    : {size / 1e6:.1f} MB  ({raw_mb / (size / 1e6):.0f}x smaller)")
    print(f"elapsed : {elapsed:.0f}s")
    print(f"sha256  : {sha256_file(out)}")
    print(
        "\nCheck that sha256 on the far side before extracting. Header encryption "
        "is on, so the archive does not reveal its filenames."
    )


@contextlib.contextmanager
def open_archive(path: Path, pw: str):
    """Open an encrypted archive, turning py7zr's internals into a clean message.

    The header itself is encrypted, so a wrong passphrase surfaces as a parse
    error deep inside archiveinfo rather than as an auth failure.
    """
    import py7zr

    try:
        archive = py7zr.SevenZipFile(path, "r", password=pw)
    except Exception as exc:  # noqa: BLE001 - py7zr raises many types here
        sys.exit(f"could not open archive (wrong passphrase, or corrupt file): {exc}")
    try:
        yield archive
    finally:
        archive.close()


def cmd_verify(args: argparse.Namespace) -> None:
    archive_path = Path(args.verify).resolve()
    if not archive_path.is_file():
        sys.exit(f"no such archive: {archive_path}")

    print(f"archive: {archive_path}")
    print(f"size   : {archive_path.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 : {sha256_file(archive_path)}")

    pw = read_passphrase(confirm=False)
    dest = Path(args.into).resolve() if args.into else None

    # py7zr >= 1.1 has no in-memory read(); pull the manifest out to a temp dir.
    with tempfile.TemporaryDirectory() as tmp:
        with open_archive(archive_path, pw) as archive:
            if MANIFEST_NAME not in archive.getnames():
                sys.exit("archive has no MANIFEST.json -- not produced by this script")
            archive.extract(path=tmp, targets=[MANIFEST_NAME])
        manifest = json.loads((Path(tmp) / MANIFEST_NAME).read_text(encoding="utf-8"))

    print(f"\ncreated {manifest['created_utc']} from repo '{manifest['source_repo']}'")
    print(f"{len(manifest['files'])} file(s), {manifest['total_bytes'] / 1e6:.1f} MB\n")
    for entry in manifest["files"]:
        print(f"  {entry['bytes'] / 1e6:9.2f} MB  {entry['path']}")

    if dest is None:
        print("\nManifest read OK (passphrase correct). Pass --into DIR to extract.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nextracting into {dest} ...")
    with open_archive(archive_path, pw) as archive:
        archive.extractall(path=dest)

    print("\nchecking hashes ...")
    bad = 0
    for entry in manifest["files"]:
        target = dest / entry["path"]
        if not target.is_file():
            print(f"  MISSING  {entry['path']}")
            bad += 1
            continue
        if sha256_file(target) != entry["sha256"]:
            print(f"  MISMATCH {entry['path']}")
            bad += 1
        else:
            print(f"  ok       {entry['path']}")

    if bad:
        sys.exit(f"\n{bad} file(s) failed verification")
    print("\nAll files match the manifest.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="repo root the paths are relative to")
    ap.add_argument("--out", default="study_data.7z", help="output archive path")
    ap.add_argument("--items", nargs="+", default=DEFAULT_ITEMS, help="repo-relative paths to include")
    ap.add_argument("--level", type=int, default=6, choices=range(10), metavar="0-9", help="LZMA2 preset")
    ap.add_argument("--force", action="store_true", help="overwrite an existing archive")
    ap.add_argument("--verify", metavar="ARCHIVE", help="inspect/verify an archive instead of creating one")
    ap.add_argument("--into", metavar="DIR", help="with --verify: extract here and check every hash")
    args = ap.parse_args()

    cmd_verify(args) if args.verify else cmd_pack(args)


if __name__ == "__main__":
    main()
