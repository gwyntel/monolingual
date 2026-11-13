#!/usr/bin/env python3
"""Build the Monolingual plugin release archive.

The version in plugin.yaml is the single source of truth. A release tag is
accepted only when it is exactly ``v<version>`` so a mistyped or stale tag
cannot publish a mislabelled artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
PLUGIN_VERSION_PATTERN = re.compile(r"^version:\s*([^\s#]+)\s*$", re.MULTILINE)
PLUGIN_FILES = (
    "LICENSE",
    "README.md",
    "plugin.yaml",
    "__init__.py",
    "config.py",
    "detector.py",
    "translator.py",
)


class ReleaseError(ValueError):
    """Raised when a release request is not publishable."""


def read_plugin_version(root: Path) -> str:
    """Return the version declared in plugin.yaml."""
    manifest = root / "plugin.yaml"
    if not manifest.is_file():
        raise ReleaseError(f"missing plugin manifest: {manifest}")
    match = PLUGIN_VERSION_PATTERN.search(manifest.read_text(encoding="utf-8"))
    if match is None:
        raise ReleaseError("plugin.yaml has no top-level version field")
    version = match.group(1).strip("'\"")
    if not VERSION_PATTERN.fullmatch(version):
        raise ReleaseError(f"plugin.yaml version is not a release version: {version!r}")
    return version


def validate_tag(root: Path, tag: str) -> str:
    """Return the version for *tag*, or raise if the tag does not match."""
    if not tag.startswith("v"):
        raise ReleaseError(f"release tag must start with 'v': {tag!r}")
    tag_version = tag[1:]
    if not VERSION_PATTERN.fullmatch(tag_version):
        raise ReleaseError(f"release tag is not a semantic version: {tag!r}")
    plugin_version = read_plugin_version(root)
    if tag_version != plugin_version:
        raise ReleaseError(
            f"tag {tag!r} does not match plugin.yaml version {plugin_version!r}"
        )
    return plugin_version


def release_filename(version: str) -> str:
    """Return the deterministic archive filename for version."""
    return f"monolingual-{version}.zip"


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release(root: Path, tag: str, output_dir: Path) -> tuple[Path, str]:
    """Build the release zip and return ``(path, sha256_digest)``."""
    version = validate_tag(root, tag)
    missing = [name for name in PLUGIN_FILES if not (root / name).is_file()]
    if missing:
        raise ReleaseError("missing plugin release files: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / release_filename(version)
    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in PLUGIN_FILES:
            source = root / name
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            bundle.writestr(info, source.read_bytes(), compresslevel=9)

    return archive, sha256(archive)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag, for example v0.1.1")
    parser.add_argument("--output", type=Path, default=Path("dist"), help="output directory")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="plugin repository root",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        archive, digest = build_release(args.root, args.tag, args.output)
    except ReleaseError as exc:
        print(f"release blocked: {exc}", file=sys.stderr)
        return 1
    print(f"built {archive}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
