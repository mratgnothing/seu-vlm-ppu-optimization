#!/usr/bin/env python3
"""Build and verify a reproducible, source-only submission candidate ZIP."""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path, PurePosixPath


PACKAGE_ROOT = "seu-vlm-ppu-optimization"
FIXED_ZIP_TIMESTAMP = (2026, 7, 24, 0, 0, 0)

ROOT_FILES = (
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "PROJECT_CONTEXT.md",
    "README.md",
    "README_ORGANIZER.md",
    "benchmark_public.py",
    "evaluation_wrapper.py",
    "pyproject.toml",
    "requirements-local.txt",
    "requirements-lock.txt",
    "requirements.txt",
)

SINGLE_FILES = (
    "configs/README.md",
    "configs/local.example.psd1",
    "configs/model-lock.json",
    "configs/organizer-lock.json",
    "data/README.md",
    "models/README.md",
    "results/README.md",
)

INCLUDE_TREES = (
    "docs",
    "ppu",
    "scripts",
    "submission",
    "tests",
)

EXCLUDED_PARTS = {
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "logs",
    "raw",
    "venv",
}

EXCLUDED_NAMES = {
    ".env",
    "configs/local.psd1",
    "desktop.ini",
    "Thumbs.db",
}

EXCLUDED_SUFFIXES = {
    ".7z",
    ".bin",
    ".ckpt",
    ".key",
    ".log",
    ".onnx",
    ".p12",
    ".pdf",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".pyc",
    ".rar",
    ".safetensors",
    ".tsv",
    ".zip",
}

SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[A-Z0-9]{16}\b")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible source-only submission candidate"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/submission-source.zip"),
        help="Output ZIP path, relative to --root unless absolute",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Audit and list included files without creating a ZIP",
    )
    return parser.parse_args()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_excluded(relative: str) -> bool:
    path = PurePosixPath(relative)
    if relative in EXCLUDED_NAMES:
        return True
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def collect_files(root: Path) -> list[Path]:
    root = root.resolve()
    candidates: list[Path] = []
    for relative in (*ROOT_FILES, *SINGLE_FILES):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required submission source is missing: {relative}")
        candidates.append(path)

    for relative in INCLUDE_TREES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required submission directory is missing: {relative}")
        candidates.extend(path for path in directory.rglob("*") if path.is_file())

    included: dict[str, Path] = {}
    for path in candidates:
        relative = _relative_path(path, root)
        if not _is_excluded(relative):
            included[relative] = path
    return [included[key] for key in sorted(included)]


def audit_file(path: Path, root: Path) -> None:
    relative = _relative_path(path, root)
    if _is_excluded(relative):
        raise ValueError(f"Excluded file reached package audit: {relative}")
    content = path.read_bytes()
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(content):
            raise ValueError(f"Potential {label} found in submission file: {relative}")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _zip_info(archive_name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(archive_name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = mode << 16
    return info


def create_package(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output if output.is_absolute() else root / output
    output = output.resolve()
    files = collect_files(root)
    for path in files:
        audit_file(path, root)

    entries: list[tuple[str, bytes, bool]] = []
    manifest_lines: list[str] = []
    for path in files:
        relative = _relative_path(path, root)
        content = path.read_bytes()
        archive_name = f"{PACKAGE_ROOT}/{relative}"
        executable = path.suffix.lower() == ".sh"
        entries.append((archive_name, content, executable))
        manifest_lines.append(f"{_sha256(content)}  {relative}")

    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for archive_name, content, executable in entries:
            archive.writestr(
                _zip_info(archive_name, executable=executable),
                content,
            )
        archive.writestr(
            _zip_info(f"{PACKAGE_ROOT}/MANIFEST.sha256"),
            manifest,
        )

    verification = verify_package(output)
    return {
        "output": str(output),
        "file_count": len(files),
        "archive_sha256": _sha256(output.read_bytes()),
        **verification,
    }


def verify_package(path: Path) -> dict[str, object]:
    path = path.resolve()
    manifest_name = f"{PACKAGE_ROOT}/MANIFEST.sha256"
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Submission package contains duplicate paths")
        if manifest_name not in names:
            raise ValueError("Submission package is missing MANIFEST.sha256")
        if any(
            member.date_time != FIXED_ZIP_TIMESTAMP
            for member in members
        ):
            raise ValueError("Submission package contains a non-reproducible timestamp")

        manifest_lines = (
            archive.read(manifest_name).decode("utf-8").splitlines()
        )
        expected: dict[str, str] = {}
        for line in manifest_lines:
            digest, separator, relative = line.partition("  ")
            if not separator or len(digest) != 64:
                raise ValueError(f"Malformed manifest line: {line}")
            expected[f"{PACKAGE_ROOT}/{relative}"] = digest

        payload_names = set(names) - {manifest_name}
        if payload_names != set(expected):
            raise ValueError("Manifest paths do not match package payload")
        for name, digest in expected.items():
            if _sha256(archive.read(name)) != digest:
                raise ValueError(f"Manifest hash mismatch: {name}")

    return {
        "verified": True,
        "manifest_entry_count": len(expected),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    files = collect_files(root)
    for path in files:
        audit_file(path, root)

    if args.list_only:
        for path in files:
            print(_relative_path(path, root))
        print(f"Audited {len(files)} source files; no package was created.")
        return 0

    report = create_package(root, args.output)
    print(f"Package: {report['output']}")
    print(f"Files: {report['file_count']}")
    print(f"SHA-256: {report['archive_sha256']}")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
