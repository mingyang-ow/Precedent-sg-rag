from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from .manifest import DatasetFile, load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "dataset_manifest.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: DatasetFile) -> None:
    actual_size = path.stat().st_size
    if actual_size != artifact.size:
        raise ValueError(f"{path.name}: expected {artifact.size} bytes, found {actual_size}")
    if artifact.sha256:
        actual_digest = sha256_file(path)
        if actual_digest != artifact.sha256:
            raise ValueError(
                f"{path.name}: expected SHA-256 {artifact.sha256}, found {actual_digest}"
            )


def download_file(url: str, destination: Path, artifact: DatasetFile) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_file(destination, artifact)
        print(f"verified existing {destination}")
        return

    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(url) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())
            verify_file(temporary_path, artifact)
            temporary_path.replace(destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    print(f"downloaded and verified {destination}")


def build_url(dataset_id: str, revision: str, filename: str) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset_id}/resolve/"
        f"{revision}/{quote(filename)}?download=true"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the pinned SG-LegalCite release")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-benchmark",
        action="store_true",
        help="also download the authors' lookup and 1000-way candidate pools",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected = [
        artifact for artifact in manifest.files if artifact.role == "core" or args.include_benchmark
    ]
    try:
        for artifact in selected:
            download_file(
                build_url(manifest.dataset_id, manifest.revision, artifact.name),
                args.output_dir / artifact.name,
                artifact,
            )
    except (OSError, ValueError) as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
