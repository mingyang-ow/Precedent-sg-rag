from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetFile:
    name: str
    role: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    revision: str
    source_url: str
    license: str
    files: tuple[DatasetFile, ...]


def load_manifest(path: Path) -> DatasetManifest:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    return DatasetManifest(
        dataset_id=raw["dataset_id"],
        revision=raw["revision"],
        source_url=raw["source_url"],
        license=raw["license"],
        files=tuple(DatasetFile(**item) for item in raw["files"]),
    )
