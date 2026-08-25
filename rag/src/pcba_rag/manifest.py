from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    file_path: Path
    sha256: str
    title: str
    language: str


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_enabled_sources(
    project_root: Path | None = None,
    manifest_relative_path: str = "rag/config/sources.v0.3.yaml",
) -> list[SourceSpec]:
    root = (project_root or default_project_root()).resolve()
    manifest_path = root / manifest_relative_path
    data: dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sources: list[SourceSpec] = []
    for item in data["sources"]:
        if not item.get("enabled", False):
            continue
        sources.append(
            SourceSpec(
                source_id=item["source_id"],
                file_path=(root / item["file_path"]).resolve(),
                sha256=item["sha256"],
                title=item["title"],
                language=item["language"],
            )
        )
    if not sources:
        raise ValueError("Source manifest contains no enabled sources")
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Duplicate source_id in source manifest")
    return sources


def validate_source(source: SourceSpec) -> None:
    if not source.file_path.is_file():
        raise FileNotFoundError(source.file_path)
    actual = sha256_file(source.file_path)
    if actual != source.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source.source_id}: expected {source.sha256}, got {actual}"
        )
