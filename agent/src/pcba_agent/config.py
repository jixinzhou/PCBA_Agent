from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class RuntimeSettings:
    project_root: Path
    raw: dict[str, Any]

    @property
    def checkpoint_path(self) -> Path:
        return self.project_root / self.raw["checkpoint"]["path"]


def load_settings(
    config_path: Path | None = None, env_path: Path | None = None
) -> RuntimeSettings:
    root = PROJECT_ROOT
    load_env_file(env_path or root / "agent/.env")
    path = config_path or root / "agent/config/runtime.v1.yaml"
    return RuntimeSettings(root, yaml.safe_load(path.read_text(encoding="utf-8")))
