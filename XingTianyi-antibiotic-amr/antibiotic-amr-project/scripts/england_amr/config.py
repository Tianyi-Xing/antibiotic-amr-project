from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    source_path: Path

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def path(self, dotted: str) -> Path:
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Missing configuration key: {dotted}")
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.source_path.parent.parent / path).resolve()

    def ensure_directories(self) -> None:
        for key in (
            "paths.raw_dir",
            "paths.interim_dir",
            "paths.processed_dir",
            "paths.external_dir",
            "paths.model_dir",
            "paths.figure_dir",
            "paths.table_dir",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> Config:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config(raw=raw, source_path=source)
    cfg.ensure_directories()
    return cfg
