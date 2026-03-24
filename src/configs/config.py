"""Run configuration for the RCSA QC pipeline."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class RunConfig(BaseModel):
    output_dir: Path = Field(default=Path("./output"))
    save_summary: bool = True
    save_presentations: bool = True
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def load(cls, path: Path | None = None) -> RunConfig:
        target = Path(path) if path else _PROJECT_ROOT / "config.yaml"
        if target.exists():
            with open(target, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()

    def resolve_output_dir(self) -> Path:
        out = self.output_dir if self.output_dir.is_absolute() else _PROJECT_ROOT / self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
