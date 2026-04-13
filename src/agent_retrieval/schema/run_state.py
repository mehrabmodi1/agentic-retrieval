from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class RunState(BaseModel):
    parametrisation_id: str
    run_id: str
    batch_name: str
    status: Literal["pending", "running", "completed", "failed"]
    claude_code_version: str
    agent_model: str = ""
    effort_mode: str = ""
    max_turns: int = 0
    allowed_tools: list[str] = []
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> RunState:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
