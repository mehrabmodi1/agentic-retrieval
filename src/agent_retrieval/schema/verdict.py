from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ScoreEntry(BaseModel):
    criterion: str
    score: float
    weight: float
    reasoning: str


class SessionMetrics(BaseModel):
    total_context_tokens: int
    total_turns: int
    tool_calls: dict[str, int]
    duration_seconds: float


class Verdict(BaseModel):
    parametrisation_id: str
    run_id: str
    batch_name: str
    scores: list[ScoreEntry]
    weighted_score: float
    session_metrics: SessionMetrics

    @classmethod
    def from_yaml(cls, path: Path) -> Verdict:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
