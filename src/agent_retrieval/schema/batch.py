from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class BatchRunConfig(BaseModel):
    experiment_id: str
    n_repeats: int
    agent_model: str | None = None
    judge_model: str | None = None


class BatchConfig(BaseModel):
    batch_name: str
    max_parallel: int
    retry_failed: bool
    judge_model: str
    runs: list[BatchRunConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> BatchConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
