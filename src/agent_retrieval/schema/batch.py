from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator


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


class BatchExperimentEntry(BaseModel):
    experiment_type: str
    filter: dict[str, list[Any]] | None = None


class BatchConfigV2(BaseModel):
    batch_name: str
    max_parallel: int
    retry_failed: bool
    judge_model: str
    experiments: list[BatchExperimentEntry]

    @model_validator(mode="before")
    @classmethod
    def normalize_experiments(cls, data: dict) -> dict:
        if "experiments" in data:
            normalized = []
            for entry in data["experiments"]:
                if isinstance(entry, str):
                    normalized.append({"experiment_type": entry})
                else:
                    normalized.append(entry)
            data["experiments"] = normalized
        return data

    @classmethod
    def from_yaml(cls, path: Path) -> BatchConfigV2:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
