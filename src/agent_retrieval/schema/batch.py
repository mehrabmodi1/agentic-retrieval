from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from agent_retrieval.schema.template import GridSpec


class BatchExperimentEntry(BaseModel):
    experiment_type: str
    filter: dict[str, list[Any]] | None = None
    grid: GridSpec | None = None

    @model_validator(mode="after")
    def validate_grid_filter_exclusive(self) -> BatchExperimentEntry:
        if self.grid is not None and self.filter is not None:
            raise ValueError(
                "BatchExperimentEntry cannot specify both 'grid' and 'filter' "
                "(grid declares the run set directly; filter narrows the template grid)"
            )
        return self


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_name: str
    max_parallel: int
    retry_failed: bool
    agent_model: str
    effort_mode: Literal["low", "medium", "high", "max"]
    n_repeats: int
    max_turns: int
    allowed_tools: list[str]
    experiments: list[BatchExperimentEntry]

    @model_validator(mode="before")
    @classmethod
    def normalize_experiments(cls, data: dict) -> dict:
        if isinstance(data, dict) and "experiments" in data:
            normalized = []
            for entry in data["experiments"]:
                if isinstance(entry, str):
                    normalized.append({"experiment_type": entry})
                else:
                    normalized.append(entry)
            data["experiments"] = normalized
        return data

    @classmethod
    def from_yaml(cls, path: Path) -> BatchConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
