from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class PlacementConfig(BaseModel):
    strategy: Literal["random_file", "specific_depth", "specific_filetype"]
    depth: int | None = None
    filetype: str | None = None


class PayloadItem(BaseModel):
    item_id: str
    item_type: Literal["config_value", "function_def", "fact", "cross_reference"]
    content_hint: str
    placement: PlacementConfig
    camouflage: Literal["low", "medium", "high"]
    depends_on: str | None = None


class CorpusSpec(BaseModel):
    content_profile: str
    target_token_count: int
    target_file_count: int
    folder_depth: int
    folder_distribution: Literal["balanced", "skewed", "flat"]
    generation_model: str
    red_herring_density: Literal["none", "low", "medium", "high"]


class PayloadSpec(BaseModel):
    insertion_model: str
    red_herring_hint: str
    items: list[PayloadItem]


class RubricCriterion(BaseModel):
    criterion: str
    weight: float


class RunnerSpec(BaseModel):
    n_repeats: int
    agent_model: str
    max_tokens: int
    allowed_tools: list[str]
    effort_mode: str = ""


class ExperimentSpec(BaseModel):
    schema_version: str
    experiment_id: str
    experiment_type: str
    corpus: CorpusSpec
    payload: PayloadSpec
    question: str
    rubric_criteria: list[RubricCriterion]
    runner: RunnerSpec

    @model_validator(mode="after")
    def validate_depends_on_references(self) -> ExperimentSpec:
        item_ids = {item.item_id for item in self.payload.items}
        for item in self.payload.items:
            if item.depends_on and item.depends_on not in item_ids:
                raise ValueError(
                    f"Item '{item.item_id}' depends_on '{item.depends_on}' "
                    f"which is not defined in payload items"
                )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentSpec:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
