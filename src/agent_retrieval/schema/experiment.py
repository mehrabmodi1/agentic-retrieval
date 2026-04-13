from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


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


# Generator-side model defaults. Single source of truth shared by
# pool generation (pool.py), payload insertion (insertion.py), and
# per-experiment CorpusSpec records. These values are the ones used
# to produce the currently-checked-in corpora; change with care.
POOL_GENERATION_MODEL = "claude-haiku-4-5-20251001"
PAYLOAD_INSERTION_MODEL_SINGLE = "claude-sonnet-4-6"
PAYLOAD_INSERTION_MODEL_MULTI = "claude-haiku-4-5-20251001"


class CorpusSpec(BaseModel):
    content_profile: str
    target_token_count: int
    target_file_count: int
    folder_depth: int
    folder_distribution: Literal["balanced", "skewed", "flat"]
    generation_model: str
    red_herring_density: Literal["none", "low", "medium", "high"]
    pool_generation_model: str = POOL_GENERATION_MODEL
    payload_insertion_model_single: str = PAYLOAD_INSERTION_MODEL_SINGLE
    payload_insertion_model_multi: str = PAYLOAD_INSERTION_MODEL_MULTI


class PayloadSpec(BaseModel):
    insertion_model: str
    red_herring_hint: str
    items: list[PayloadItem]


class RubricCriterion(BaseModel):
    criterion: str
    weight: float


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    experiment_id: str
    experiment_type: str
    corpus: CorpusSpec
    payload: PayloadSpec
    question: str
    rubric_criteria: list[RubricCriterion]

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
