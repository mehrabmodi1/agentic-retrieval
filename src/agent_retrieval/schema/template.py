from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator

from agent_retrieval.schema.experiment import RubricCriterion


class QuestionExample(BaseModel):
    question: str
    answer: str
    needle: str | None = None
    chain: list[dict[str, str]] | None = None
    items: list[dict[str, str]] | None = None


class GridSpec(BaseModel):
    content_profile: list[str]
    corpus_token_count: list[int]
    discriminability: list[Literal["easy", "hard"]]
    reference_clarity: list[Literal["exact", "synonym", "contextual"]]
    n_items: list[int] | None = None


class PayloadTemplateSpec(BaseModel):
    item_type: str


class ExperimentTemplate(BaseModel):
    schema_version: str
    experiment_type: Literal["single_needle", "multi_chain", "multi_reasoning"]
    payload: PayloadTemplateSpec
    question_examples: dict[str, dict[str, QuestionExample]]
    rubric_criteria: list[RubricCriterion]
    grid: GridSpec

    @model_validator(mode="after")
    def validate_grid_n_items(self) -> ExperimentTemplate:
        is_multi = self.experiment_type in ("multi_chain", "multi_reasoning")
        has_n_items = self.grid.n_items is not None
        if is_multi and not has_n_items:
            raise ValueError(
                f"Experiment type '{self.experiment_type}' requires 'n_items' in grid"
            )
        if not is_multi and has_n_items:
            raise ValueError(
                f"Experiment type '{self.experiment_type}' must not have 'n_items' in grid"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentTemplate:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


def _format_token_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}m"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


class Parametrisation(BaseModel):
    experiment_type: str
    content_profile: str
    corpus_token_count: int
    discriminability: str
    reference_clarity: str
    n_items: int | None = None

    @property
    def parametrisation_id(self) -> str:
        parts = [
            self.experiment_type,
            self.content_profile,
            _format_token_count(self.corpus_token_count),
            self.discriminability,
            self.reference_clarity,
        ]
        if self.n_items is not None:
            parts.append(f"n{self.n_items}")
        return "__".join(parts)
