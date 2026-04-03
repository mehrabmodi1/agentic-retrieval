from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agent_retrieval.schema.experiment import RubricCriterion


class AnswerKeyItem(BaseModel):
    item_id: str
    inserted_text: str
    file_path: str
    line_range: list[int]
    context_summary: str


class ExpectedAnswers(BaseModel):
    question: str
    correctness: str
    completeness: str = ""


class AnswerKey(BaseModel):
    experiment_id: str
    generated_at: str
    parametrisation_id: str | None = None
    parameters: dict[str, Any] | None = None
    items: list[AnswerKeyItem]
    expected_answers: ExpectedAnswers
    rubric_criteria: list[RubricCriterion]

    @classmethod
    def from_yaml(cls, path: Path) -> AnswerKey:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
