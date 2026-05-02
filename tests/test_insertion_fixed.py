from pathlib import Path

import pytest

from agent_retrieval.generator.insertion_fixed import (
    build_fixed_insertion_prompt,
    write_fixed_pool_answer_key,
)
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


@pytest.fixture
def sample_template() -> ExperimentTemplate:
    return ExperimentTemplate.model_validate({
        "experiment_type": "multi_retrieval",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": "Find the {n} canary deployment parameters.",
                    "answer": "List of {n} parameters",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "recall", "weight": 1.0},
            {"criterion": "precision", "weight": 0.3},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "corpus_token_count": [800000],
            "discriminability": ["hard"],
            "reference_clarity": ["contextual"],
            "n_items": [2],
        },
        "fixed_pool": {
            "python_repo": [
                {"inserted_text": "canary_traffic_split = 5", "value": "5",
                 "content_hint": "canary traffic split percentage"},
                {"inserted_text": "evaluation_window_hours = 6", "value": "6",
                 "content_hint": "canary evaluation window"},
            ],
        },
    })


class TestBuildFixedInsertionPrompt:
    def test_prompt_lists_each_selected_needle(self, sample_template):
        param = Parametrisation(
            experiment_type="multi_retrieval",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
            n_items=2,
        )
        selected = sample_template.fixed_pool["python_repo"]  # both items
        prompt = build_fixed_insertion_prompt(
            template=sample_template,
            parametrisation=param,
            selected_items=selected,
            target_files_content="### File: foo.py\n```\npass\n```",
            answer_key_path=Path("/tmp/ak.yaml"),
        )
        assert "canary_traffic_split = 5" in prompt
        assert "evaluation_window_hours = 6" in prompt

    def test_prompt_forbids_inventing_new_needles(self, sample_template):
        param = Parametrisation(
            experiment_type="multi_retrieval",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
            n_items=2,
        )
        selected = sample_template.fixed_pool["python_repo"]
        prompt = build_fixed_insertion_prompt(
            template=sample_template,
            parametrisation=param,
            selected_items=selected,
            target_files_content="...",
            answer_key_path=Path("/tmp/ak.yaml"),
        )
        # Must explicitly forbid invention.
        assert "do not invent" in prompt.lower() or "do not modify" in prompt.lower()


class TestWriteFixedPoolAnswerKey:
    def test_writes_valid_answer_key_with_values(self, sample_template, tmp_path):
        param = Parametrisation(
            experiment_type="multi_retrieval",
            content_profile="python_repo",
            corpus_token_count=800000,
            discriminability="hard",
            reference_clarity="contextual",
            n_items=2,
        )
        selected = sample_template.fixed_pool["python_repo"]
        # Simulate the insertion agent having written file_path/line_range.
        items_with_locations = [
            {**selected[0], "file_path": "deploy/canary.py", "line_range": [10, 10]},
            {**selected[1], "file_path": "deploy/canary.py", "line_range": [20, 20]},
        ]
        ak_path = tmp_path / "ak.yaml"
        write_fixed_pool_answer_key(
            template=sample_template,
            parametrisation=param,
            items_with_locations=items_with_locations,
            answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "multi_retrieval__python_repo__800k__hard__contextual__n2"
        assert len(ak.items) == 2
        assert ak.items[0].value == "5"
        assert "{n}" not in ak.expected_answers.question  # n was substituted
        assert "2" in ak.expected_answers.question
