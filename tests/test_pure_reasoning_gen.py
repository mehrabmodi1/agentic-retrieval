from pathlib import Path

import pytest

from agent_retrieval.generator.pure_reasoning_gen import generate_pure_reasoning_cell
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


@pytest.fixture
def sample_template() -> ExperimentTemplate:
    return ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "You have {n} parameters. Below are the facts:\n\n"
                        "{facts_block}\n\n"
                        "Derive the narrowest safe-migration window."
                    ),
                    "answer": "Window endpoints + citations.",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "endpoint_correctness", "weight": 1.0},
            {"criterion": "classification_accuracy", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "n_items": [2],
        },
        "fixed_pool": {
            "python_repo": [
                {"text": "REPLICATION_LAG_RECOVERY_S = 300",
                 "bound_direction": "lower", "bound_value": "300",
                 "context_summary": "must wait 300s after replication peak"},
                {"text": "BACKUP_WINDOW_OPEN_S = 7200",
                 "bound_direction": "upper", "bound_value": "7200",
                 "context_summary": "must complete before next backup at 7200s"},
            ],
        },
    })


@pytest.fixture
def large_template() -> ExperimentTemplate:
    """Template with 8 lower + 8 upper pool entries for balance_key testing."""
    lower_items = [
        {"text": f"LOWER_CONST_{i} = {100 + i * 10}", "bound_direction": "lower",
         "bound_value": str(100 + i * 10), "context_summary": f"lower fact {i}"}
        for i in range(8)
    ]
    upper_items = [
        {"text": f"UPPER_CONST_{i} = {500 + i * 10}", "bound_direction": "upper",
         "bound_value": str(500 + i * 10), "context_summary": f"upper fact {i}"}
        for i in range(8)
    ]
    return ExperimentTemplate.model_validate({
        "experiment_type": "pure_reasoning",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": (
                        "You have {n} parameters. Below are the facts:\n\n"
                        "{facts_block}\n\n"
                        "Derive the narrowest safe-migration window."
                    ),
                    "answer": "Window endpoints + citations.",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "endpoint_correctness", "weight": 1.0},
            {"criterion": "classification_accuracy", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "n_items": [2],
        },
        "fixed_pool": {
            "python_repo": lower_items + upper_items,
        },
    })


class TestGeneratePureReasoningCell:
    def test_writes_answer_key_with_facts_inlined_in_question(self, sample_template, tmp_path):
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=sample_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert ak.parametrisation_id == "pure_reasoning__python_repo__n2"
        assert len(ak.items) == 2
        # Facts must be inlined into the question prompt.
        assert "REPLICATION_LAG_RECOVERY_S = 300" in ak.expected_answers.question
        assert "BACKUP_WINDOW_OPEN_S = 7200" in ak.expected_answers.question
        # Each item carries its bound metadata.
        directions = sorted(it.bound_direction for it in ak.items)
        assert directions == ["lower", "upper"]

    def test_idempotent_on_existing_answer_key(self, sample_template, tmp_path):
        """Running twice with skip_existing should not overwrite."""
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=sample_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        first = ak_path.read_text()
        generate_pure_reasoning_cell(
            template=sample_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        assert ak_path.read_text() == first  # unchanged


class TestGeneratePureReasoningCellStratified:
    def test_n2_large_pool_always_has_one_lower_and_one_upper(self, large_template, tmp_path):
        """With 8L/8U pool and n=2, balance_key ensures 1 lower + 1 upper — never degenerate."""
        param = Parametrisation(
            experiment_type="pure_reasoning",
            content_profile="python_repo",
            n_items=2,
        )
        ak_path = tmp_path / "ak.yaml"
        generate_pure_reasoning_cell(
            template=large_template,
            parametrisation=param,
            answer_key_path=ak_path,
        )
        ak = AnswerKey.from_yaml(ak_path)
        assert len(ak.items) == 2
        directions = [it.bound_direction for it in ak.items]
        assert "lower" in directions, f"No lower-bound item in generated answer key: {directions}"
        assert "upper" in directions, f"No upper-bound item in generated answer key: {directions}"
