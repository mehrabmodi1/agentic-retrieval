from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agent_retrieval.generator.generate import generate_experiment_v2
from agent_retrieval.schema.template import ExperimentTemplate


@pytest.fixture
def pure_reasoning_template_yaml(tmp_path: Path) -> Path:
    template = {
        "experiment_type": "pure_reasoning",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": "You have {n} facts: {facts_block}. Derive window.",
                    "answer": "answer",
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
                {"text": "A", "bound_direction": "lower", "bound_value": "100"},
                {"text": "B", "bound_direction": "upper", "bound_value": "500"},
            ],
        },
    }
    p = tmp_path / "pure_reasoning.yaml"
    p.write_text(yaml.dump(template))
    return p


class TestGenerateDispatchPureReasoning:
    @pytest.mark.asyncio
    async def test_pure_reasoning_writes_answer_key_without_corpus(
        self, pure_reasoning_template_yaml, tmp_workspace
    ):
        template = ExperimentTemplate.from_yaml(pure_reasoning_template_yaml)
        workspace = tmp_workspace / "workspace"

        # No SDK call should occur for pure_reasoning generation.
        with patch("agent_retrieval.generator.generate.generate_pool") as mock_pool, \
             patch("agent_retrieval.generator.generate.assemble_corpus") as mock_assemble, \
             patch("agent_retrieval.generator.generate.insert_payloads") as mock_insert:
            generated = await generate_experiment_v2(template, workspace)

        assert mock_pool.call_count == 0
        assert mock_assemble.call_count == 0
        assert mock_insert.call_count == 0

        ak_path = workspace / "judge" / "answer_keys" / "pure_reasoning__python_repo__n2.yaml"
        assert ak_path.exists()
        assert generated == ["pure_reasoning__python_repo__n2"]


@pytest.fixture
def multi_retrieval_template_yaml(tmp_path: Path) -> Path:
    template = {
        "experiment_type": "multi_retrieval",
        "payload": {"item_type": "fact"},
        "question_examples": {
            "python_repo": {
                "hard_contextual": {
                    "question": "Find the {n} canary deployment parameters.",
                    "answer": "answer",
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
                {"inserted_text": "A = 1", "value": "1", "content_hint": "x"},
                {"inserted_text": "B = 2", "value": "2", "content_hint": "y"},
            ],
        },
    }
    p = tmp_path / "multi_retrieval.yaml"
    p.write_text(yaml.dump(template))
    return p


class TestGenerateDispatchMultiRetrieval:
    @pytest.mark.asyncio
    async def test_multi_retrieval_routes_to_fixed_insertion(
        self, multi_retrieval_template_yaml, tmp_workspace
    ):
        template = ExperimentTemplate.from_yaml(multi_retrieval_template_yaml)
        workspace = tmp_workspace / "workspace"

        # The pool dir must exist with at least one file so the corpus-based path
        # doesn't try to call generate_pool. We patch generate_pool just to be safe.
        pool_dir = workspace / "background_corpora" / "python_repo"
        pool_dir.mkdir(parents=True, exist_ok=True)
        (pool_dir / "stub.py").write_text("# stub\npass\n")

        with patch("agent_retrieval.generator.generate.generate_pool"), \
             patch("agent_retrieval.generator.generate.assemble_corpus"), \
             patch("agent_retrieval.generator.generate.insert_fixed_payloads") as mock_fixed, \
             patch("agent_retrieval.generator.generate.insert_payloads") as mock_regular:
            await generate_experiment_v2(template, workspace)

        assert mock_fixed.call_count == 1, (
            f"Expected insert_fixed_payloads to be called once, got {mock_fixed.call_count}"
        )
        assert mock_regular.call_count == 0, (
            f"Expected insert_payloads NOT to be called, got {mock_regular.call_count}"
        )
