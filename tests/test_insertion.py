import yaml
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from agent_retrieval.generator.insertion import insert_payloads, build_insertion_prompt
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation, QuestionExample


@pytest.fixture
def corpus_dir(tmp_path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "config").mkdir()
    (corpus / "config" / "settings.md").write_text("# Settings\nDEBUG = True\nPORT = 8080\n")
    (corpus / "core").mkdir()
    (corpus / "core" / "app.md").write_text("# App\ndef main():\n    pass\n")
    return corpus


@pytest.fixture
def single_template() -> ExperimentTemplate:
    return ExperimentTemplate.model_validate({
        "schema_version": "2.0",
        "experiment_type": "single_needle",
        "payload": {"item_type": "config_value"},
        "question_examples": {
            "python_repo": {
                "easy_exact": {
                    "question": "What is the value of MAX_POOL_SIZE?",
                    "needle": "MAX_POOL_SIZE = 25",
                    "answer": "25",
                },
            },
        },
        "rubric_criteria": [
            {"criterion": "correctness", "weight": 1.0},
            {"criterion": "completeness", "weight": 0.5},
        ],
        "grid": {
            "content_profile": ["python_repo"],
            "corpus_token_count": [20000],
            "discriminability": ["easy"],
            "reference_clarity": ["exact"],
        },
        "runner": {
            "n_repeats": 3,
            "agent_model": "claude-sonnet-4-6",
            "max_tokens": 100000,
            "allowed_tools": ["Read", "Glob", "Grep", "Bash"],
        },
    })


@pytest.fixture
def parametrisation() -> Parametrisation:
    return Parametrisation(
        experiment_type="single_needle",
        content_profile="python_repo",
        corpus_token_count=20000,
        discriminability="easy",
        reference_clarity="exact",
    )


class TestBuildInsertionPrompt:
    def test_contains_experiment_type(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"), "file content here")
        assert "single_needle" in prompt

    def test_contains_discriminability_rubric(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"), "file content here")
        assert "easy" in prompt
        assert "Findable by exact string search" in prompt

    def test_contains_reference_clarity(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"), "file content here")
        assert "exact" in prompt

    def test_contains_examples(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"), "file content here")
        assert "MAX_POOL_SIZE" in prompt

    def test_contains_answer_key_path(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/tmp/answer.yaml"), "file content here")
        assert "/tmp/answer.yaml" in prompt

    def test_contains_target_files(self, single_template, parametrisation):
        prompt = build_insertion_prompt(single_template, parametrisation, Path("/answer.yaml"), "### File: config/settings.md\n```\nDEBUG=True\n```")
        assert "config/settings.md" in prompt
        assert "DEBUG=True" in prompt

    def test_multi_chain_specifies_n_items(self):
        tmpl = ExperimentTemplate.model_validate({
            "schema_version": "2.0",
            "experiment_type": "multi_chain",
            "payload": {"item_type": "cross_reference"},
            "question_examples": {
                "python_repo": {
                    "easy_exact": {
                        "question": "Follow the chain.",
                        "chain": [{"needle": "A", "file_context": "a.md"}],
                        "answer": "A",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
                "n_items": [4],
            },
            "runner": {
                "n_repeats": 1,
                "agent_model": "claude-sonnet-4-6",
                "max_tokens": 100000,
                "allowed_tools": ["Read"],
            },
        })
        param = Parametrisation(
            experiment_type="multi_chain",
            content_profile="python_repo",
            corpus_token_count=20000,
            discriminability="easy",
            reference_clarity="exact",
            n_items=4,
        )
        prompt = build_insertion_prompt(tmpl, param, Path("/answer.yaml"), "file content")
        assert "4" in prompt
        assert "chain" in prompt.lower() or "sequential" in prompt.lower()


class TestInsertPayloads:
    @pytest.mark.asyncio
    async def test_calls_agent_sdk(self, corpus_dir, single_template, parametrisation, tmp_path):
        answer_key_path = tmp_path / "answer_keys" / f"{parametrisation.parametrisation_id}.yaml"

        # Simulate the agent writing the answer key
        async def fake_query(prompt, options):
            answer_key_path.parent.mkdir(parents=True, exist_ok=True)
            answer_key_path.write_text(yaml.dump({
                "parametrisation_id": parametrisation.parametrisation_id,
                "experiment_type": "single_needle",
                "generated_at": "2026-04-03T10:00:00Z",
                "parameters": {
                    "content_profile": "python_repo",
                    "corpus_token_count": 20000,
                    "discriminability": "easy",
                    "reference_clarity": "exact",
                },
                "items": [{
                    "item_id": "target_001",
                    "inserted_text": "MAX_POOL_SIZE = 25",
                    "file_path": "config/settings.md",
                    "line_range": [3, 3],
                    "context_summary": "Added as config constant",
                }],
                "expected_answers": {
                    "question": "What is the value of MAX_POOL_SIZE?",
                    "correctness": "25",
                    "completeness": "Found in config/settings.md",
                },
                "rubric_criteria": [
                    {"criterion": "correctness", "weight": 1.0},
                    {"criterion": "completeness", "weight": 0.5},
                ],
            }))
            result = MagicMock()
            result.session_id = "test-session"
            yield result

        with patch("agent_retrieval.generator.insertion.query", side_effect=fake_query):
            await insert_payloads(
                template=single_template,
                parametrisation=parametrisation,
                corpus_dir=corpus_dir,
                answer_key_path=answer_key_path,
            )

        assert answer_key_path.exists()
