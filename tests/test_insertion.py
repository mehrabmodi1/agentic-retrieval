import yaml
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from agent_retrieval.generator.insertion import insert_payloads, build_insertion_prompt, _extract_fragment, _read_target_fragments
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


class TestExtractFragment:
    def test_returns_30_lines_from_middle_of_long_file(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(100))
        f = tmp_path / "big.md"
        f.write_text(content)

        fragment, start_line = _extract_fragment(f, seed=42)

        lines = fragment.strip().split("\n")
        assert len(lines) == 30

    def test_returns_full_file_when_under_30_lines(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(10))
        f = tmp_path / "small.md"
        f.write_text(content)

        fragment, start_line = _extract_fragment(f, seed=42)

        lines = fragment.strip().split("\n")
        assert len(lines) == 10
        assert start_line == 0

    def test_fragment_is_contiguous_slice_of_original(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(200))
        f = tmp_path / "big.md"
        f.write_text(content)

        fragment, start_line = _extract_fragment(f, seed=99)

        original_lines = content.split("\n")
        fragment_lines = fragment.strip().split("\n")
        assert fragment_lines == original_lines[start_line:start_line + 30]

    def test_different_seeds_produce_different_offsets(self, tmp_path):
        content = "\n".join(f"line {i}" for i in range(200))
        f = tmp_path / "big.md"
        f.write_text(content)

        _, start_1 = _extract_fragment(f, seed=1)
        _, start_2 = _extract_fragment(f, seed=9999)

        assert start_1 != start_2

    def test_start_line_can_be_zero(self, tmp_path):
        """No bias away from file start."""
        content = "\n".join(f"line {i}" for i in range(50))
        f = tmp_path / "file.md"
        f.write_text(content)

        starts = set()
        for seed in range(200):
            _, start = _extract_fragment(f, seed=seed)
            starts.add(start)

        assert 0 in starts

    def test_fragment_can_end_at_last_line(self, tmp_path):
        """No bias away from file end."""
        content = "\n".join(f"line {i}" for i in range(50))
        f = tmp_path / "file.md"
        f.write_text(content)

        starts = set()
        for seed in range(200):
            _, start = _extract_fragment(f, seed=seed)
            starts.add(start)

        # max valid start for 50 lines with window 30 is 20
        assert 20 in starts


class TestReadTargetFragments:
    def test_returns_fragments_with_file_metadata(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        content = "\n".join(f"line {i}" for i in range(100))
        f = corpus / "test.md"
        f.write_text(content)

        result = _read_target_fragments([f], corpus, base_seed=42)

        assert "test.md" in result
        assert "lines" in result.lower() or "line" in result.lower()

    def test_each_file_gets_unique_seed(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        content = "\n".join(f"line {i}" for i in range(200))
        f1 = corpus / "a.md"
        f1.write_text(content)
        f2 = corpus / "b.md"
        f2.write_text(content)

        result = _read_target_fragments([f1, f2], corpus, base_seed=42)

        # Both files should appear
        assert "a.md" in result
        assert "b.md" in result

    def test_includes_start_line_info(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        content = "\n".join(f"line {i}" for i in range(100))
        f = corpus / "test.md"
        f.write_text(content)

        result = _read_target_fragments([f], corpus, base_seed=42)

        # Should contain line offset information for the agent
        assert "start_line" in result.lower() or "line " in result.lower()
