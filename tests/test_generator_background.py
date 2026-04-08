import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.generator.background import BackgroundGenerator
from agent_retrieval.schema.experiment import ExperimentSpec


@pytest.fixture
def spec(sample_spec_dict) -> ExperimentSpec:
    return ExperimentSpec.model_validate(sample_spec_dict)


class TestBackgroundGenerator:
    @pytest.mark.asyncio
    async def test_creates_folder_structure(self, spec, tmp_workspace):
        corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id

        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": "# Generated Python file\nimport os\n\ndef main():\n    pass\n"})()]

        with patch("agent_retrieval.generator.background.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            gen = BackgroundGenerator()
            await gen.generate(spec, corpus_dir)

        assert corpus_dir.exists()
        files = list(corpus_dir.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        assert file_count == spec.corpus.target_file_count

    @pytest.mark.asyncio
    async def test_red_herring_files_created(self, spec, tmp_workspace):
        corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id

        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": "# content\nx = 1\n"})()]

        with patch("agent_retrieval.generator.background.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            gen = BackgroundGenerator()
            red_herring_hint = "Variables with similar names"
            await gen.generate(spec, corpus_dir, red_herring_hint=red_herring_hint)

        assert corpus_dir.exists()


from unittest.mock import MagicMock
from agent_retrieval.generator.generate import generate_experiment_v2
from agent_retrieval.schema.template import ExperimentTemplate


class TestGenerateExperimentV2:
    @pytest.mark.asyncio
    async def test_orchestrates_all_phases(self, tmp_workspace):
        workspace_dir = tmp_workspace / "workspace"
        template_dict = {
            "schema_version": "2.0",
            "experiment_type": "single_needle",
            "payload": {"item_type": "config_value"},
            "question_examples": {
                "python_repo": {
                    "easy_exact": {
                        "question": "What is X?",
                        "needle": "X = 1",
                        "answer": "1",
                    },
                },
            },
            "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
            "grid": {
                "content_profile": ["python_repo"],
                "corpus_token_count": [20000],
                "discriminability": ["easy"],
                "reference_clarity": ["exact"],
            },
            "runner": {
                "n_repeats": 1,
                "agent_model": "claude-sonnet-4-6",
                "max_tokens": 100000,
                "allowed_tools": ["Read"],
            },
        }
        template = ExperimentTemplate.model_validate(template_dict)

        # Pre-create a pool so pool generation is skipped
        pool_dir = workspace_dir / "background_corpora" / "python_repo"
        pool_dir.mkdir(parents=True)
        for i in range(10):
            (pool_dir / f"file_{i}.md").write_text(f"# File {i}\n" + "content " * 500)

        with patch("agent_retrieval.generator.generate.insert_payloads", new_callable=AsyncMock) as mock_insert, \
             patch("agent_retrieval.generator.generate.generate_pool", new_callable=AsyncMock) as mock_pool:
            await generate_experiment_v2(template, workspace_dir)

        # Pool generation should NOT have been called (pool exists with enough tokens)
        mock_pool.assert_not_called()
        # Insertion should have been called once (1 parametrisation)
        assert mock_insert.call_count == 1
        # Corpus should have been assembled
        corpus_dir = workspace_dir / "runner" / "corpora" / "single_needle__python_repo__20k__easy__exact"
        assert corpus_dir.exists()
        assert len(list(corpus_dir.rglob("*.md"))) > 0
