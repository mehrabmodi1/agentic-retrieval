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
