import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.generator.payload import PayloadInserter
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.experiment import ExperimentSpec


@pytest.fixture
def spec(sample_spec_dict) -> ExperimentSpec:
    return ExperimentSpec.model_validate(sample_spec_dict)


@pytest.fixture
def corpus_with_files(tmp_workspace, spec) -> Path:
    corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id
    corpus_dir.mkdir(parents=True)
    for i in range(5):
        f = corpus_dir / "src" / f"module_{i}.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# Module {i}\nimport os\n\ndef func_{i}():\n    pass\n")
    return corpus_dir


class TestPayloadInserter:
    @pytest.mark.asyncio
    async def test_inserts_payload_and_produces_answer_key(self, spec, corpus_with_files, tmp_workspace):
        answer_key_path = tmp_workspace / "workspace" / "judge" / "answer_keys" / f"{spec.experiment_id}.yaml"

        mock_insertion = json.dumps({
            "modified_content": "# Module 0\nimport os\n\nCONNECTION_TIMEOUT = 42\n\ndef func_0():\n    pass\n",
            "inserted_text": "CONNECTION_TIMEOUT = 42",
            "line_range": [3, 3],
            "context_summary": "Added as module-level constant",
        })
        mock_expected = json.dumps({
            "correctness": "42 seconds in src/module_0.py",
            "completeness": "Found the timeout value",
        })

        call_count = 0
        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            if call_count == 1:
                resp.content = [type("TB", (), {"text": mock_insertion})()]
            else:
                resp.content = [type("TB", (), {"text": mock_expected})()]
            return resp

        with patch("agent_retrieval.generator.payload.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=mock_create)
            inserter = PayloadInserter()
            answer_key = await inserter.insert(spec, corpus_with_files, answer_key_path)

        assert isinstance(answer_key, AnswerKey)
        assert len(answer_key.items) == 1
        assert answer_key.items[0].item_id == "target_001"
        assert answer_key.items[0].inserted_text == "CONNECTION_TIMEOUT = 42"
        assert answer_key_path.exists()

    @pytest.mark.asyncio
    async def test_dependency_order_respected(self, sample_spec_dict, tmp_workspace):
        sample_spec_dict["payload"]["items"].append({
            "item_id": "target_002",
            "depends_on": "target_001",
            "item_type": "cross_reference",
            "content_hint": "Imports timeout from target_001",
            "placement": {"strategy": "random_file"},
            "camouflage": "low",
        })
        spec = ExperimentSpec.model_validate(sample_spec_dict)

        corpus_dir = tmp_workspace / "workspace" / "runner" / "corpora" / spec.experiment_id
        corpus_dir.mkdir(parents=True)
        for i in range(5):
            f = corpus_dir / "src" / f"module_{i}.py"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"# Module {i}\n")

        answer_key_path = tmp_workspace / "workspace" / "judge" / "answer_keys" / f"{spec.experiment_id}.yaml"

        call_count = 0
        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            # First 2 calls are insertions, 3rd is expected answers
            if call_count <= 2:
                resp.content = [type("TB", (), {"text": json.dumps({
                    "modified_content": "# modified\n",
                    "inserted_text": "INSERTED",
                    "line_range": [1, 1],
                    "context_summary": "test",
                })})()]
            else:
                resp.content = [type("TB", (), {"text": json.dumps({
                    "correctness": "test",
                    "completeness": "test",
                })})()]
            return resp

        with patch("agent_retrieval.generator.payload.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(side_effect=mock_create)
            inserter = PayloadInserter()
            answer_key = await inserter.insert(spec, corpus_dir, answer_key_path)

        assert len(answer_key.items) == 2
        assert answer_key.items[0].item_id == "target_001"
        assert answer_key.items[1].item_id == "target_002"
