import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from agent_retrieval.judge.scoring import score_response
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import ScoreEntry


@pytest.fixture
def answer_key() -> AnswerKey:
    return AnswerKey.model_validate({
        "experiment_id": "test-001",
        "generated_at": "2026-04-03T10:00:00Z",
        "items": [{"item_id": "target_001", "inserted_text": "TIMEOUT = 42",
                    "file_path": "src/config.py", "line_range": [10, 10],
                    "context_summary": "Module-level constant"}],
        "expected_answers": {"question": "What is the timeout?",
                             "correctness": "42 seconds in src/config.py",
                             "completeness": "Found in config.py"},
        "rubric_criteria": [{"criterion": "correctness", "weight": 1.0},
                            {"criterion": "completeness", "weight": 0.5}],
    })


class TestScoreResponse:
    @pytest.mark.asyncio
    async def test_returns_score_entries(self, answer_key):
        mock_response_text = json.dumps({
            "scores": [
                {"criterion": "correctness", "score": 0.9, "reasoning": "Correct value found"},
                {"criterion": "completeness", "score": 1.0, "reasoning": "All items found"},
            ]
        })
        mock_response = AsyncMock()
        mock_response.content = [type("TB", (), {"text": mock_response_text})()]

        with patch("agent_retrieval.judge.scoring.get_llm_client") as mock_client:
            mock_client.return_value.messages.create = AsyncMock(return_value=mock_response)
            scores = await score_response(
                agent_response="The timeout is 42 seconds, found in src/config.py",
                answer_key=answer_key,
                judge_model="opus",
            )
        assert len(scores) == 2
        assert scores[0].criterion == "correctness"
        assert scores[0].score == 0.9
