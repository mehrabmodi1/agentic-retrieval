import inspect
from unittest.mock import AsyncMock, patch

import pytest

from agent_retrieval.judge import scoring
from agent_retrieval.judge.scoring import JUDGE_MODEL
from agent_retrieval.schema.verdict import ScoreEntry


def test_judge_model_pinned():
    assert JUDGE_MODEL == "claude-sonnet-4-6"


def test_score_response_signature_drops_judge_model():
    sig = inspect.signature(scoring.score_response)
    assert "judge_model" not in sig.parameters


@pytest.mark.asyncio
async def test_judge_run_records_judge_model(tmp_path):
    """judge_run writes JUDGE_MODEL into the verdict file."""
    from agent_retrieval.judge.judge import judge_run
    from agent_retrieval.schema.answer_key import AnswerKey

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "response.json").write_text(
        '{"response_text": "the answer is 42",'
        ' "session_id": "sid", "num_turns": 1,'
        ' "total_cost_usd": 0.0, "usage": {}}'
    )
    (run_dir / "session.jsonl").write_text("")

    ak = AnswerKey.model_validate({
        "parametrisation_id": "pid",
        "generated_at": "2026-04-13T00:00:00Z",
        "items": [{
            "item_id": "target_001",
            "inserted_text": "X = 42",
            "file_path": "f.md",
            "line_range": [1, 1],
            "context_summary": "t",
        }],
        "expected_answers": {
            "question": "What is X?",
            "correctness": "42",
        },
        "rubric_criteria": [{"criterion": "correctness", "weight": 1.0}],
    })

    verdict_path = tmp_path / "verdict.yaml"

    fake_scores = [ScoreEntry(criterion="correctness", score=1.0,
                              weight=1.0, reasoning="ok")]
    with patch("agent_retrieval.judge.judge.score_response",
               new=AsyncMock(return_value=fake_scores)):
        verdict = await judge_run(
            run_dir=run_dir, answer_key=ak,
            batch_run_name="b", verdict_path=verdict_path,
        )

    assert verdict.judge_model == JUDGE_MODEL
