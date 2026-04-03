import json
from pathlib import Path
import pytest
from agent_retrieval.judge.metrics import extract_session_metrics
from agent_retrieval.schema.verdict import SessionMetrics


@pytest.fixture
def sample_jsonl(tmp_path) -> Path:
    messages = [
        {"type": "assistant", "timestamp": "2026-04-03T10:00:00Z",
         "message": {"content": [{"type": "tool_use", "name": "Grep", "input": {"pattern": "timeout"}}],
                     "usage": {"input_tokens": 5000, "output_tokens": 200}}},
        {"type": "tool_result", "timestamp": "2026-04-03T10:00:05Z"},
        {"type": "assistant", "timestamp": "2026-04-03T10:00:06Z",
         "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"path": "config.py"}},
                                  {"type": "tool_use", "name": "Grep", "input": {"pattern": "import"}}],
                     "usage": {"input_tokens": 8000, "output_tokens": 300}}},
        {"type": "tool_result", "timestamp": "2026-04-03T10:00:10Z"},
        {"type": "assistant", "timestamp": "2026-04-03T10:00:11Z",
         "message": {"content": [{"type": "text", "text": "The timeout is 42 seconds."}],
                     "usage": {"input_tokens": 10000, "output_tokens": 100}}},
        {"type": "result", "timestamp": "2026-04-03T10:00:30Z",
         "duration_ms": 30000, "num_turns": 3,
         "usage": {"input_tokens": 23000, "output_tokens": 600}},
    ]
    jsonl_path = tmp_path / "session.jsonl"
    with open(jsonl_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return jsonl_path


class TestExtractSessionMetrics:
    def test_extracts_token_counts(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.total_context_tokens == 23000

    def test_extracts_tool_calls(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.tool_calls["Grep"] == 2
        assert metrics.tool_calls["Read"] == 1

    def test_extracts_turns(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.total_turns == 3

    def test_extracts_duration(self, sample_jsonl):
        metrics = extract_session_metrics(sample_jsonl)
        assert metrics.duration_seconds == 30.0

    def test_empty_jsonl(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        metrics = extract_session_metrics(empty)
        assert metrics.total_context_tokens == 0
        assert metrics.total_turns == 0
        assert metrics.tool_calls == {}
