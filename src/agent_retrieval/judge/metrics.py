from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from agent_retrieval.schema.verdict import SessionMetrics


def extract_session_metrics(session_path: Path, response_path: Path | None = None) -> SessionMetrics:
    tool_calls: Counter[str] = Counter()
    total_context_tokens = 0
    total_turns = 0
    duration_seconds = 0.0

    # Extract tool calls from session.jsonl
    if session_path.exists() and session_path.stat().st_size > 0:
        with open(session_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("type") == "assistant":
                    message = entry.get("message", {})
                    content = message.get("content", [])
                    for block in content:
                        if block.get("type") == "tool_use":
                            tool_calls[block["name"]] += 1

    # Extract usage and turns from response.json
    if response_path and response_path.exists():
        data = json.loads(response_path.read_text())
        usage = data.get("usage", {})
        total_context_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        total_turns = data.get("num_turns", 0)

    return SessionMetrics(total_context_tokens=total_context_tokens, total_turns=total_turns,
                          tool_calls=dict(tool_calls), duration_seconds=duration_seconds)
