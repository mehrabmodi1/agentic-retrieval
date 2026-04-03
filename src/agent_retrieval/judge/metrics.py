from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from agent_retrieval.schema.verdict import SessionMetrics

def extract_session_metrics(jsonl_path: Path) -> SessionMetrics:
    tool_calls: Counter[str] = Counter()
    total_input_tokens = 0
    total_turns = 0
    duration_seconds = 0.0

    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return SessionMetrics(total_context_tokens=0, total_turns=0, tool_calls={}, duration_seconds=0.0)

    with open(jsonl_path) as f:
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
            if entry.get("type") == "result":
                usage = entry.get("usage", {})
                total_input_tokens = usage.get("input_tokens", 0)
                total_turns = entry.get("num_turns", 0)
                duration_seconds = entry.get("duration_ms", 0) / 1000.0

    return SessionMetrics(total_context_tokens=total_input_tokens, total_turns=total_turns,
                          tool_calls=dict(tool_calls), duration_seconds=duration_seconds)
