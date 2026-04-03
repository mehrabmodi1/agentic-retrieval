from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


@dataclass
class AgentResult:
    response_text: str
    session_id: str
    num_turns: int
    total_cost_usd: float | None
    usage: dict


def get_claude_version() -> str:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


def _find_session_jsonl(session_id: str, cwd: str) -> Path | None:
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None
    for project_dir in claude_dir.iterdir():
        candidate = project_dir / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


async def run_agent_session(
    question: str,
    corpus_dir: Path,
    model: str,
    allowed_tools: list[str],
    max_tokens: int,
    run_id: str,
    run_dir: Path,
) -> AgentResult:
    system_prompt = (
        f"Answer the following question by searching the provided codebase. "
        f"Your session ID is: {run_id}\n\n"
        f"Question: {question}"
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        cwd=str(corpus_dir),
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        max_turns=50,
    )
    response_parts: list[str] = []
    session_id = ""
    num_turns = 0
    total_cost: float | None = None
    usage: dict = {}

    async for message in query(prompt=question, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            session_id = message.session_id
            num_turns = message.num_turns
            total_cost = message.total_cost_usd
            usage = message.usage or {}

    # Copy session JSONL from SDK storage to our run directory
    if session_id:
        jsonl_src = _find_session_jsonl(session_id, str(corpus_dir))
        if jsonl_src:
            import shutil
            shutil.copy2(jsonl_src, run_dir / "session.jsonl")

    return AgentResult(
        response_text="\n".join(response_parts),
        session_id=session_id,
        num_turns=num_turns,
        total_cost_usd=total_cost,
        usage=usage,
    )
