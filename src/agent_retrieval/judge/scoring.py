from __future__ import annotations
import asyncio
import json
import tempfile
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
)

from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import ScoreEntry

JUDGE_MODEL = "claude-sonnet-4-6"

MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds


async def score_response(agent_response: str, answer_key: AnswerKey) -> list[ScoreEntry]:
    criteria_desc = "\n".join(f"- {c.criterion} (weight: {c.weight})" for c in answer_key.rubric_criteria)
    items_desc = "\n".join(f"- {it.item_id}: '{it.inserted_text}' at {it.file_path}" for it in answer_key.items)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    validate_cmd = (
        f"python3 -c \""
        f"import json, sys; "
        f"d = json.load(open('{output_path}')); "
        f"assert 'scores' in d, 'missing scores key'; "
        f"[( assert s.get('criterion'), 'missing criterion', "
        f"  assert isinstance(s.get('score'), (int, float)), 'score not numeric', "
        f"  assert 0.0 <= s['score'] <= 1.0, f'score out of range: {{s[\\\"score\\\"]}}', "
        f"  assert s.get('reasoning'), 'missing reasoning' "
        f") for s in d['scores']]; "
        f"print('VALID')\""
    )

    system_prompt = (
        "You are a strict judge evaluating an AI agent's response to a retrieval question. "
        "Score each criterion from 0.0 to 1.0. "
        f"Write your scores to {output_path} using the Write tool, then validate by running: "
        f"python3 -c \"import json; d=json.load(open('{output_path}')); "
        f"[print('ERROR: missing', k) for s in d['scores'] for k in ['criterion','score','reasoning'] if k not in s]; "
        f"[print('ERROR: score out of range', s['score']) for s in d['scores'] if not 0<=s['score']<=1]; "
        f"print('VALID') if all('criterion' in s and 'score' in s and 'reasoning' in s and 0<=s['score']<=1 for s in d['scores']) else None\" "
        "If the output is not VALID, fix your JSON and validate again before finishing. "
        "The file must contain valid JSON in exactly this format: "
        '{"scores": [{"criterion": "...", "score": 0.0, "reasoning": "..."}]}'
    )

    prompt = (
        f"**Question:** {answer_key.expected_answers.question}\n\n"
        f"**Ground truth items inserted into the codebase:**\n{items_desc}\n\n"
        f"**Expected correct answer:** {answer_key.expected_answers.correctness}\n"
        f"**Expected complete answer:** {answer_key.expected_answers.completeness}\n\n"
        f"**Agent's response:**\n{agent_response}\n\n"
        f"**Scoring criteria:**\n{criteria_desc}\n\n"
        f"Write your scores to {output_path}, validate, and fix if needed."
    )

    options = ClaudeAgentOptions(
        model=JUDGE_MODEL,
        system_prompt=system_prompt,
        allowed_tools=["Write", "Bash"],
        permission_mode="acceptEdits",
        max_turns=5,
    )

    for attempt in range(MAX_RETRIES):
        try:
            output_path.unlink(missing_ok=True)
            async for _ in query(prompt=prompt, options=options):
                pass

            if not output_path.exists():
                raise ValueError("Judge did not write output file")

            result = json.loads(output_path.read_text())
            _validate(result, answer_key)

            weight_map = {c.criterion: c.weight for c in answer_key.rubric_criteria}
            return [
                ScoreEntry(
                    criterion=s["criterion"], score=s["score"],
                    weight=weight_map.get(s["criterion"], 1.0),
                    reasoning=s["reasoning"],
                )
                for s in result["scores"]
            ]

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  Judge failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {backoff}s: {e}")
            await asyncio.sleep(backoff)
        finally:
            output_path.unlink(missing_ok=True)


def _validate(result: dict, answer_key: AnswerKey) -> None:
    scores = result.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError("'scores' must be a non-empty list")
    expected = {c.criterion for c in answer_key.rubric_criteria}
    for s in scores:
        for key in ("criterion", "score", "reasoning"):
            if key not in s:
                raise ValueError(f"Score entry missing '{key}'")
        if not isinstance(s["score"], (int, float)):
            raise ValueError(f"score must be numeric, got {s['score']!r}")
        if not 0.0 <= float(s["score"]) <= 1.0:
            raise ValueError(f"score out of range: {s['score']}")
        if s["criterion"] not in expected:
            raise ValueError(f"unexpected criterion: {s['criterion']!r}")
