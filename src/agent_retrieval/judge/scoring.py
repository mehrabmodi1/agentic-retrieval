from __future__ import annotations
import asyncio
import json

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import ScoreEntry

# Pinned judge model. Standardised across all batches so that judge
# quality is held constant when comparing agent models / effort modes.
JUDGE_MODEL = "claude-sonnet-4-6"

MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds


async def score_response(agent_response: str, answer_key: AnswerKey) -> list[ScoreEntry]:
    criteria_desc = "\n".join(f"- {c.criterion} (weight: {c.weight})" for c in answer_key.rubric_criteria)
    items_desc = "\n".join(f"- {it.item_id}: '{it.inserted_text}' at {it.file_path}" for it in answer_key.items)

    system_prompt = (
        "You are a strict judge evaluating an AI agent's response to a retrieval question. "
        "Score each criterion from 0.0 to 1.0. Return ONLY valid JSON in this format: "
        '{"scores": [{"criterion": "...", "score": 0.0, "reasoning": "..."}]}'
    )

    prompt = (
        f"**Question:** {answer_key.expected_answers.question}\n\n"
        f"**Ground truth items inserted into the codebase:**\n{items_desc}\n\n"
        f"**Expected correct answer:** {answer_key.expected_answers.correctness}\n"
        f"**Expected complete answer:** {answer_key.expected_answers.completeness}\n\n"
        f"**Agent's response:**\n{agent_response}\n\n"
        f"**Scoring criteria:**\n{criteria_desc}\n\n"
        f"Score each criterion and return ONLY valid JSON."
    )

    options = ClaudeAgentOptions(
        model=JUDGE_MODEL,
        system_prompt=system_prompt,
        allowed_tools=[],
        permission_mode="acceptEdits",
        max_turns=1,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response_text = ""
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                elif isinstance(message, ResultMessage):
                    break
            break  # success
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            print(f"  Rate limited (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {backoff}s: {e}")
            await asyncio.sleep(backoff)

    # Extract JSON from response (handle markdown code blocks)
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Return zero scores if judge returned non-JSON
        weight_map = {c.criterion: c.weight for c in answer_key.rubric_criteria}
        return [
            ScoreEntry(criterion=c.criterion, score=0.0, weight=c.weight, reasoning="Judge returned invalid JSON")
            for c in answer_key.rubric_criteria
        ]

    weight_map = {c.criterion: c.weight for c in answer_key.rubric_criteria}
    return [
        ScoreEntry(
            criterion=s["criterion"], score=s["score"],
            weight=weight_map.get(s["criterion"], 1.0), reasoning=s["reasoning"],
        )
        for s in result["scores"]
    ]
