from __future__ import annotations
import json
from agent_retrieval.generator.llm_client import get_llm_client
from agent_retrieval.schema.answer_key import AnswerKey
from agent_retrieval.schema.verdict import ScoreEntry

async def score_response(agent_response: str, answer_key: AnswerKey, judge_model: str) -> list[ScoreEntry]:
    criteria_desc = "\n".join(f"- {c.criterion} (weight: {c.weight})" for c in answer_key.rubric_criteria)
    items_desc = "\n".join(f"- {it.item_id}: '{it.inserted_text}' at {it.file_path}" for it in answer_key.items)

    prompt = (
        f"You are a strict judge evaluating an AI agent's response to a retrieval question.\n\n"
        f"**Question:** {answer_key.expected_answers.question}\n\n"
        f"**Ground truth items inserted into the codebase:**\n{items_desc}\n\n"
        f"**Expected correct answer:** {answer_key.expected_answers.correctness}\n"
        f"**Expected complete answer:** {answer_key.expected_answers.completeness}\n\n"
        f"**Agent's response:**\n{agent_response}\n\n"
        f"**Scoring criteria:**\n{criteria_desc}\n\n"
        f"Score each criterion from 0.0 to 1.0. Return JSON: "
        f'{{\"scores\": [{{\"criterion\": \"...\", \"score\": 0.0-1.0, \"reasoning\": \"...\"}}]}}\n'
        f"Return ONLY valid JSON."
    )

    client = get_llm_client()
    response = await client.messages.create(model=judge_model, max_tokens=2048,
                                             messages=[{"role": "user", "content": prompt}])
    result = json.loads(response.content[0].text)
    weight_map = {c.criterion: c.weight for c in answer_key.rubric_criteria}
    return [
        ScoreEntry(criterion=s["criterion"], score=s["score"],
                   weight=weight_map.get(s["criterion"], 1.0), reasoning=s["reasoning"])
        for s in result["scores"]
    ]
