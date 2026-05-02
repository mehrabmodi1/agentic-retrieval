from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_retrieval.generator.fixed_pool import sample_fixed_pool, sample_fixed_pool_l3
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def _format_facts_block(items: list[dict[str, Any]]) -> str:
    """Format selected facts as a numbered list for inclusion in the prompt."""
    return "\n".join(f"{i}. {item['text']}" for i, item in enumerate(items, start=1))


def _format_world_state_block(world_state: dict[str, Any]) -> str:
    """Render a world-state dict as a bulleted block."""
    return "\n".join(f"- {k} = {v}" for k, v in world_state.items())


def _to_comparable(value: str) -> float:
    """Parse a bound_value string into a comparable float.

    Handles plain numeric strings ('300', '1.5') and 12-hour time-of-day
    strings ('7:30 PM', '12:15 AM') — the latter converted to minutes
    since midnight.
    """
    s = value.strip()
    try:
        return float(s)
    except ValueError:
        pass
    s_upper = s.upper()
    is_pm = "PM" in s_upper
    is_am = "AM" in s_upper
    if not (is_pm or is_am):
        raise ValueError(f"unrecognised bound_value: {value!r}")
    s_clean = s_upper.replace("PM", "").replace("AM", "").strip()
    if ":" in s_clean:
        h_str, m_str = s_clean.split(":", 1)
        h = int(h_str)
        m = int(m_str)
    else:
        h = int(s_clean)
        m = 0
    if is_pm and h != 12:
        h += 12
    elif is_am and h == 12:
        h = 0
    return float(h * 60 + m)


def _compute_expected_endpoints(items: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    """Compute (max_lower_value, max_lower_text, min_upper_value, min_upper_text).

    For L3 callers should pre-filter to live items.
    """
    lowers = [it for it in items if it["bound_direction"] == "lower"]
    uppers = [it for it in items if it["bound_direction"] == "upper"]
    if not lowers or not uppers:
        return "", "", "", ""
    max_lower = max(lowers, key=lambda it: _to_comparable(it["bound_value"]))
    min_upper = min(uppers, key=lambda it: _to_comparable(it["bound_value"]))
    return (
        max_lower["bound_value"],
        max_lower["text"],
        min_upper["bound_value"],
        min_upper["text"],
    )


def _build_ak_item(idx: int, it: dict[str, Any]) -> dict[str, Any]:
    """Build a single AK item dict from a pool item, including any extra metadata fields."""
    ak_item = {
        "item_id": f"target_{idx:03d}",
        "inserted_text": it["text"],
        "value": it["bound_value"],
        "bound_direction": it["bound_direction"],
        "context_summary": it.get("context_summary", ""),
    }
    # Carry through L2/L3 metadata if present
    for extra in ("variant_id", "variant_text", "gate_clause", "gate_world_var", "live"):
        if extra in it:
            ak_item[extra] = it[extra]
    return ak_item


def generate_pure_reasoning_cell(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
    world_state: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Generate one pure_reasoning / pure_reasoning_l2 / pure_reasoning_l3 answer key.

    No corpus, no LLM calls — purely deterministic.

    For L3, `world_state` must be provided as a dict keyed by content_profile.
    """
    if answer_key_path.exists():
        return

    profile = parametrisation.content_profile
    pool = template.fixed_pool.get(profile, [])
    if not pool:
        raise ValueError(f"fixed_pool is empty for profile {profile}")

    n = parametrisation.n_items or 1
    is_l3 = template.experiment_type == "pure_reasoning_l3"

    if is_l3:
        if world_state is None or profile not in world_state:
            raise ValueError(f"world_state required for pure_reasoning_l3, profile {profile}")
        selected = sample_fixed_pool_l3(
            pool, n=n, parametrisation_id=parametrisation.parametrisation_id,
        )
    else:
        selected = sample_fixed_pool(
            pool, n=n, parametrisation_id=parametrisation.parametrisation_id,
            balance_key="bound_direction",
        )

    profile_examples = template.question_examples.get(profile, {})
    example = profile_examples.get("hard_contextual")
    if example is None:
        raise ValueError(f"No 'hard_contextual' example for profile {profile}")

    facts_block = _format_facts_block(selected)
    question = example.question.replace("{n}", str(n)).replace("{facts_block}", facts_block)
    if is_l3:
        ws_block = _format_world_state_block(world_state[profile])
        question = question.replace("{world_state_block}", ws_block)

    items_yaml = [_build_ak_item(i, it) for i, it in enumerate(selected, start=1)]

    # For L3, compute endpoints from live items only.
    endpoint_items = [it for it in selected if it.get("live", True)]
    max_lower_val, max_lower_text, min_upper_val, min_upper_text = _compute_expected_endpoints(endpoint_items)

    rubric_yaml = [{"criterion": c.criterion, "weight": c.weight} for c in template.rubric_criteria]

    if is_l3:
        correctness = (
            f"The answer must derive the narrowest valid window from the *live* "
            f"(applicable) facts only. A fact is live iff its precondition is "
            f"satisfied by the world state given in the prompt. The lower endpoint "
            f"is established by the live fact whose bound_direction is 'lower' and "
            f"bound_value is largest: {max_lower_text!r} (value: {max_lower_val}). "
            f"The upper endpoint is established by the live fact whose "
            f"bound_direction is 'upper' and bound_value is smallest: "
            f"{min_upper_text!r} (value: {min_upper_val}). The agent must cite both "
            f"facts. The agent's cited evidence must come from the {n} facts in the "
            f"question. Citing a dead (precondition-not-satisfied) fact toward the "
            f"endpoints is incorrect. Citing a fact not present in the question is a "
            f"HALLUCINATION."
        )
        completeness = (
            f"The agent must classify each of the {n} facts as live/dead per the "
            f"world state, and (for live facts) as lower or upper bound consistent "
            f"with the answer-key bound_direction field. Misclassifying a live "
            f"fact as dead, or treating a dead fact as binding, reduces "
            f"classification_accuracy. Hallucinated facts also reduce it."
        )
    else:
        correctness = (
            f"The answer must derive the narrowest valid window. The lower endpoint "
            f"is established by the fact whose bound_direction is 'lower' and "
            f"bound_value is largest: {max_lower_text!r} (value: {max_lower_val}). "
            f"The upper endpoint is established by the fact whose bound_direction "
            f"is 'upper' and bound_value is smallest: {min_upper_text!r} "
            f"(value: {min_upper_val}). The agent must cite both facts. The "
            f"agent's cited evidence must come from the {n} facts provided in the "
            f"question. Any cited fact that is not verbatim one of the {n} provided "
            f"facts is a HALLUCINATION and must be penalized under the "
            f"endpoint_correctness criterion."
        )
        completeness = (
            f"The agent must classify each of the {n} facts as a lower or upper "
            f"bound consistent with the answer-key bound_direction field, OR "
            f"justify any deviation in the reasoning. Classifications attached to "
            f"facts not actually present in the question prompt are hallucinations "
            f"and reduce classification_accuracy."
        )

    parameters = {"content_profile": profile, "n_items": n}

    ak_dict = {
        "parametrisation_id": parametrisation.parametrisation_id,
        "experiment_type": template.experiment_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "items": items_yaml,
        "expected_answers": {
            "question": question,
            "correctness": correctness,
            "completeness": completeness,
        },
        "rubric_criteria": rubric_yaml,
    }

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answer_key_path, "w") as f:
        yaml.dump(ak_dict, f, default_flow_style=False, sort_keys=False)
