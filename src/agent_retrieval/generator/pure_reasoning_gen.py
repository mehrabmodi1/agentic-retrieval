from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agent_retrieval.generator.fixed_pool import sample_fixed_pool
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def _format_facts_block(items: list[dict[str, Any]]) -> str:
    """Format selected facts as a numbered list for inclusion in the prompt."""
    return "\n".join(f"{i}. {item['text']}" for i, item in enumerate(items, start=1))


def _to_comparable(value: str) -> float:
    """Parse a bound_value string into a comparable float.

    Handles plain numeric strings ('300', '1.5') and 12-hour time-of-day
    strings ('7:30 PM', '12:15 AM') — the latter converted to minutes
    since midnight (so '7:30 PM' → 19*60 + 30 = 1170.0).
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

    bound_value is treated as a string here. Numeric comparison is left to the
    judge; the answer key records the chosen items so the judge can verify.
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


def generate_pure_reasoning_cell(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    answer_key_path: Path,
) -> None:
    """Generate one pure_reasoning answer key. No corpus, no LLM calls."""
    if answer_key_path.exists():
        return

    profile = parametrisation.content_profile
    pool = template.fixed_pool.get(profile, [])
    if not pool:
        raise ValueError(f"fixed_pool is empty for profile {profile}")

    n = parametrisation.n_items or 1
    selected = sample_fixed_pool(
        pool,
        n=n,
        parametrisation_id=parametrisation.parametrisation_id,
        balance_key="bound_direction",
    )

    profile_examples = template.question_examples.get(profile, {})
    example = profile_examples.get("hard_contextual")
    if example is None:
        raise ValueError(f"No 'hard_contextual' example for profile {profile}")

    facts_block = _format_facts_block(selected)
    question = example.question.replace("{n}", str(n)).replace("{facts_block}", facts_block)

    items_yaml = []
    for i, it in enumerate(selected, start=1):
        items_yaml.append({
            "item_id": f"target_{i:03d}",
            "inserted_text": it["text"],
            "value": it["bound_value"],
            "bound_direction": it["bound_direction"],
            "context_summary": it.get("context_summary", ""),
        })

    max_lower_val, max_lower_text, min_upper_val, min_upper_text = _compute_expected_endpoints(selected)

    rubric_yaml = [
        {"criterion": c.criterion, "weight": c.weight}
        for c in template.rubric_criteria
    ]

    ak_dict = {
        "parametrisation_id": parametrisation.parametrisation_id,
        "experiment_type": "pure_reasoning",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {"content_profile": profile, "n_items": n},
        "items": items_yaml,
        "expected_answers": {
            "question": question,
            "correctness": (
                f"The answer must derive the narrowest valid window. The lower "
                f"endpoint is established by the fact whose bound_direction is "
                f"'lower' and bound_value is largest: {max_lower_text!r} "
                f"(value: {max_lower_val}). The upper endpoint is established "
                f"by the fact whose bound_direction is 'upper' and bound_value "
                f"is smallest: {min_upper_text!r} (value: {min_upper_val}). "
                f"The agent must cite both facts."
            ),
            "completeness": (
                f"The agent must classify each of the {n} facts as a lower or "
                f"upper bound consistent with the answer-key bound_direction "
                f"field, OR justify any deviation in the reasoning."
            ),
        },
        "rubric_criteria": rubric_yaml,
    }

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answer_key_path, "w") as f:
        yaml.dump(ak_dict, f, default_flow_style=False, sort_keys=False)
