from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    ToolUseBlock,
    query,
)

from agent_retrieval.generator.fixed_pool import sample_fixed_pool
from agent_retrieval.generator.insertion import (
    InsertionStats,
    _read_target_fragments,
    _select_target_files,
)
from agent_retrieval.schema.experiment import PAYLOAD_INSERTION_MODEL_MULTI
from agent_retrieval.schema.template import ExperimentTemplate, Parametrisation


def build_fixed_insertion_prompt(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    selected_items: list[dict[str, Any]],
    target_files_content: str,
    answer_key_path: Path,
) -> str:
    """Build a system prompt for inserting pre-authored needles.

    Differs from build_insertion_prompt in that the needles are fixed —
    the agent's only job is to find good insertion sites and report
    file_path + line_range per item.  The agent writes a simple JSON
    locations file (not the full YAML answer key); the driver writes the
    schema-valid AK after the SDK call returns.
    """
    n_items = parametrisation.n_items or len(selected_items)
    needles_block = "\n".join(
        f"{i + 1}. inserted_text: {item['inserted_text']!r}\n"
        f"   value: {item['value']!r}\n"
        f"   content_hint: {item.get('content_hint', '')!r}"
        for i, item in enumerate(selected_items)
    )
    locations_path = answer_key_path.with_suffix(".locations.json")

    return (
        f"You are inserting {n_items} pre-authored needles into a corpus for a "
        f"retrieval experiment. The needles are FIXED — do not invent new ones, "
        f"do not modify their text, do not paraphrase. Insert each verbatim.\n\n"
        f"Experiment type: multi_retrieval\n"
        f"Number of items: {n_items}\n\n"
        f"## Needles to insert (verbatim, no modifications)\n{needles_block}\n\n"
        f"## Target fragments (pre-selected — use these)\n{target_files_content}\n\n"
        f"## Instructions\n"
        f"Each fragment above is a 30-line window from a corpus file, with its "
        f"start_line offset. Do NOT read or browse any files. Work only with the "
        f"fragments provided above.\n\n"
        f"1. For each of the {n_items} needles, choose a target fragment and an "
        f"insertion line within that fragment where the needle reads naturally.\n"
        f"2. Use the Edit tool to insert each needle VERBATIM (same text, same "
        f"capitalisation, same operators, same spacing).\n"
        f"3. Write a JSON file to {locations_path.resolve()} that records where "
        f"each needle was inserted. This is your ONLY output file — do NOT write "
        f"a YAML answer key.\n\n"
        f"IMPORTANT: Batch ALL Edit and Write tool calls into a single response. "
        f"Do not use multiple turns.\n\n"
        f"## locations.json schema\n"
        f"The file must be a JSON array with exactly {n_items} objects, one per "
        f"needle in order (needle 1 → target_001, etc.):\n"
        f"```json\n"
        f"[\n"
        f"  {{\"item_id\": \"target_001\", \"file_path\": \"<relative path>\", "
        f"\"line_range\": [<start>, <end>], \"context_summary\": \"<one sentence>\"}},\n"
        f"  {{\"item_id\": \"target_002\", \"file_path\": \"<relative path>\", "
        f"\"line_range\": [<start>, <end>], \"context_summary\": \"<one sentence>\"}}\n"
        f"  // ... through target_{n_items:03d}\n"
        f"]\n"
        f"```\n"
    )


def write_fixed_pool_answer_key(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    items_with_locations: list[dict[str, Any]],
    answer_key_path: Path,
) -> None:
    """Write a complete answer key for a multi_retrieval cell.

    items_with_locations includes the fixed-pool fields (inserted_text, value)
    plus the insertion-site fields (file_path, line_range, context_summary).
    """
    n = len(items_with_locations)
    profile = parametrisation.content_profile
    profile_examples = template.question_examples.get(profile, {})
    example = profile_examples.get("hard_contextual")
    if example is None:
        raise ValueError(f"No 'hard_contextual' example for profile {profile}")
    question = example.question.replace("{n}", str(n))

    items_yaml = []
    for i, it in enumerate(items_with_locations, start=1):
        items_yaml.append({
            "item_id": f"target_{i:03d}",
            "inserted_text": it["inserted_text"],
            "value": it["value"],
            "file_path": it.get("file_path"),
            "line_range": it.get("line_range"),
            "context_summary": it.get("context_summary", ""),
        })

    parameters: dict[str, Any] = {"content_profile": profile}
    if parametrisation.corpus_token_count is not None:
        parameters["corpus_token_count"] = parametrisation.corpus_token_count
    if parametrisation.discriminability is not None:
        parameters["discriminability"] = parametrisation.discriminability
    if parametrisation.reference_clarity is not None:
        parameters["reference_clarity"] = parametrisation.reference_clarity
    if parametrisation.n_items is not None:
        parameters["n_items"] = parametrisation.n_items

    rubric_yaml = [
        {"criterion": c.criterion, "weight": c.weight}
        for c in template.rubric_criteria
    ]

    ak_dict = {
        "parametrisation_id": parametrisation.parametrisation_id,
        "experiment_type": parametrisation.experiment_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "items": items_yaml,
        "expected_answers": {
            "question": question,
            "correctness": (
                f"Each of the {n} items must be reported with both the verbatim "
                f"inserted_text and the corresponding value, matching the "
                f"answer-key items."
            ),
            "completeness": (
                f"All {n} items must be reported. Recall is fraction of items "
                f"correctly reported."
            ),
        },
        "rubric_criteria": rubric_yaml,
    }

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(answer_key_path, "w") as f:
        yaml.dump(ak_dict, f, default_flow_style=False, sort_keys=False)


async def insert_fixed_payloads(
    template: ExperimentTemplate,
    parametrisation: Parametrisation,
    corpus_dir: Path,
    answer_key_path: Path,
) -> InsertionStats | None:
    """Insertion driver for multi_retrieval: use fixed pool, not LLM-invented needles."""
    if answer_key_path.exists():
        return None

    answer_key_path.parent.mkdir(parents=True, exist_ok=True)

    pool = template.fixed_pool.get(parametrisation.content_profile, [])
    if not pool:
        raise ValueError(
            f"fixed_pool is empty for profile {parametrisation.content_profile}"
        )
    n_items = parametrisation.n_items or 1
    selected = sample_fixed_pool(pool, n=n_items, parametrisation_id=parametrisation.parametrisation_id)

    n_target_files = max(n_items * 2, 4)
    target_files = _select_target_files(corpus_dir, parametrisation, n_target_files)
    base_seed = hash(parametrisation.parametrisation_id) ^ 0xDEAD
    target_files_content = _read_target_fragments(target_files, corpus_dir, base_seed)

    system_prompt = build_fixed_insertion_prompt(
        template=template,
        parametrisation=parametrisation,
        selected_items=selected,
        target_files_content=target_files_content,
        answer_key_path=answer_key_path,
    )

    options = ClaudeAgentOptions(
        model=PAYLOAD_INSERTION_MODEL_MULTI,
        system_prompt=system_prompt,
        cwd=str(corpus_dir.resolve()),
        allowed_tools=["Edit", "Write"],
        permission_mode="acceptEdits",
        max_turns=max(n_items * 3, 10),
    )

    prompt = (
        "Insert the pre-authored needles into the provided fragments and write "
        "the answer key. Do not invent new needles; do not modify the provided "
        "needle text. Batch all Edit and Write calls into a single response."
    )

    stats = InsertionStats(model=PAYLOAD_INSERTION_MODEL_MULTI)

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            if message.usage:
                stats.input_tokens += message.usage.get("input_tokens", 0)
                stats.output_tokens += message.usage.get("output_tokens", 0)
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    stats.tool_calls.append(block.name)
        elif isinstance(message, ResultMessage):
            stats.num_turns = message.num_turns
            stats.duration_ms = message.duration_ms
            stats.total_cost_usd = message.total_cost_usd or 0.0
            stats.is_error = message.is_error
            stats.errors = message.errors or []
            break

    # Driver writes the schema-valid AK from the agent-produced locations file.
    locations_path = answer_key_path.with_suffix(".locations.json")
    locations_ok = locations_path.exists()
    if locations_ok:
        try:
            locations = json.loads(locations_path.read_text())
            if len(locations) < n_items:
                locations_ok = False
                stats.errors.append(
                    f"locations.json has {len(locations)} entries, expected {n_items}"
                )
        except Exception as exc:
            locations_ok = False
            stats.errors.append(f"failed to parse locations.json: {exc}")

    if locations_ok:
        items_with_locations = [
            {**selected[i], **locations[i]}
            for i in range(n_items)
        ]
        write_fixed_pool_answer_key(template, parametrisation, items_with_locations, answer_key_path)
        locations_path.unlink(missing_ok=True)

    stats.answer_key_written = answer_key_path.exists()
    if not stats.answer_key_written and stats.tool_calls:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        stats.errors.append("rolled back corpus: edits applied but no answer key")

    return stats
